"""Per-tenant conversation state: the durable multi-turn context, kept in MySQL.

A conversation is reached through tenant + user + optional session id, so a
caller can never read another tenant's or another user's turns. The state only
feeds follow-up handling: any store failure degrades to a stateless single-turn
answer, never to an error. Without a store (``MYSQL_URL`` unset) chat stays
single turn.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import unquote, urlsplit

from . import config
from .auth import Principal
from .errors import AppError
from .messages import BUSY_MSG
from .observability import metric
from .runtime import get_runtime
from .synonyms import synonym_terms

SESSION_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
# A follow-up either continues the previous question ("and for children?") or is
# too short to carry a topic of its own; only then is the previous question
# reused for retrieval. A long, self-contained question is searched as asked.
CONTINUATION_RE = re.compile(
    r"^(and|also|or|then|plus|but|what about|how about|what if|same|those|them|it|that)\b",
    re.IGNORECASE,
)
# "Is a 7MB PDF within the documented limit here?" is long enough to look
# self-contained but names no topic of its own; "here" means the previous turn.
# A demonstrative standing at the end of a clause ("was that included?") works
# the same way, while "this platform's refund policy" names its own topic and
# must not turn every such question into a follow-up.
REFERENCE_RE = re.compile(
    r"\b(?:here|there)\b|\b(?:this|that|these|those|same)\s*(?:[?.!]|$)",
    re.IGNORECASE,
)
# A repair ("actually it is for my son", "no, make that for me") or a request to
# continue ("finish the search first") refers to the previous request and names
# no topic of its own. The model can rewrite it, but when the model is slow or
# rate limited the deterministic path must still search the same subject.
FOLLOWUP_CUE_RE = re.compile(
    r"^\s*(?:actually|sorry|no|wait|instead|correction|to\s+be\s+clear|make\s+that|i\s+mean)\b"
    r"|\bfor\s+(?:my|his|her|our|their)\s+\w+"
    r"|\bfor\s+(?:me|myself|someone\s+else)\b"
    r"|\b(?:finish|continue|go\s+on|keep\s+going)\b",
    re.IGNORECASE,
)
# A correction supersedes something said earlier, including who a request is for
# ("actually it is for my son", "no, make that for me instead"). "Finish the
# search" and "also for children" refer back without replacing anything, so the
# lead-in, not the backward reference, is what makes a message a correction.
CORRECTION_RE = re.compile(
    r"^\s*(?:actually|no|wait|sorry|correction|to\s+be\s+clear|i\s+mean|make\s+that|instead)\b",
    re.IGNORECASE,
)
# "Actually, book it for me." names a product flow, but it corrects who the
# previous request is for instead of starting a new one. The anaphor and the
# beneficiary separate it from "Actually, how do I book a doctor?", which names
# its own flow.
BENEFICIARY_RE = re.compile(
    r"\bfor\s+(?:me|myself|someone\s+else|(?:my|his|her|our|their)\s+\w+)\b",
    re.IGNORECASE,
)
ANAPHOR_RE = re.compile(r"\b(?:it|that|this|these|those)\b", re.IGNORECASE)
# Who a booking or request is for. A correction replaces the earlier
# beneficiary ("actually it is for my son" then "no, make that for me"), so the
# answer must use only the newest one, even when the documents describe the
# "someone else" flow.
SELF_BENEFICIARY_RE = re.compile(r"\bfor\s+(?:me|myself)\b", re.IGNORECASE)
OTHER_BENEFICIARY_RE = re.compile(r"\bfor\s+someone\s+else\b", re.IGNORECASE)
RELATIVE_BENEFICIARY_RE = re.compile(r"\bfor\s+(?:my|his|her|our|their)\s+(\w+)\b", re.IGNORECASE)
# Words that carry no topic of their own. A short question borrows the previous
# one only when everything it adds is filler or already there; "how do I reset
# my password?" after a booking question names a new topic and is searched as
# asked, while "and for the app?" adds nothing the earlier question lacked.
TOPIC_STOPWORDS = frozenset(
    """how what where when why who whom whose which is are was were be been being do does did done
    doing can could would should will shall may might must i me my mine we us our ours you your yours
    he him his she her hers it its they them their this that these those a an the to for in on at of
    with about from by as and or but if then than so also please tell show give find get need want
    have has had there here again same just now any some all more most much many own am not no""".split()
)
# Product-flow words name a topic even when the synonym catalogue has nothing
# to add: "Now I am attaching during booking." is the context for "is a 7MB PDF
# within the limit here?". Without this, an older question would be borrowed.
FLOW_RE = re.compile(
    r"\b(?:book\w*|appointments?|visits?|schedul\w+|reschedul\w+|cancel\w*|"
    r"attach\w*|upload\w*|files?|documents?|pdfs?|lab\s+reports?|settings?|"
    r"plans?|insurance|deductibles?|co-?pays?|premiums?)\b",
    re.IGNORECASE,
)
# People correct themselves inline: "actually, i live in boston", "sorry, i
# mean i work in boston". The lead-in does not change what the rest of the
# sentence is, so it is skipped before the statement patterns.
CORRECTION_PREFIX = (
    r"(?:(?:actually|sorry|correction|to\s+be\s+clear|rather|instead|no|wait)\b[\s,:]*"
    r"|i\s+mean\b[\s,:]*)*"
)
# A statement about the caller (name, location, role) is answered from the
# conversation: there is nothing in the corpus to retrieve for it. "i m lucas"
# and "im lucas" are how people actually type it, so both count; "am Patient A"
# is the same sentence with the leading "i" lost.
SELF_DISCLOSURE_RE = re.compile(
    rf"^{CORRECTION_PREFIX}"
    r"(?:i'?m|i\s+m|i am|am\s+(?:a\s+|an\s+)?patient\b|my name is|call me|i live|i work|"
    r"i\s+now\s+(?:live|work|reside)|i\s+(?:have\s+)?moved\b|i'?ve\s+moved\b)\b",
    re.IGNORECASE,
)
# Only clear identity or origin statements override retrieved documents. The
# broad disclosure pattern above also covers every "i'm ...", and "i'm trying
# to book an appointment" is a question for the corpus, not a fact about the
# caller, so forcing those from history would lose the documents.
PERSONAL_STATEMENT_RE = re.compile(
    rf"^{CORRECTION_PREFIX}(?:"
    r"(?:i'?m|i\s+m|i am)\s+(?:now\s+)?(?:from|based\s+in|located\s+in)\b"
    r"|(?:my\s+name\s+is|call\s+me|i'?m\s+called)\b"
    r"|i\s+(?:live|reside|stay|work)\b"
    r"|i\s+now\s+(?:live|reside|work)\b"
    r"|i\s+(?:have\s+)?moved\s+(?:to|from)\b"
    r"|i'?ve\s+moved\s+(?:to|from)\b"
    r")",
    re.IGNORECASE,
)
# A question about the conversation itself ("summarize what I said") is answered
# from the stored turns: the corpus cannot summarize the caller's own words.
CONVERSATION_RE = re.compile(
    r"\b(?:summari[sz]e|recap|repeat|restate)\b.*\b(?:conversation|chat|what (?:i|we)|(?:my|our) \w+)\b"
    r"|\bwhat (?:i|we) (?:said|asked|mentioned)\b.*\b(?:summari[sz]e|recap|repeat|restate)\b"
    r"|\bwhat (?:did|have) i (?:say|said|ask|asked|mention|mentioned|tell|told)\b"
    r"|\bwhat (?:have|did) we (?:discuss(?:ed)?|talk(?:ed)? about|say|said)\b"
    r"|\b(?:my|our) (?:last|previous|earlier) (?:question|message)\b"
    r"|\bdo you (?:remember|recall)\b",
    re.IGNORECASE,
)

# Columns are bounded the same way the prompt is: one turn is never huge.
MAX_CONTENT_CHARS = 2000

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS conversation_turn (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  tenant_id VARCHAR(64) NOT NULL,
  subject VARCHAR(128) NOT NULL,
  session_id VARCHAR(64) NOT NULL,
  speaker ENUM('user', 'assistant') NOT NULL,
  content VARCHAR({MAX_CONTENT_CHARS}) NOT NULL,
  created_at BIGINT UNSIGNED NOT NULL,
  PRIMARY KEY (id),
  KEY conversation (tenant_id, subject, session_id, id),
  KEY created (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""

_SWEEP_SECONDS = 60  # retention is enforced in the background, not on every read
_SWEEP_LIMIT = 1000  # rows deleted per sweep; keeps the write path short
_last_sweep = 0.0


@dataclass(frozen=True)
class Turn:
    role: str  # "user" or "assistant"
    text: str


def _now() -> int:
    return int(time.time())


def _session(session_id: str | None) -> str:
    if session_id is None:
        return "default"
    if not SESSION_RE.match(session_id):
        raise ValueError("invalid session id")
    return session_id


def conversation(principal: Principal, session_id: str | None = None) -> tuple[str, str, str]:
    """Tenant, hashed role + subject and session: state cannot leak across any of them.

    The role is part of the digest because one subject can hold a patient and a
    provider identity; their conversations must never be merged.
    """
    subject = hashlib.sha256(f"{principal.role}:{principal.subject}".encode()).hexdigest()[:16]
    return principal.tenant_id, subject, _session(session_id)


def _pool() -> Any | None:
    try:
        return get_runtime().mysql
    except RuntimeError:
        return None


def _clip(text: str) -> str:
    return text.strip()[: min(config.HISTORY_MAX_CHARS, MAX_CONTENT_CHARS)]


def _dsn(url: str | None = None, timeout: float | None = None) -> dict[str, Any]:
    parts = urlsplit(url if url is not None else config.MYSQL_URL)
    return {
        "host": parts.hostname or "localhost",
        "port": parts.port or 3306,
        "user": unquote(parts.username or "root"),
        "password": unquote(parts.password or ""),
        "db": parts.path.lstrip("/") or None,
        "charset": "utf8mb4",
        "autocommit": True,  # every statement here is independent
        "minsize": 1,
        "maxsize": max(config.MYSQL_POOL_SIZE, 1),
        "connect_timeout": max(int(timeout if timeout is not None else config.MYSQL_TIMEOUT), 1),
        "pool_recycle": 3600,  # survive a MySQL restart without stale sockets
    }


async def open_store() -> Any | None:
    """Open the conversation pool and create its table; None keeps chat stateless."""
    if not config.MYSQL_URL or not config.HISTORY_ENABLED or config.HISTORY_TURNS <= 0:
        return None
    import aiomysql  # optional dependency: only imported when a store is configured

    pool = await aiomysql.create_pool(**_dsn(config.MYSQL_URL, config.MYSQL_TIMEOUT))
    try:
        async with pool.acquire() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(SCHEMA)
    except BaseException:
        pool.close()
        await pool.wait_closed()
        raise
    return pool


async def open_provider_store() -> Any | None:
    """Open the provider operational DB pool (read-only Text-to-SQL).

    The tables are owned by the platform, so this never creates schema. None
    when PROVIDER_DB_URL is unset; Text-to-SQL is then unavailable (the analytics
    API or a marked mock answers instead) rather than sharing another pool.
    """
    if not config.PROVIDER_DB_URL:
        return None
    import aiomysql  # optional dependency: only imported when a store is configured

    return await aiomysql.create_pool(**_dsn(config.PROVIDER_DB_URL, config.PROVIDER_DB_TIMEOUT))


async def close_store(pool: Any) -> None:
    pool.close()
    await pool.wait_closed()


async def load_history(principal: Principal, session_id: str | None = None) -> list[Turn]:
    if not config.HISTORY_ENABLED or config.HISTORY_TURNS <= 0:
        return []
    pool = _pool()
    if pool is None:
        return []
    tenant, subject, session = conversation(principal, session_id)
    try:
        async with asyncio.timeout(config.MYSQL_TIMEOUT):
            async with pool.acquire() as connection:
                async with connection.cursor() as cursor:
                    await cursor.execute(
                        "SELECT speaker, content FROM conversation_turn"
                        " WHERE tenant_id = %s AND subject = %s AND session_id = %s AND created_at >= %s"
                        " ORDER BY id DESC LIMIT %s",
                        (tenant, subject, session, _now() - config.HISTORY_TTL, config.HISTORY_TURNS),
                    )
                    rows = await cursor.fetchall()
    except Exception as exc:
        metric(history_error=type(exc).__name__)
        return []
    return [
        Turn(speaker, content)
        for speaker, content in reversed(rows)
        if speaker in ("user", "assistant") and isinstance(content, str)
    ]


async def append_turns(principal: Principal, session_id: str | None, question: str, answer: str) -> None:
    """Record one exchange; called after the answer has been streamed."""
    if not config.HISTORY_ENABLED or config.HISTORY_TURNS <= 0:
        return
    pool = _pool()
    if pool is None:
        return
    tenant, subject, session = conversation(principal, session_id)
    now = _now()
    sweep = _sweep_due(now)
    try:
        async with asyncio.timeout(config.MYSQL_TIMEOUT):
            async with pool.acquire() as connection:
                async with connection.cursor() as cursor:
                    await cursor.executemany(
                        "INSERT INTO conversation_turn"
                        " (tenant_id, subject, session_id, speaker, content, created_at)"
                        " VALUES (%s, %s, %s, %s, %s, %s)",
                        [
                            (tenant, subject, session, "user", _clip(question), now),
                            (tenant, subject, session, "assistant", _clip(answer), now),
                        ],
                    )
                    # Keep only the newest turns of this conversation.
                    await cursor.execute(
                        "DELETE FROM conversation_turn"
                        " WHERE tenant_id = %s AND subject = %s AND session_id = %s"
                        "   AND id < (SELECT boundary FROM ("
                        "     SELECT id AS boundary FROM conversation_turn"
                        "      WHERE tenant_id = %s AND subject = %s AND session_id = %s"
                        "      ORDER BY id DESC LIMIT 1 OFFSET %s"
                        "   ) AS newest)",
                        (
                            tenant,
                            subject,
                            session,
                            tenant,
                            subject,
                            session,
                            max(config.HISTORY_TURNS - 1, 0),
                        ),
                    )
                    if sweep:
                        # Also forget conversations that were simply abandoned.
                        await cursor.execute(
                            "DELETE FROM conversation_turn WHERE created_at < %s LIMIT %s",
                            (now - config.HISTORY_TTL, _SWEEP_LIMIT),
                        )
    except Exception as exc:
        metric(history_write_error=type(exc).__name__)


def _sweep_due(now: int) -> bool:
    global _last_sweep
    if now - _last_sweep < _SWEEP_SECONDS:
        return False
    _last_sweep = now
    return True


async def clear_history(principal: Principal, session_id: str | None = None) -> bool:
    """Delete the caller's turns; returns False when history is not stored at all.

    A reset that ran against a configured store must have happened, so a store
    failure raises instead of pretending the conversation is gone.
    """
    pool = _pool()
    if pool is None:
        return False  # nothing is stored, so there is nothing to forget
    tenant, subject, session = conversation(principal, session_id)
    try:
        async with asyncio.timeout(config.MYSQL_TIMEOUT):
            async with pool.acquire() as connection:
                async with connection.cursor() as cursor:
                    await cursor.execute(
                        "DELETE FROM conversation_turn WHERE tenant_id = %s AND subject = %s AND session_id = %s",
                        (tenant, subject, session),
                    )
    except Exception as exc:
        metric(history_error=type(exc).__name__)
        raise AppError(503, "busy", BUSY_MSG) from None
    return True


def prompt_turns(history: list[Turn]) -> list[dict[str, str]]:
    return [{"role": turn.role, "content": turn.text} for turn in history]


def last_question(history: list[Turn]) -> str:
    """The newest user turn that names a topic; "summarize" and "i'm lucas" name none.

    A statement such as "I am authorized to manage her care" names no topic
    either, so it is skipped when an earlier turn does; borrowing it would make
    the next follow-up unsearchable.
    """
    fallback = ""
    for turn in reversed(history):
        if turn.role != "user" or is_conversation_question(turn.text) or is_personal_statement(turn.text):
            continue
        if synonym_terms(turn.text) or FLOW_RE.search(turn.text):
            return turn.text
        fallback = fallback or turn.text
    return fallback


def _content_words(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9']+", text.lower())
    return {word for word in words if len(word) > 1 and word not in TOPIC_STOPWORDS}


def is_contextual_followup(text: str, history: list[Turn]) -> bool:
    """A correction, beneficiary change or "continue" that names no topic itself."""
    if not history:
        return False
    if synonym_terms(text):
        return False
    if FOLLOWUP_CUE_RE.search(text) and not FLOW_RE.search(text):
        return True
    return bool(is_correction(text) and BENEFICIARY_RE.search(text) and ANAPHOR_RE.search(text))


def is_correction(text: str) -> bool:
    """True when the message replaces an earlier detail rather than adding one."""
    return bool(CORRECTION_RE.match(text))


def beneficiary(text: str) -> str:
    """Who the latest request is for, or "" when it names nobody."""
    if OTHER_BENEFICIARY_RE.search(text):
        return "someone else"
    if SELF_BENEFICIARY_RE.search(text):
        return "the user themselves"
    match = RELATIVE_BENEFICIARY_RE.search(text)
    return f"the user's {match.group(1)}" if match else ""


def is_followup(text: str, history: list[Turn]) -> bool:
    """A short or explicitly continued question leans on the previous turn."""
    if not history:
        return False
    if CONTINUATION_RE.match(text):
        return True
    if is_contextual_followup(text, history):
        return True
    words = text.split()
    # One or two words cannot stand alone ("children", "and then?").
    if len(words) <= 3:
        return True
    # "Is a 7MB PDF within the documented limit here?" is long enough to look
    # self-contained but names nothing on its own; "here" answers only in the
    # context of the previous turn.
    if REFERENCE_RE.search(text) and not synonym_terms(text):
        return True
    if len(words) <= 6:
        previous = last_question(history)
        return not previous or not (_content_words(text) - _content_words(previous))
    return False


def is_conversation_question(text: str) -> bool:
    """True when the question is about the conversation, not about the corpus."""
    return bool(CONVERSATION_RE.search(text))


# "lucas here" introduces the caller by name. A specialty ("dermatologist
# here") is a search, not a name, so the synonym catalogue has the last word.
NAME_INTRO_RE = re.compile(
    r"^[a-z][a-z'.-]{1,30}\s+(?:here|speaking|talking|writing)\b",
    re.IGNORECASE,
)


def is_name_intro(text: str) -> bool:
    return bool(NAME_INTRO_RE.match(text)) and not synonym_terms(text)


def is_personal_statement(text: str) -> bool:
    """True for statements about the caller that no document can answer."""
    return bool(PERSONAL_STATEMENT_RE.match(text)) or is_name_intro(text)


def retrieval_text(text: str, history: list[Turn]) -> str:
    previous = last_question(history)
    if not previous or not is_followup(text, history):
        return search_only(text)
    return search_only(f"{previous} {text}")


# "Move my visit to Friday and just say done" is a request plus an instruction
# about the reply. The instruction names no topic and pulls retrieval off the
# track, so it is dropped from the text used for search only; the model still
# sees the caller's words.
REPLY_DIRECTIVE_RE = re.compile(
    r"\s*(?:,|and|then)?\s*(?:just|only)\s+(?:say|reply|answer|output|print|write|tell\s+me)\b.*$",
    re.IGNORECASE,
)


def search_only(text: str) -> str:
    return REPLY_DIRECTIVE_RE.sub("", text).strip() or text


def context_scope(history: list[Turn]) -> str:
    """Cache identity for multi-turn answers: reused only for an identical history."""
    if not history:
        return ""
    digest = hashlib.sha256("\n".join(f"{turn.role}:{turn.text}" for turn in history).encode()).hexdigest()
    return digest[:12]
