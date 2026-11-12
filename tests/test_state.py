from __future__ import annotations

from types import SimpleNamespace

import pytest

from chat_search import config, state
from chat_search.auth import Principal
from chat_search.errors import AppError
from chat_search.state import (
    SELF_DISCLOSURE_RE,
    Turn,
    append_turns,
    beneficiary,
    clear_history,
    context_scope,
    conversation,
    is_contextual_followup,
    is_conversation_question,
    is_correction,
    is_followup,
    is_name_intro,
    is_personal_statement,
    load_history,
    retrieval_text,
    search_only,
)


def principal(subject: str = "user-1", tenant: str = "acme") -> Principal:
    return Principal(
        subject=subject,
        role="patient",
        tenant_id=tenant,
        patient_id="patient-1",
        provider_id=None,
        permissions=frozenset({"docs:read"}),
    )


class FakeMySQL:
    """Just enough of an aiomysql pool to exercise the SQL the store issues."""

    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.executed: list[str] = []
        self._next_id = 1

    def acquire(self) -> FakeAcquisition:
        return FakeAcquisition(self)


class FakeAcquisition:
    def __init__(self, db: FakeMySQL) -> None:
        self.db = db

    async def __aenter__(self) -> FakeConnection:
        return FakeConnection(self.db)

    async def __aexit__(self, *_: object) -> bool:
        return False


class FakeConnection:
    def __init__(self, db: FakeMySQL) -> None:
        self.db = db

    def cursor(self) -> FakeCursorAcquisition:
        return FakeCursorAcquisition(self.db)


class FakeCursorAcquisition:
    def __init__(self, db: FakeMySQL) -> None:
        self.db = db

    async def __aenter__(self) -> FakeCursor:
        return FakeCursor(self.db)

    async def __aexit__(self, *_: object) -> bool:
        return False


class FakeCursor:
    def __init__(self, db: FakeMySQL) -> None:
        self.db = db
        self.rows: list[tuple[str, str]] = []

    def _conversation(self, tenant: str, subject: str, session: str) -> list[dict]:
        return [r for r in self.db.rows if (r["tenant"], r["subject"], r["session"]) == (tenant, subject, session)]

    async def execute(self, sql: str, params: tuple = ()) -> None:
        statement = " ".join(sql.split())
        self.db.executed.append(statement)
        if statement.startswith("SELECT speaker"):
            tenant, subject, session, cutoff, limit = params
            matched = [r for r in self._conversation(tenant, subject, session) if r["created"] >= cutoff]
            matched.sort(key=lambda r: r["id"], reverse=True)
            self.rows = [(r["speaker"], r["content"]) for r in matched[:limit]]
        elif statement.startswith("DELETE") and "created_at <" in statement:
            cutoff, limit = params
            stale = [r for r in self.db.rows if r["created"] < cutoff][:limit]
            for row in stale:
                self.db.rows.remove(row)
        elif statement.startswith("DELETE") and "id <" in statement:
            tenant, subject, session, *_rest, offset = params
            newest_first = sorted(self._conversation(tenant, subject, session), key=lambda r: r["id"], reverse=True)
            if len(newest_first) > offset:
                boundary = newest_first[offset]["id"]
                dropped = {r["id"] for r in newest_first if r["id"] < boundary}
                self.db.rows = [r for r in self.db.rows if r["id"] not in dropped]
        elif statement.startswith("DELETE"):
            tenant, subject, session = params
            self.db.rows = [r for r in self.db.rows if r not in self._conversation(tenant, subject, session)]

    async def executemany(self, sql: str, params: list[tuple]) -> None:
        self.db.executed.append(" ".join(sql.split()))
        for tenant, subject, session, speaker, content, created in params:
            self.db.rows.append(
                {
                    "id": self.db._next_id,
                    "tenant": tenant,
                    "subject": subject,
                    "session": session,
                    "speaker": speaker,
                    "content": content,
                    "created": created,
                }
            )
            self.db._next_id += 1

    async def fetchall(self) -> list[tuple[str, str]]:
        return self.rows


def use_store(monkeypatch, db: FakeMySQL) -> FakeMySQL:
    monkeypatch.setattr(state, "get_runtime", lambda: SimpleNamespace(mysql=db))
    monkeypatch.setattr(state, "_last_sweep", 0.0)
    return db


async def test_turns_are_stored_and_loaded_for_one_tenant_and_user(monkeypatch):
    use_store(monkeypatch, FakeMySQL())

    await append_turns(principal(), None, "how do I reset my password?", "From Account > Security.")

    assert await load_history(principal()) == [
        Turn("user", "how do I reset my password?"),
        Turn("assistant", "From Account > Security."),
    ]
    assert await load_history(principal(tenant="other")) == []
    assert await load_history(principal(subject="user-2")) == []
    assert await load_history(principal(), "thread-2") == []


async def test_only_the_last_turns_are_kept(monkeypatch):
    db = use_store(monkeypatch, FakeMySQL())
    monkeypatch.setattr(config, "HISTORY_TURNS", 4)

    for index in range(3):
        await append_turns(principal(), None, f"question {index}", f"answer {index}")

    assert [turn.text for turn in await load_history(principal())] == [
        "question 1",
        "answer 1",
        "question 2",
        "answer 2",
    ]
    assert len(db.rows) == 4  # older turns are deleted, not just hidden


async def test_stored_turns_are_clipped(monkeypatch):
    use_store(monkeypatch, FakeMySQL())
    monkeypatch.setattr(config, "HISTORY_MAX_CHARS", 5)

    await append_turns(principal(), None, "how do I reset my password?", "From Account > Security.")

    assert await load_history(principal()) == [Turn("user", "how d"), Turn("assistant", "From ")]


async def test_a_conversation_is_forgotten_when_idle(monkeypatch):
    db = use_store(monkeypatch, FakeMySQL())
    monkeypatch.setattr(state, "_now", lambda: 1000)
    await append_turns(principal(), None, "question", "answer")

    monkeypatch.setattr(state, "_now", lambda: 1000 + config.HISTORY_TTL + 1)
    assert await load_history(principal()) == []

    # The next write sweeps the expired rows out of the table.
    monkeypatch.setattr(state, "_last_sweep", 0.0)
    await append_turns(principal(), None, "next", "answer")

    assert [row["content"] for row in db.rows] == ["next", "answer"]


async def test_a_broken_store_degrades_to_a_stateless_answer(monkeypatch):
    class Broken:
        def acquire(self):
            raise ConnectionError("mysql down")

    monkeypatch.setattr(state, "get_runtime", lambda: SimpleNamespace(mysql=Broken()))

    assert await load_history(principal()) == []
    await append_turns(principal(), None, "question", "answer")  # must not raise


async def test_history_is_skipped_when_disabled(monkeypatch):
    db = use_store(monkeypatch, FakeMySQL())
    monkeypatch.setattr(config, "HISTORY_ENABLED", False)

    await append_turns(principal(), None, "question", "answer")

    assert db.rows == []
    assert await load_history(principal()) == []


async def test_chat_is_stateless_without_a_store(monkeypatch):
    def no_runtime():
        raise RuntimeError("Runtime is not initialized")

    monkeypatch.setattr(state, "get_runtime", no_runtime)

    assert await load_history(principal()) == []
    await append_turns(principal(), None, "question", "answer")
    # Nothing stored, so nothing was forgotten; callers are told so.
    assert await clear_history(principal()) is False


async def test_clear_forgets_only_that_conversation(monkeypatch):
    db = use_store(monkeypatch, FakeMySQL())
    await append_turns(principal(), "thread-1", "question", "answer")
    await append_turns(principal(), "thread-2", "question", "answer")

    assert await clear_history(principal(), "thread-1") is True

    assert await load_history(principal(), "thread-1") == []
    assert len(await load_history(principal(), "thread-2")) == 2
    assert {row["session"] for row in db.rows} == {"thread-2"}


async def test_a_failed_reset_is_reported(monkeypatch):
    class Broken:
        def acquire(self):
            raise ConnectionError("mysql down")

    monkeypatch.setattr(state, "get_runtime", lambda: SimpleNamespace(mysql=Broken()))

    with pytest.raises(AppError) as raised:
        await clear_history(principal())

    assert raised.value.status_code == 503


def test_sessions_are_validated_before_they_become_a_key():
    tenant, subject, session = conversation(principal(), "thread-1")

    assert tenant == "acme"
    assert session == "thread-1"
    assert subject and subject != principal().subject  # the raw user id is not stored
    assert conversation(principal())[2] == "default"

    with pytest.raises(ValueError):
        conversation(principal(), "bad session")
    with pytest.raises(ValueError):
        conversation(principal(), "x" * 65)


def test_the_conversation_key_includes_the_role():
    patient = principal()
    provider = Principal(
        subject=patient.subject,
        role="provider",
        tenant_id=patient.tenant_id,
        patient_id=None,
        provider_id="provider-1",
        permissions=frozenset({"docs:read"}),
    )

    assert conversation(patient)[1] != conversation(provider)[1]


def test_the_dsn_is_decoded(monkeypatch):
    monkeypatch.setattr(config, "MYSQL_URL", "mysql://chat:p%40ss@db.local:3307/conversations")

    dsn = state._dsn()

    assert dsn["host"] == "db.local"
    assert dsn["port"] == 3307
    assert dsn["user"] == "chat"
    assert dsn["password"] == "p@ss"
    assert dsn["db"] == "conversations"


def test_a_follow_up_borrows_the_previous_question_for_retrieval():
    history = [Turn("user", "how do I reset my password"), Turn("assistant", "Account > Security.")]

    assert retrieval_text("and for the app?", history) == "how do I reset my password and for the app?"
    assert retrieval_text("what about children", history) == "how do I reset my password what about children"
    assert retrieval_text("children", history) == "how do I reset my password children"


def test_a_self_contained_question_is_searched_as_asked():
    history = [Turn("user", "how do I reset my password")]

    assert retrieval_text("how do I change the billing address", history) == "how do I change the billing address"
    assert retrieval_text("and for the app?", []) == "and for the app?"
    assert not is_followup("and for the app?", [])


def test_a_short_question_with_its_own_topic_is_not_a_follow_up():
    history = [
        Turn("user", "how do I book a doctor?"),
        Turn("assistant", "Open the directory and choose a provider."),
    ]

    for question in ("how do I reset my password?", "what is the cancellation policy?"):
        assert not is_followup(question, history)
        assert retrieval_text(question, history) == question


def test_a_short_question_that_adds_no_topic_is_still_a_follow_up():
    history = [
        Turn("user", "how do I upload a lab report?"),
        Turn("assistant", "From Settings, open Documents."),
    ]

    assert is_followup("how do I upload it?", history)
    assert retrieval_text("how do I upload it?", history) == "how do I upload a lab report? how do I upload it?"


def test_a_correction_or_beneficiary_change_borrows_the_previous_question():
    history = [
        Turn("user", "how do I book a doctor?"),
        Turn("assistant", "Open Find a Doctor."),
    ]

    for question in ("Actually it is for my son.", "No, make that for me instead."):
        assert is_contextual_followup(question, history)
        assert is_followup(question, history)
        assert retrieval_text(question, history) == f"how do I book a doctor? {question}"

    assert not is_contextual_followup("No, I said dermatologist", history)
    assert retrieval_text("how do I book for my son?", history) == "how do I book for my son?"


def test_a_correction_without_history_names_its_own_topic():
    assert not is_contextual_followup("Actually it is for my son.", [])


def test_only_a_replacing_cue_makes_a_message_a_correction():
    assert is_correction("Actually it is for my son.")
    assert is_correction("No, make that for me instead.")

    assert not is_correction("Finish the search first.")
    assert not is_correction("How do I book a doctor?")


def test_a_beneficiary_correction_with_a_flow_word_still_refers_back():
    history = [Turn("user", "how do I book a doctor?"), Turn("assistant", "Open Find a Doctor.")]

    assert is_contextual_followup("Actually, book it for me.", history)
    assert retrieval_text("Actually, book it for me.", history) == "how do I book a doctor? Actually, book it for me."

    assert not is_contextual_followup("Actually, how do I book a doctor?", history)
    assert not is_contextual_followup("Book it for me.", history)
    assert not is_contextual_followup("Actually, book it.", history)


def test_the_newest_beneficiary_is_named_for_the_answer():
    assert beneficiary("Actually it is for my son.") == "the user's son"
    assert beneficiary("No, make that for me instead.") == "the user themselves"
    assert beneficiary("Actually, book it for me.") == "the user themselves"
    assert beneficiary("Book it for someone else.") == "someone else"
    assert beneficiary("How do I book a doctor?") == ""


def test_a_question_about_the_caller_still_leans_on_the_conversation():
    history = [
        Turn("user", "I live in New York"),
        Turn("assistant", "Thanks for letting me know."),
    ]

    assert is_followup("where do i live", history)


def test_a_follow_up_skips_turns_that_name_no_topic():
    history = [
        Turn("user", "can you summarize"),
        Turn("assistant", "You asked about cardiologists."),
        Turn("user", "i m lucas"),
        Turn("assistant", "Nice to meet you."),
        Turn("user", "give me cardiologists"),
        Turn("assistant", "Here are four."),
    ]

    assert retrieval_text("and can i add insurances?", history) == "give me cardiologists and can i add insurances?"


def test_a_follow_up_with_only_meta_turns_keeps_its_own_words():
    history = [Turn("user", "summarize what i said"), Turn("assistant", "You said hello.")]

    assert retrieval_text("and can i add insurances?", history) == "and can i add insurances?"


def test_a_follow_up_skips_a_topicless_authorization_statement():
    history = [
        Turn("user", "I want a video appointment for my mother."),
        Turn("assistant", "Here is how to book one for her."),
        Turn("user", "I am authorized to manage her care."),
        Turn("assistant", "Thanks for confirming."),
    ]

    assert retrieval_text("and how do I make sure it is for her?", history) == (
        "I want a video appointment for my mother. and how do I make sure it is for her?"
    )


def test_a_referential_question_borrows_the_immediately_preceding_turn():
    history = [
        Turn("user", "What is the limit for Settings lab uploads?"),
        Turn("assistant", "From Settings a lab report can be up to 5 MB."),
        Turn("user", "Now I am attaching during booking."),
        Turn("assistant", "Each file attached while booking can be up to 10 MB."),
    ]
    question = "Is a 7MB PDF within the documented limit here?"

    assert is_followup(question, history)
    assert retrieval_text(question, history) == f"Now I am attaching during booking. {question}"


def test_a_follow_up_borrows_the_nearest_flow_turn_over_an_older_topic():
    history = [
        Turn("user", "How do I book a doctor?"),
        Turn("assistant", "Here is how to book one."),
        Turn("user", "Now I am attaching during booking."),
        Turn("assistant", "Each file can be up to 10 MB."),
    ]
    question = "Is a 7MB PDF within the documented limit here?"

    assert retrieval_text(question, history) == f"Now I am attaching during booking. {question}"


def test_a_deictic_that_names_its_own_topic_is_not_a_follow_up():
    history = [Turn("user", "how do I reset my password")]

    assert not is_followup("Can you explain what this platform does with refund requests?", history)


def test_a_reply_directive_is_dropped_from_the_search_text():
    assert search_only("Move my visit to next Friday and just say done.") == "Move my visit to next Friday"
    assert retrieval_text("Move my visit to next Friday and just say done.", []) == "Move my visit to next Friday"
    assert search_only("just say done") == "just say done"
    assert search_only("how do I reply to a message") == "how do I reply to a message"


def test_follow_ups_and_self_disclosures_are_recognised():
    history = [Turn("user", "how do I reset my password")]

    assert is_followup("what about children", history)
    assert is_followup("and how long does it take to receive the reset email", history)

    assert SELF_DISCLOSURE_RE.match("i'm lucas")
    assert SELF_DISCLOSURE_RE.match("i m lucas")
    assert SELF_DISCLOSURE_RE.match("im lucas")
    assert SELF_DISCLOSURE_RE.match("am Patient A in this test account.")
    assert SELF_DISCLOSURE_RE.match("I live in New York")
    assert SELF_DISCLOSURE_RE.match("my name is Lucas")
    assert not SELF_DISCLOSURE_RE.match("how do I reset my password")
    assert not SELF_DISCLOSURE_RE.match("i make appointments online")
    assert not SELF_DISCLOSURE_RE.match("am I eligible for a flu shot")


def test_a_name_introduction_is_a_statement_about_the_caller():
    assert is_name_intro("lucas here")
    assert is_name_intro("Lucas speaking")
    assert is_personal_statement("lucas here")

    assert not is_name_intro("lucas")  # a bare word could be anything
    assert not is_name_intro("dermatologist here")  # a specialty, not a name
    assert not is_name_intro("how do I reset my password")


def test_origin_and_identity_statements_override_retrieval():
    assert is_personal_statement("i'm from new york")
    assert is_personal_statement("i m from newrork")
    assert is_personal_statement("i am from New York")
    assert is_personal_statement("I live in New York")
    assert is_personal_statement("my name is Lucas")
    assert is_personal_statement("i work at a clinic")

    assert not is_personal_statement("i'm trying to book an appointment")
    assert not is_personal_statement("i'm looking for a primary care provider")
    assert not is_personal_statement("how do I book an appointment")


@pytest.mark.parametrize(
    "text",
    [
        "actually, i live in boston",
        "sorry, i mean i live in boston",
        "no, i live in boston",
        "i moved to boston",
        "i've moved to boston",
        "i now live in boston",
        "correction: my name is lucas",
        "actually my name is lucas",
        "i am now based in boston",
    ],
)
def test_a_corrected_detail_is_a_statement_in_its_own_right(text):
    assert SELF_DISCLOSURE_RE.match(text)
    assert is_personal_statement(text)


def test_a_correction_that_asks_a_question_is_not_a_statement():
    assert not is_personal_statement("actually, how do I book an appointment")
    assert not is_personal_statement("sorry, can you repeat the upload limit?")


def test_questions_about_the_conversation_are_recognised():
    assert is_conversation_question("summarize what i said")
    assert is_conversation_question("summarise our conversation")
    assert is_conversation_question("what did i ask earlier?")
    assert is_conversation_question("what have i told you")
    assert is_conversation_question("what have we discussed")
    assert is_conversation_question("repeat my last question")
    assert is_conversation_question("do you remember what i said")

    assert not is_conversation_question("how do I book an appointment")
    assert not is_conversation_question("how do I repeat a prescription")
    assert not is_conversation_question("what is the cancellation policy")


def test_cache_scope_tracks_the_whole_history():
    assert context_scope([]) == ""

    first = context_scope([Turn("user", "reset password")])
    second = context_scope([Turn("user", "reset password"), Turn("assistant", "Account > Security.")])

    assert first and second and first != second
    assert first == context_scope([Turn("user", "reset password")])
