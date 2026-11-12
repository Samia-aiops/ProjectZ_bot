"""User-facing messages.

Keeping them in one module makes review and localisation straightforward and
ensures error handlers never echo raw input back to the caller.
"""

from __future__ import annotations

import re

BUSY_MSG = "The assistant is busy right now. Please try again in a moment or contact support."
ERROR_MSG = "Something went wrong on our side. Please try again or contact support."
NO_ANSWER = "I couldn't find that in the available resources. Please contact platform support for help."
SEARCH_UNAVAILABLE_MSG = "Provider search is temporarily unavailable. Please try again or contact support."
PROVIDER_DENIED_MSG = "Provider search isn't available for provider accounts."
PROVIDER_PERMISSION_MSG = "Provider search isn't available for your account."

GREETING_REPLY = (
    "Hello! I'm the platform assistant. I can help with policies, onboarding and "
    "finding your way around the portal. What would you like to know?"
)

# The injection model flags some one- or two-word fragments ("tell") as
# instructions. Telling the caller the request is off-limits is confusing when
# there is no request, so short fragments are asked to grow a little instead.
FRAGMENT_REPLY = (
    "Could you say a bit more about what you'd like help with? I can answer "
    "questions about platform policies, onboarding and navigation."
)

# Asked to recap a conversation that has no earlier turns, there is nothing to
# summarize; saying so is clearer than a "nothing found" reply.
NO_HISTORY_MSG = (
    "There's nothing earlier in this conversation for me to summarize yet. Ask "
    "me a question and I can recap it afterwards."
)

# A "nothing found" reply still names the topic the caller asked about, so the
# next step is obvious. The first matching topic wins; anything else uses
# NO_ANSWER.
FALLBACKS = (
    (
        re.compile(r"\b(refunds?|billing|invoices?|payments?|charged|subscription|credits?)\b", re.I),
        "I couldn't find that in the available resources. For billing, refund or payment questions, "
        "contact platform support.",
    ),
    (
        re.compile(
            r"\b(book\w*|appointments?|visits?|schedul\w+|reschedul\w+|cancel\w*|"
            r"move\s+(?:my\s+|the\s+)?(?:visit|appointment))\b",
            re.I,
        ),
        "I couldn't find that in the available resources. For help booking, changing or cancelling "
        "an appointment, contact platform support.",
    ),
    (
        re.compile(r"\b(upload\w*|file sizes?|attachments?|lab reports?|\d+\s?mb|pdfs?)\b", re.I),
        "I couldn't find that in the available resources. For upload problems, check the file type "
        "and the size limit shown on that screen, then contact platform support if it still fails.",
    ),
)


def fallback_reply(text: str) -> str:
    for pattern, reply in FALLBACKS:
        if pattern.search(text):
            return reply
    return NO_ANSWER


# Every "nothing found" reply: none says anything about the caller's topic, so
# none becomes context for the next turn.
FALLBACK_TEXTS = frozenset(reply for _, reply in FALLBACKS) | {NO_ANSWER}

REFUSAL = {
    "emergency": (
        "This may be a medical emergency. Call your local emergency number now (911 in the US) "
        "or go to the nearest emergency room; do not wait for this assistant. I can't help with "
        "urgent medical situations. If you're thinking about harming yourself, call or text 988 "
        "in the US, contact your local crisis line, or reach out to someone you trust now."
    ),
    # The caller asked the assistant to break its own rules; say so plainly
    # instead of dressing the refusal up as a capability limit.
    "injection": "Blocked: security violation",
    # Ending a session or becoming another patient is a sign-in, not a chat
    # action; the caller is told the route that actually works.
    "identity": (
        "I can't switch accounts or act as another patient without signing in. To use "
        "another patient's account, sign out and sign in with that patient's own "
        "credentials, or contact support for help."
    ),
    "privacy": "Blocked: security violation",
    "restricted": "I can't help with that request. I can answer questions about platform policies, onboarding and navigation.",
    "medical": (
        "I can't give medical advice, diagnoses or medication guidance. Please speak with your "
        "healthcare provider. I'm happy to help with questions about the platform itself."
    ),
}

HTTP_ERROR_MESSAGES = {
    401: "Authentication required",
    403: "Access denied",
    404: "Not found",
    405: "Method not allowed",
}
