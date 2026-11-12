"""Redis token-bucket rate limiting.

Runs before any embedding, retrieval or LLM work so abusive traffic is cheap to
reject. One Lua script refills and charges the user, tenant and (optional)
source-address buckets in a single atomic call: a burst cannot straddle a window
boundary, and a request that is denied consumes no token. When Redis is
unavailable the configured fail-open/fail-closed policy decides.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math

from . import config
from .auth import Principal
from .errors import AppError
from .messages import BUSY_MSG
from .observability import client_ip_ctx, metric
from .runtime import get_runtime

log = logging.getLogger(__name__)

# KEYS: enabled buckets. ARGV: capacity, rate (tokens/second) per key, then the
# TTL in milliseconds. Returns {allowed, retry_after_ms}.
_BUCKETS = """
-- rl:buckets
local clock = redis.call('time')
local now = tonumber(clock[1]) + tonumber(clock[2]) / 1000000
local allowed = true
local retry = 0
local available = {}
for i = 1, #KEYS do
  local capacity = tonumber(ARGV[i * 2 - 1])
  local rate = tonumber(ARGV[i * 2])
  if capacity > 0 and rate > 0 then
    local state = redis.call('hmget', KEYS[i], 't', 's')
    local tokens = tonumber(state[1]) or capacity
    local stamp = tonumber(state[2]) or now
    tokens = math.min(capacity, tokens + math.max(now - stamp, 0) * rate)
    available[i] = tokens
    if tokens < 1 then
      allowed = false
      local wait = math.ceil((1 - tokens) / rate * 1000)
      if wait > retry then retry = wait end
    end
  end
end
for i = 1, #KEYS do
  if available[i] then
    local tokens = available[i]
    if allowed then tokens = tokens - 1 end
    redis.call('hset', KEYS[i], 't', tokens, 's', now)
    redis.call('pexpire', KEYS[i], ARGV[#ARGV])
  end
end
return {allowed and 1 or 0, retry}
"""


def _keys(principal: Principal) -> list[str]:
    user = hashlib.sha256(principal.subject.encode()).hexdigest()[:16]
    keys = [f"rl:u:{principal.tenant_id}:{user}", f"rl:t:{principal.tenant_id}"]
    if config.RATE_LIMIT_IP > 0:
        # Hashed: the limiter never stores a raw address.
        if address := client_ip_ctx.get():
            keys.append(f"rl:i:{hashlib.sha256(address.encode()).hexdigest()[:16]}")
    return keys


def _capacities(count: int) -> list[int]:
    limits = [config.RATE_LIMIT_USER, config.RATE_LIMIT_TENANT, config.RATE_LIMIT_IP]
    return limits[:count]


async def enforce_rate_limit(principal: Principal) -> None:
    keys = _keys(principal)
    window = max(config.RATE_LIMIT_WINDOW, 1)
    args: list[float] = []
    for capacity in _capacities(len(keys)):
        args.extend((capacity, max(capacity / window, 0.001)))
    args.append(window * 1000)
    try:
        allowed, retry_ms = await asyncio.wait_for(
            get_runtime().redis.eval(_BUCKETS, len(keys), *keys, *args), config.REDIS_TIMEOUT
        )
    except Exception as exc:
        metric(rate_limit="unavailable")
        log.warning("rate limiter unavailable: %s", type(exc).__name__)
        if config.RATE_LIMIT_FAIL_OPEN:
            return
        raise AppError(503, "busy", BUSY_MSG) from None

    if not allowed:
        metric(rate_limit="blocked")
        raise AppError(
            429,
            "rate_limited",
            "You're sending requests too quickly. Please try again in a moment or contact support.",
            {"Retry-After": str(max(math.ceil(retry_ms / 1000), 1))},
        )
