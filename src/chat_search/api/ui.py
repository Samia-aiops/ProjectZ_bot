"""Minimal browser chat page.

Laravel redirects the signed-in user to ``https://<chatbot>/#token=<JWT>``.
A URL fragment is never sent to servers or written to access logs; the page
removes it from the address bar after reading it.
"""

from __future__ import annotations

CHAT_PAGE_HTML = """<!DOCTYPE html>
<html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Assistant</title>
<style>
body{margin:0;font-family:Arial,sans-serif;background:#f7f8fb}
.header{height:36px;background:#142d82}
.box{max-width:800px;margin:30px auto;padding:0 20px}
#log{min-height:200px}
.msg{margin:12px 0;padding:14px;border-radius:10px;white-space:pre-wrap}
.user{background:#142d82;color:#fff;margin-left:20%}
.bot{background:#fff;box-shadow:0 2px 10px rgba(0,0,0,.06);margin-right:10%}
.src{font-size:12px;color:#666;margin-top:8px}
textarea{width:100%;box-sizing:border-box;padding:12px;border:1px solid #ddd;border-radius:8px;min-height:70px}
button{margin-top:8px;padding:10px 24px;border:0;border-radius:8px;background:#142d82;color:#fff;cursor:pointer}
button:disabled{opacity:.6;cursor:default}
#new{background:#e8ebf3;color:#142d82;margin-left:8px}
</style></head>
<body><div class="header"></div>
<div class="box">
  <div id="log"></div>
  <textarea id="q" maxlength="1000" placeholder="Ask about policies, onboarding or how to find something..."></textarea>
  <button id="send">Ask</button>
  <button id="new">New chat</button>
</div>
<script>
const token = new URLSearchParams(location.hash.slice(1)).get("token");
history.replaceState(null, "", location.pathname);  // keep token out of the URL bar
const log = document.getElementById("log");
const btn = document.getElementById("send");

function add(cls, text) {
  const d = document.createElement("div");
  d.className = "msg " + cls;
  const t = document.createElement("span");
  t.textContent = text;
  d.appendChild(t);
  log.appendChild(d);
  return {box: d, text: t, pending: false};
}

function clear_pending(bot) {
  if (!bot.pending) return;
  bot.pending = false;
  bot.text.textContent = "";
}

function handle(block, bot) {
  let ev = "message", data = "";
  for (const line of block.split("\\n")) {
    if (line.startsWith("event:")) ev = line.slice(6).trim();
    else if (line.startsWith("data:")) data += line.slice(5).trim();
  }
  if (!data) return;
  const p = JSON.parse(data);
  if (ev === "token") { clear_pending(bot); bot.text.textContent += p.text; }
  else if (ev === "error") { clear_pending(bot); bot.text.textContent += (bot.text.textContent ? "\\n\\n" : "") + p.message; }
  else if (ev === "done" && p.sources && p.sources.length) {
    const s = document.createElement("div");
    s.className = "src";
    s.textContent = "Sources: " + p.sources.map(x => x.document + " (p." + x.page + ")").join(", ");
    bot.box.appendChild(s);
  }
}

async function send() {
  const q = document.getElementById("q");
  const message = q.value.trim();
  if (!message) return;
  if (!token) { add("bot", "Please open the assistant from the portal."); return; }
  add("user", message);
  q.value = "";
  btn.disabled = true;
  const bot = add("bot", "");
  // Retrieval and the guard run before the first token; show progress instead of an empty bubble.
  bot.text.textContent = "Answering your query\\u2026";
  bot.pending = true;
  try {
    const r = await fetch("/v1/chat/stream", {
      method: "POST",
      headers: {"Content-Type": "application/json", "Authorization": "Bearer " + token},
      body: JSON.stringify({message})
    });
    if (!r.ok) {
      const d = await r.json().catch(() => ({}));
      bot.text.textContent = r.status === 401
        ? "Your session expired. Please reopen the assistant from the portal."
        : ((d.error && d.error.message) || "Request failed");
      return;
    }
    const reader = r.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    while (true) {
      const {value, done} = await reader.read();
      if (done) break;
      buf += dec.decode(value, {stream: true});
      let i;
      while ((i = buf.indexOf("\\n\\n")) >= 0) {
        handle(buf.slice(0, i), bot);
        buf = buf.slice(i + 2);
      }
    }
  } catch (e) {
    bot.text.textContent = "Connection problem. Please try again.";
  } finally {
    btn.disabled = false;
  }
}
btn.onclick = send;

document.getElementById("new").onclick = async () => {
  log.textContent = "";
  if (!token) return;
  try {
    await fetch("/v1/chat/session", {method: "DELETE", headers: {"Authorization": "Bearer " + token}});
  } catch (e) {}
  add("bot", "New conversation started.");
};
</script></body></html>
"""
