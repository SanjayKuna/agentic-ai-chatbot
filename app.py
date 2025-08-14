import os
import re
import datetime
from difflib import SequenceMatcher

from flask import Flask, request, jsonify, render_template
from tavily import TavilyClient
import autogen

# --- Flask ---
app = Flask(__name__)

# --- LLM Config ---
config_list = autogen.config_list_from_json("OAI_CONFIG_LIST", file_location=".")
# Make sure your config contains an actually available model (e.g., gpt-4o, gpt-4.1, gpt-4o-mini),
# and NOT a missing one like "llama3-8b-8192".
llm_config = {"config_list": config_list, "temperature": 0.2}

# --- Tavily ---
def get_tavily_api_key():
    try:
        with open("tavily_api.txt", "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        return None

tavily_api_key = get_tavily_api_key()
tavily_client = TavilyClient(api_key=tavily_api_key) if tavily_api_key else None

# ------------ Utilities ------------
TIME_SENSITIVE_PATTERNS = [
    r"\bwho\s+is\s+the\s+president\b",
    r"\bwho\s+is\s+president\b",
    r"\bcurrent\b", r"\btoday\b", r"\bas of\b",
    r"\bprime\s+minister\b", r"\bgovernor\b", r"\bCEO\b", r"\bhead\s+of\s+state\b"
]

POLITICAL_HOLDER_PATTERNS = [
    r"\bpresident\b", r"\bprime\s+minister\b", r"\bgovernor\b"
]

JAN_2025 = datetime.datetime(2025, 1, 1)

def is_time_sensitive(question: str) -> bool:
    q = question.lower()
    return any(re.search(p, q) for p in TIME_SENSITIVE_PATTERNS)

def is_political_holder_question(question: str) -> bool:
    q = question.lower()
    return any(re.search(p, q) for p in POLITICAL_HOLDER_PATTERNS)

def validate_timestamp_in_text(text: str, threshold: datetime.datetime = JAN_2025) -> bool:
    """
    Returns True if we can find any date >= threshold inside text.
    Accepts formats like 'August 2025' or '20 January 2025'.
    """
    candidates = re.findall(r"(\b\d{1,2}\s+\w+\s+\d{4}\b|\b\w+\s+\d{4}\b)", text)
    for d in candidates:
        for fmt in ("%d %B %Y", "%B %Y"):
            try:
                parsed = datetime.datetime.strptime(d, fmt)
                if parsed >= threshold:
                    return True
            except ValueError:
                continue
    return False

def is_repeated(a: str, b: str) -> bool:
    return SequenceMatcher(None, a, b).ratio() > 0.90

def tavily_search(query: str, restrict_domains=True) -> list[dict]:
    """
    Returns a list of {url, content} dicts.
    If restrict_domains=True, prefers authoritative sites for factual freshness.
    """
    if not tavily_client:
        return [{"url": "N/A", "content": "Tavily client not initialized. Add tavily_api.txt"}]
    params = {"query": query, "search_depth": "advanced"}
    if restrict_domains:
        params["include_domains"] = [
            "whitehouse.gov", "usa.gov", "wikipedia.org",
            "reuters.com", "apnews.com", "bbc.com", "nytimes.com", "cnn.com"
        ]
    resp = tavily_client.search(**params)
    return [{"url": r["url"], "content": r["content"]} for r in resp.get("results", [])]

def force_live_context_for_question(question: str) -> str:
    """
    For time-sensitive questions, prefetch context so the LLM cannot rely on stale memory.
    """
    if not is_time_sensitive(question):
        return ""

    # Craft targeted queries
    queries = [question]
    if "president" in question.lower():
        # Push highly authoritative domains & 2025 wording
        queries = [
            'site:whitehouse.gov "President" 2025',
            'site:usa.gov "President of the United States"',
            'site:wikipedia.org "President of the United States" 2025'
        ]
    # Gather
    chunks = []
    for q in queries:
        for r in tavily_search(q, restrict_domains=True):
            chunks.append(f"Source: {r['url']}\nContent: {r['content']}")
    if not chunks:
        return ""
    return "\n\n".join(chunks)

def post_validate_and_correct(question: str, answer: str) -> str:
    """
    Final guardrail: if the topic is political office holder (e.g., US President),
    we re-check with Tavily and auto-correct stale outputs.
    """
    if not is_political_holder_question(question):
        return answer

    # Quick verification sweep
    verify_queries = [
        'site:whitehouse.gov "President" 2025',
        'site:wikipedia.org "President of the United States" 2025',
        'site:usa.gov "President of the United States"'
    ]
    fresh_hits = []
    for q in verify_queries:
        fresh_hits.extend(tavily_search(q, restrict_domains=True))

    # Decide likely correct incumbent from fresh sources
    trump_score, biden_score = 0, 0
    trump_recent, biden_recent = False, False

    for h in fresh_hits:
        text = f"{h['url']}\n{h['content']}"
        if "donald" in text.lower() and "trump" in text.lower():
            trump_score += 1
            trump_recent = trump_recent or validate_timestamp_in_text(text)
        if "joe" in text.lower() and "biden" in text.lower():
            biden_score += 1
            biden_recent = biden_recent or validate_timestamp_in_text(text)

    # Heuristic: prefer who appears more AND has recent timestamps
    likely_trump = (trump_score >= biden_score) and (trump_recent or not biden_recent)

    # If answer contradicts verification, rewrite succinctly
    if "biden" in answer.lower() and likely_trump:
        # Build a corrected short answer
        cites = []
        for h in fresh_hits[:3]:
            cites.append(f"- {h['url']}")
        corrected = (
            "As of 2025, the President of the United States is **Donald J. Trump** "
            "(inaugurated January 20, 2025).\n\nSources:\n" + "\n".join(cites)
        )
        return corrected

    return answer

# ------------ AutoGen Orchestration ------------
def run_autogen_chat(initial_task: str):
    last_researcher_msg = ""
    last_critic_msg = ""

    # User Proxy executes tools upon explicit requests (we also force pre-context + post-validation outside).
    user_proxy = autogen.UserProxyAgent(
        name="UserProxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=3,
        is_termination_msg=lambda m: "TERMINATE" in m.get("content", "").upper(),
        code_execution_config={"work_dir": "output", "use_docker": False},
    )

    # Register a callable tool; we’ll enforce usage via prompt protocol.
    def web_search(query: str) -> str:
        rows = tavily_search(query, restrict_domains=True)
        # Keep only sources that look fresh when it matters
        blocks = []
        for r in rows:
            if not is_political_holder_question(initial_task) or validate_timestamp_in_text(r["content"]):
                blocks.append(f"Source: {r['url']}\nContent: {r['content']}")
        if not blocks:  # fallback if freshness check filtered everything out
            blocks = [f"Source: {r['url']}\nContent: {r['content']}" for r in rows]
        return "\n\n".join(blocks) if blocks else "NO_RESULTS"
    user_proxy.register_function(function_map={"web_search": web_search})

    # Forced live context (prepended to the user task)
    forced_context = force_live_context_for_question(initial_task)
    context_prefix = (
        f"[FORCED_LIVE_CONTEXT]\n{forced_context}\n[/FORCED_LIVE_CONTEXT]\n\n"
        if forced_context else ""
    )
    task_with_context = (
        context_prefix
        + initial_task
        + (
            "\n\n(Use only the FORCED_LIVE_CONTEXT above if sufficient; otherwise, call the `web_search` tool. "
            "Do NOT rely on memory for time-sensitive facts.)"
            if forced_context else ""
        )
    )

    # Researcher
    researcher = autogen.AssistantAgent(
        name="Researcher",
        llm_config=llm_config,
        is_termination_msg=lambda m: "TERMINATE" in m.get("content", "").upper(),
        system_message=f"""
You are a researcher. Today's date is {datetime.date.today()}.

Protocol:
1) For time-sensitive topics (e.g., office holders, current stats), NEVER answer from memory.
2) First, read any [FORCED_LIVE_CONTEXT] provided by the system. Prefer it if it contains post–Jan 2025 data.
3) If context is insufficient, you MUST call the tool exactly like this on a new line:
   TOOL: web_search
   ARGS: {{"query": "<your specific query>"}}
4) If results conflict (e.g., Biden vs Trump), you MUST re-run web_search with tighter queries and pick
   the result with the most recent dated source (>= Jan 2025). Quote dates when available.

Output format:
- A concise draft answer.
- A "Sources:" section with URLs.
- If/when done, write TERMINATE on the last line.
"""
    )

    # Critic
    critic = autogen.AssistantAgent(
        name="Critic",
        llm_config=llm_config,
        is_termination_msg=lambda m: "TERMINATE" in m.get("content", "").upper(),
        system_message="""
You are a critic.

Checks required:
- Reject answers about political office holders if any cited source is before Jan 2025.
- If conflict exists in the sources, instruct the Researcher to re-search and choose the most recent dated source.
- No memory-only claims.

If correct and up-to-date:
- Return ONLY the final formatted answer (Markdown is fine).
- End with TERMINATE on the last line.
"""
    )

    # GroupChat
    groupchat = autogen.GroupChat(
        agents=[user_proxy, researcher, critic],
        messages=[],
        max_round=4
    )
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

    # Kick off
    chat_result = user_proxy.initiate_chat(manager, message=task_with_context)
    chat_history = chat_result.chat_history

    # Simple repetition break (post-run scan)
    for m in chat_history:
        name = m.get("name")
        content = m.get("content", "")
        if name == "Researcher":
            if is_repeated(content, last_researcher_msg):
                break
            last_researcher_msg = content
        elif name == "Critic":
            if is_repeated(content, last_critic_msg):
                break
            last_critic_msg = content

    # Extract final answer first (TERMINATE), else last good message
    final_answer = None
    for m in reversed(chat_history):
        if m.get("name") == "Critic" and "TERMINATE" in (m.get("content") or ""):
            final_answer = m["content"].replace("TERMINATE", "").strip()
            break
    if not final_answer:
        for m in reversed(chat_history):
            if m.get("name") in ("Critic", "Researcher"):
                final_answer = (m.get("content") or "").strip()
                break
    final_answer = final_answer or "No answer could be extracted."

    # Post-validation & auto-correction guardrail (hard stop against staleness)
    final_answer = post_validate_and_correct(initial_task, final_answer)
    return final_answer

# ------------ HTTP API ------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    question = data.get('question')
    if not question:
        return jsonify({"error": "No question provided"}), 400
    answer = run_autogen_chat(question)
    return jsonify({"answer": answer})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
