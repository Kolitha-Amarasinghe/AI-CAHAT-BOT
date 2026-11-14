import re
from datetime import date

import ollama
import streamlit as st
from ddgs import DDGS
from pypdf import PdfReader

# ---------- Settings (tuned for an 8 GB CPU-only laptop) ----------
FAST_MODEL = "llama3.2:1b"   # normal chat: fast
SMART_MODEL = "llama3.2"     # tools mode: slower but picks tools reliably
OPTS = {"num_ctx": 2048, "num_predict": 500}  # small memory use, capped reply length
KEEP = "10m"                 # keep the model loaded between messages
HISTORY = 6                  # only send the last 6 messages
DOC_LIMIT = 4000             # characters of an uploaded file to use


# ---------- Tools ----------
def web_search(query: str) -> str:
    """Search the web for current information.

    Args:
        query: What to search for.
    """
    try:
        results = DDGS().text(query, max_results=3)
    except Exception as e:
        return f"Search failed: {e}"
    if not results:
        return "No results found."
    return "\n\n".join(f"{r['title']}\n{r['body'][:300]}" for r in results)


def calculator(expression: str) -> str:
    """Calculate a maths expression, for example 12*(3+4).

    Args:
        expression: The maths expression to calculate.
    """
    if not re.fullmatch(r"[0-9+\-*/().% ]+", expression) or "**" in expression:
        return "Invalid expression."
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Error: {e}"


TOOLS = [web_search, calculator]
TOOL_MAP = {f.__name__: f for f in TOOLS}


# ---------- Helpers ----------
@st.cache_data(show_spinner=False)
def read_upload(name: str, data: bytes) -> str:
    if name.lower().endswith(".pdf"):
        import io
        return "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(data)).pages)
    return data.decode("utf-8", errors="ignore")


def system_prompt(doc: str, tools: bool) -> dict:
    text = (
        "You are Turing AI, a friendly assistant made in Sri Lanka, running locally. "
        "Answer briefly and clearly. "
        f"Today is {date.today():%B %d, %Y}."
    )
    if tools:
        text += " Use web_search for current events and calculator for maths."
    if doc:
        text += "\n\nUse this uploaded file to answer questions:\n" + doc[:DOC_LIMIT]
    return {"role": "system", "content": text}


def stream_chat(model, msgs):
    for c in ollama.chat(model=model, messages=msgs, options=OPTS,
                         keep_alive=KEEP, stream=True):
        yield c["message"]["content"]


def tool_rounds(msgs, status):
    """Let the model use tools. Returns final text, or None if a streamed answer is still needed."""
    for _ in range(3):
        resp = ollama.chat(model=SMART_MODEL, messages=msgs, tools=TOOLS,
                           options=OPTS, keep_alive=KEEP)
        msg = resp.message
        if not msg.tool_calls:
            return msg.content
        msgs.append(msg)
        for call in msg.tool_calls:
            name, args = call.function.name, call.function.arguments
            status.write(f"Using {name}: {args}")
            fn = TOOL_MAP.get(name)
            try:
                result = fn(**args) if fn else f"Unknown tool: {name}"
            except Exception as e:
                result = f"Error: {e}"
            msgs.append({"role": "tool", "tool_name": name, "content": str(result)})
    return None


# ---------- Page ----------
st.set_page_config(page_title="Turing AI", page_icon="🤖")
st.title("Turing AI")

if "messages" not in st.session_state:
    st.session_state.messages = []

doc = ""
with st.sidebar:
    st.header("Turing AI")
    use_tools = st.toggle("Use tools (web search, maths)", value=False,
                          help="Slower. Turn on only when you need current info or exact maths.")
    st.caption(f"Model: {SMART_MODEL if use_tools else FAST_MODEL}")
    upload = st.file_uploader("Upload a file to ask about", type=["txt", "md", "csv", "py", "pdf"])
    if upload:
        doc = read_upload(upload.name, upload.getvalue())
        st.success(f"Loaded {upload.name}")
    if st.button("New chat"):
        st.session_state.messages = []
        st.rerun()

st.caption("Fast mode is on." if not use_tools else "Tools mode is on. Replies are slower.")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("Message Turing AI"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    msgs = [system_prompt(doc, use_tools)] + st.session_state.messages[-HISTORY:]

    with st.chat_message("assistant"):
        try:
            if use_tools:
                with st.status("Thinking…") as status:
                    text = tool_rounds(msgs, status)
                    status.update(label="Done", state="complete")
                if text is not None:
                    st.markdown(text)
                    reply = text
                else:
                    reply = st.write_stream(stream_chat(SMART_MODEL, msgs))
            else:
                reply = st.write_stream(stream_chat(FAST_MODEL, msgs))
        except Exception as e:
            reply = f"Something went wrong: {e}. Check that Ollama is running and the model is downloaded."
            st.error(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})