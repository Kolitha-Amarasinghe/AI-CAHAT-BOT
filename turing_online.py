import io
from datetime import date

import streamlit as st
from ddgs import DDGS
from google import genai
from google.genai import types
from pypdf import PdfReader


def secret(name, default=None):
    try:
        return st.secrets[name]
    except Exception:
        return default


# ---------- Settings ----------
MODEL = secret("GEMINI_MODEL", "gemini-flash-latest")  # change in Secrets if you get a 404
MAX_MESSAGES = 30   # per visitor session, protects your free quota
HISTORY = 10        # recent messages sent to the model
DOC_LIMIT = 12000   # characters of an uploaded file to use

st.set_page_config(page_title="Turing AI", page_icon="🤖")
st.title("Turing AI")
st.caption("Ask in English, Sinhala or Tamil.")

# ---------- Optional password ----------
password = secret("APP_PASSWORD")
if password and not st.session_state.get("ok"):
    entered = st.text_input("Password", type="password")
    if entered == password:
        st.session_state.ok = True
        st.rerun()
    elif entered:
        st.error("Wrong password.")
    st.stop()

api_key = secret("GEMINI_API_KEY")
if not api_key:
    st.error("GEMINI_API_KEY is missing. Add it in the app's Secrets settings.")
    st.stop()


@st.cache_resource
def get_client():
    return genai.Client(api_key=api_key)


@st.cache_data(show_spinner=False)
def read_upload(name: str, data: bytes) -> str:
    if name.lower().endswith(".pdf"):
        return "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(data)).pages)
    return data.decode("utf-8", errors="ignore")


def web_context(query: str) -> str:
    try:
        results = DDGS().text(query, max_results=4)
    except Exception:
        return ""
    return "\n\n".join(f"{r['title']}\n{r['body'][:300]}\n{r['href']}" for r in results or [])


def build_system(doc: str, web: str) -> str:
    text = (
        "You are Turing AI, a friendly assistant made in Sri Lanka. "
        "Always reply in the same language the user writes in. For Sinhala and Tamil, "
        "use proper Sinhala or Tamil script. Be clear and concise. "
        f"Today is {date.today():%B %d, %Y}."
    )
    if doc:
        text += "\n\nThe user uploaded a file. Use it to answer:\n" + doc[:DOC_LIMIT]
    if web:
        text += "\n\nWeb search results (mention the source links when you use them):\n" + web
    return text


def stream_reply(history, system):
    contents = [
        types.Content(role="user" if m["role"] == "user" else "model",
                      parts=[types.Part(text=m["content"])])
        for m in history
    ]
    config = types.GenerateContentConfig(system_instruction=system, max_output_tokens=1024)
    for chunk in get_client().models.generate_content_stream(
        model=MODEL, contents=contents, config=config
    ):
        if chunk.text:
            yield chunk.text


# ---------- Page ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

doc = ""
with st.sidebar:
    st.header("Turing AI")
    use_web = st.toggle("Search the web", value=False)
    upload = st.file_uploader("Upload a file to ask about", type=["txt", "md", "csv", "py", "pdf"])
    if upload:
        doc = read_upload(upload.name, upload.getvalue())
        st.success(f"Loaded {upload.name}")
    if st.button("New chat"):
        st.session_state.messages = []
        st.rerun()
    st.caption("Do not share passwords or private information. Messages are sent to Google's Gemini API.")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if len(st.session_state.messages) >= MAX_MESSAGES * 2:
    st.warning("Message limit reached for this session. Click New chat to continue.")
    st.stop()

if prompt := st.chat_input("Message Turing AI"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    web = web_context(prompt) if use_web else ""
    system = build_system(doc, web)
    history = st.session_state.messages[-HISTORY:]

    with st.chat_message("assistant"):
        try:
            reply = st.write_stream(stream_reply(history, system))
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                reply = "Turing AI is busy right now (free usage limit reached). Please try again in a minute."
            elif "404" in str(e):
                reply = "The model name is not available. The owner should set GEMINI_MODEL in Secrets."
            else:
                reply = "Something went wrong. Please try again."
            st.error(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})