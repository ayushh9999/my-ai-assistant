# ─────────────────────────────────────────────
#  CV Chatbot  |  Streamlit UI
#  Run with: streamlit run app.py
# ─────────────────────────────────────────────

import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ── Load API key ───────────────────────────────
load_dotenv()


def get_groq_api_key():
    """Read the key from Streamlit Cloud secrets or local environment variables."""
    try:
        cloud_key = st.secrets.get("GROQ_API_KEY")
    except FileNotFoundError:
        cloud_key = None

    return cloud_key or os.getenv("GROQ_API_KEY")


GROQ_API_KEY = get_groq_api_key()
DEFAULT_CV = Path(__file__).with_name("abcde14.pdf")

# ── Page config ────────────────────────────────
st.set_page_config(page_title="Ayush's AI Assistant", page_icon="👓", layout="centered")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
        --ink: #17324d;
        --muted: #6b7280;
        --cream: #fbf8f2;
        --paper: #ffffff;
        --coral: #e56b55;
        --gold: #f4b942;
        --line: #e8e1d8;
    }

    .stApp {
        background: var(--cream);
        color: var(--ink);
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stAppViewContainer"] > .main {
        padding-top: 3.5rem;
    }

    .block-container {
        max-width: 860px;
        padding: 2rem 1.25rem 7rem;
    }

    .brand-mark {
        display: inline-flex;
        align-items: center;
        gap: 0.7rem;
        color: var(--ink);
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(1.8rem, 4vw, 3rem);
        font-weight: 700;
        letter-spacing: 0;
        line-height: 1.1;
    }

    .brand-dot {
        width: 0.75rem;
        height: 0.75rem;
        border-radius: 50%;
        background: var(--coral);
        box-shadow: 0.45rem 0.45rem 0 var(--gold);
    }

    .brand-subtitle {
        margin: 0.85rem 0 2.2rem 1.45rem;
        color: var(--muted);
        font-family: 'DM Sans', sans-serif;
        font-size: 1rem;
    }

    [data-testid="stChatMessage"] {
        border: 1px solid var(--line);
        border-radius: 1rem;
        margin: 0.85rem 0;
        padding: 0.9rem 1rem;
        background: var(--paper);
        box-shadow: 0 8px 24px rgba(23, 50, 77, 0.05);
    }

    [data-testid="stChatMessage"] p {
        color: var(--ink) !important;
        font-family: 'DM Sans', sans-serif;
        line-height: 1.65;
    }

    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] strong,
    [data-testid="stChatMessage"] code {
        color: var(--ink) !important;
    }

    [data-testid="stChatMessage"] [data-testid="stChatMessageAvatarAssistant"] {
        background: var(--gold);
    }

    [data-testid="stChatInput"] {
        bottom: 1.5rem;
    }

    [data-testid="stBottom"] {
        background: var(--cream);
    }

    [data-testid="stChatInput"] > div {
        border: 1px solid var(--ink);
        border-radius: 1rem;
        background: var(--paper);
        box-shadow: 0 12px 28px rgba(23, 50, 77, 0.12);
    }

    [data-testid="stChatInput"] textarea {
        color: var(--ink) !important;
        font-family: 'DM Sans', sans-serif;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: var(--muted) !important;
        opacity: 1;
    }

    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="brand-mark"><span class="brand-dot"></span>Ayush\'s AI Assistant</div>'
    '<div class="brand-subtitle">I\'m Ayush\'s assistant. If you want to know anything about him, ask me.</div>',
    unsafe_allow_html=True,
)

# ── Helper functions (same logic as chatbot.py) ─

def load_cv(source):
    """Load a CV from the bundled path or an uploaded file."""
    if isinstance(source, (str, Path)):
        file_path = Path(source)
        if file_path.suffix.lower() == ".pdf":
            reader = PdfReader(str(file_path))
            return [
                Document(page_content=page.extract_text() or "", metadata={"page": page_number})
                for page_number, page in enumerate(reader.pages)
            ]
        return [Document(page_content=file_path.read_text(encoding="utf-8"), metadata={})]

    temp_path = Path(f"temp_{source.name}")
    temp_path.write_bytes(source.getvalue())
    try:
        return load_cv(temp_path)
    finally:
        temp_path.unlink(missing_ok=True)


def build_chain(documents):
    """Takes CV documents and returns a ready-to-use chain."""

    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not configured. Add it to Streamlit Cloud "
            "Settings > Secrets or to the local .env file."
        )

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    # Embeddings + vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)

    # Prompt
    prompt = PromptTemplate.from_template("""
You are my personal AI assistant, speaking on my behalf to recruiters.
Answer as if you are me in a natural, relaxed, professional conversation.
Use the information in my CV as your source of truth and speak in first person.
Sound warm and human: use natural phrasing and contractions when they fit, avoid
stiff corporate language, and keep answers focused. Use a short paragraph unless
bullets make the answer easier to read. If my CV does not contain the answer, be
honest and say you do not have that detail rather than guessing.

CV Information:
{context}

Recruiter's Question: {question}

Answer:""")

    # Groq LLM
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=GROQ_API_KEY,
        temperature=0.2,
        max_tokens=1024,
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


# ── Session state ──────────────────────────────
# st.session_state persists values across reruns (Streamlit reruns on every interaction)

if "messages" not in st.session_state:
    st.session_state.messages = []   # chat history

if "chain" not in st.session_state:
    st.session_state.chain = None    # the RAG chain

if "initialization_error" not in st.session_state:
    st.session_state.initialization_error = None

if st.session_state.chain is None and DEFAULT_CV.exists():
    with st.spinner("Preparing the candidate profile..."):
        try:
            st.session_state.chain = build_chain(load_cv(DEFAULT_CV))
        except Exception as error:
            st.session_state.initialization_error = error


# ── Main area — Chat ───────────────────────────
if not st.session_state.chain:
    if st.session_state.initialization_error:
        st.error(f"I could not start: {st.session_state.initialization_error}")
    else:
        st.error("My profile could not be loaded. Check that the bundled CV file is available.")
else:
    # Show chat history
    for message in st.session_state.messages:
        avatar = "assistant_mascot.svg" if message["role"] == "assistant" else "🙂"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    # Chat input box (appears at the bottom)
    question = st.chat_input("Ask me anything...")

    if question:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user", avatar="🙂"):
            st.markdown(question)

        # Get and show bot answer
        with st.chat_message("assistant", avatar="assistant_mascot.svg"):
            with st.spinner("Thinking..."):
                try:
                    answer = st.session_state.chain.invoke(question)
                except Exception as error:
                    response = getattr(error, "response", None)
                    status_code = getattr(response, "status_code", None)
                    if status_code == 401:
                        st.error(
                            "Groq rejected the API key. Check GROQ_API_KEY "
                            "in Streamlit Cloud Secrets."
                        )
                    elif status_code == 429:
                        st.error("Groq rate limit or quota reached. Try again later.")
                    else:
                        st.error(
                            "Groq could not answer this request. Verify the "
                            "deployment secret and model availability."
                        )
                    st.stop()
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})