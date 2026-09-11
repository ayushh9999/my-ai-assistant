# ─────────────────────────────────────────────
#  CV Chatbot  |  LangChain + Groq + FAISS
# ─────────────────────────────────────────────
#
#  HOW IT WORKS:
#  1. Load your CV (PDF or TXT)
#  2. Split it into small chunks
#  3. Turn chunks into vectors using a LOCAL embedding model (no API needed)
#  4. When a question comes in → find the most relevant chunks
#  5. Send those chunks + the question to Groq → get answer
#
# ─────────────────────────────────────────────

import os
from pathlib import Path
from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings        # free, runs locally
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ── 1. Load environment variables (.env file) ──
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY not found. Please add it to your .env file.")


# ── 2. Load your CV ────────────────────────────
def load_cv(file_path: str):
    """Loads a PDF or TXT file and returns a list of document pages."""
    print(f"📄 Loading CV from: {file_path}")

    path = Path(file_path)
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        documents = [
            Document(page_content=page.extract_text() or "", metadata={"page": page_number})
            for page_number, page in enumerate(reader.pages)
        ]
    elif path.suffix.lower() == ".txt":
        documents = [Document(page_content=path.read_text(encoding="utf-8"), metadata={})]
    else:
        raise ValueError("❌ Only .pdf and .txt files are supported.")

    print(f"   ✅ Loaded {len(documents)} page(s)")
    return documents


# ── 3. Split CV into chunks ────────────────────
def split_into_chunks(documents):
    """
    Splits the CV text into smaller overlapping chunks.
    - chunk_size: how many characters per chunk
    - chunk_overlap: characters shared between chunks (avoids cutting mid-sentence)
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )
    chunks = splitter.split_documents(documents)
    print(f"   ✅ Split into {len(chunks)} chunks")
    return chunks


# ── 4. Create vector store ─────────────────────
def create_vector_store(chunks):
    """
    Converts text chunks into embeddings using a LOCAL model (no API key needed).
    'all-MiniLM-L6-v2' is small, fast, and great for semantic search.
    Downloads once (~90MB), then cached on your machine.
    """
    print("🔢 Loading embedding model (downloads once ~90MB, then cached)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    print("   ✅ Embedding model ready")
    print("📦 Building vector store...")
    vector_store = FAISS.from_documents(chunks, embeddings)
    print("   ✅ Vector store ready")
    return vector_store


# ── 5. Build the QA chain ──────────────────────
def build_qa_chain(vector_store):
    """
    Connects everything using LangChain's LCEL (pipe) syntax:
      retriever | prompt | llm | output_parser
    """

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

    # GPT OSS 120B is available through Groq's hosted inference API.
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=GROQ_API_KEY,
        temperature=0.2,
        max_tokens=1024,
    )

    # Retriever: fetch top 3 most relevant CV chunks per question
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # Format the retrieved documents into a single string for the prompt
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


# ── 6. Chat loop ───────────────────────────────
def run_chat(chain):
    """Simple terminal chat loop. Type 'exit' to quit."""
    print("\n" + "═" * 50)
    print("  🤖 CV Chatbot is ready!")
    print("  Ask anything about me.")
    print("  Type 'exit' to quit.")
    print("═" * 50 + "\n")

    while True:
        question = input("You: ").strip()

        if not question:
            continue
        if question.lower() in ["exit", "quit", "bye"]:
            print("Bot: Goodbye! 👋")
            break

        print("Bot: Thinking...", end="\r")
        answer = chain.invoke(question)
        print(f"Bot: {answer}\n")


# ── 7. Main entry point ────────────────────────
if __name__ == "__main__":
    CV_FILE = "abcde14.pdf"   # ← Change this to your CV filename

    documents    = load_cv(CV_FILE)
    chunks       = split_into_chunks(documents)
    vector_store = create_vector_store(chunks)
    chain        = build_qa_chain(vector_store)

    run_chat(chain)