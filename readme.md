# Ayush's AI Assistant

A recruiter-facing AI assistant that speaks on Ayush's behalf. It reads the bundled CV, finds the most relevant experience for each question, and answers in a natural first-person voice.

The project uses retrieval-augmented generation (RAG): the assistant searches the CV first, then sends only the relevant context to Groq.

## What It Does

- Loads Ayush's bundled CV automatically on startup
- Answers questions about skills, projects, education, and experience
- Uses semantic search rather than simple keyword matching
- Speaks naturally in first person using "I" and "my"
- Avoids inventing details that are not in the CV
- Provides a branded Streamlit chat UI with a custom assistant mascot
- Includes a terminal interface for quick testing

## Project Structure

```text
cv-ai2/
├── app.py                  # Streamlit chat application
├── chatbot.py              # Terminal chat application
├── abcde14.pdf             # Bundled CV used as the assistant's profile
├── assistant_mascot.svg    # Assistant avatar shown in chat
├── requirements.txt        # Python dependencies
├── .env.example            # Safe environment variable template
└── .streamlit/config.toml  # Streamlit runtime configuration
```

## Tech Stack

| Technology | Role |
| --- | --- |
| Streamlit | Web interface |
| Groq | Hosted LLM inference |
| LangChain | RAG pipeline orchestration |
| FAISS | In-memory vector search |
| Sentence Transformers | Local CV embeddings |
| pypdf | PDF text extraction |

## Setup

### 1. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Configure Groq

Create an API key at [console.groq.com](https://console.groq.com), then create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Never commit `.env` or expose the key in source code.

## Run the App

Start the recruiter-facing web app:

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser. The CV is loaded automatically; there is no upload step.

For terminal testing:

```bash
python chatbot.py
```

Type `exit`, `quit`, or `bye` to stop the terminal assistant.

## How Retrieval Works

```text
Bundled CV
   -> Extract text with pypdf
   -> Split text into overlapping chunks
   -> Create local embeddings
   -> Store chunks in FAISS
   -> Retrieve the three closest chunks for each question
   -> Send context and question to Groq
   -> Return a first-person answer
```

The embedding model runs locally. Only the retrieved CV context and recruiter question are sent to Groq.

## Configuration

The main retrieval and response settings are defined in `app.py` and `chatbot.py`:

| Setting | Current value | Purpose |
| --- | ---: | --- |
| Chunk size | `500` characters | Size of each searchable CV section |
| Chunk overlap | `100` characters | Keeps context across section boundaries |
| Retrieved chunks | `3` | CV sections sent to the model per question |
| Temperature | `0.2` | Keeps answers focused and factual |
| Groq model | `openai/gpt-oss-120b` | Generates the recruiter-facing response |

## Deploying on Streamlit Cloud

1. Push the project to GitHub.
2. Create a new Streamlit Cloud app using `app.py` as the entry point.
3. Add this secret under **App settings -> Secrets**:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

4. Keep `abcde14.pdf` in the repository so the assistant can load its profile.

## Troubleshooting

**The app says the profile could not be loaded**

Check that `abcde14.pdf` exists beside `app.py` and that the filename matches `DEFAULT_CV`.

**Groq authentication fails**

Check that `GROQ_API_KEY` exists in `.env` locally or Streamlit Secrets in deployment. Do not include quotes or extra spaces in the `.env` value.

**The first startup is slow**

The local embedding model downloads on its first use. Later runs use the local model cache.

**The terminal command fails with a file-not-found error**

Run `python chatbot.py`, not `python run chatbot.py`. The latter asks Python to open a file named `run`.

## Privacy

Keep the CV and API key private. The CV is processed locally for retrieval, while the selected context for each question is sent to Groq to generate the answer.

---