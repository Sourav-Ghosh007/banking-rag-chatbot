#  Banking RAG Chatbot

> Multi-agent banking assistant powered by LangGraph, GPT-4o, Hybrid RAG, and Streamlit.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green)
![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-orange)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector--Store-purple)

---

##  What It Does

Ask questions about your banking data in plain English:

- *"What is my total transaction amount?"* → ₹97,050
-  *"Calculate EMI for ₹5 lakh loan at 8.5% for 5 years"* → ₹10,224/month
-  *"What are the current FD interest rates?"* → Fetches from uploaded rates.xlsx
-  *"Send an email to my manager about loan approval"* → Sends via Gmail OAuth2

---

##  Architecture

```
User Question
      ↓
Orchestrator Agent (GPT-4o classifies intent)
      ↓
┌──────────────────────────────────────────┐
│  account_agent  │  loan_agent            │
│  rates_agent    │  comms_agent (notify)  │
│  compliance_agent │ scheduling_agent     │
└──────────────────────────────────────────┘
      ↓
Hybrid Search Pipeline:
  ChromaDB (Dense) + BM25 (Sparse) → RRF Fusion → Cross-Encoder Reranker
      ↓
SQLite Aggregation Engine (for totals, averages, counts)
      ↓
GPT-4o Final Answer with Source Citations
```

---

##  How to Run

### Prerequisites
- Python 3.10+
- OpenAI API Key (get from platform.openai.com)

### Step 1 — Clone the repository
```bash
git clone https://github.com/Sourav-Ghosh007/banking-rag-chatbot.git
cd banking-rag-chatbot
```

### Step 2 — Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
pip install chromadb==0.5.23 numpy==1.26.4 bcrypt==4.0.1 --force-reinstall
pip install huggingface_hub==0.20.3 transformers==4.37.0 sentence-transformers==2.4.0 --force-reinstall
pip install email-validator python-dotenv aiosqlite
```

### Step 4 — Create .env file
Create a `.env` file in the project root:
```
OPENAI_API_KEY=your-openai-api-key-here
OPENAI_MODEL=gpt-4o
SECRET_KEY=your-random-secret-key
FERNET_KEY=ZmDfcTF7_60GrrY167zsiPd67pEvs0aGOv2oasOM1Pg=
FRONTEND_URL=http://localhost:8501
API_URL=http://localhost:8000
```

### Step 5 — Initialize database and load sample data
```bash
# Set your API key first (Windows)
set OPENAI_API_KEY=your-openai-api-key-here

python scripts/init_db.py
python scripts/seed_data.py
```

### Step 6 — Start Backend (Terminal 1)
```bash
set OPENAI_API_KEY=your-openai-api-key-here
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### Step 7 — Start Frontend (Terminal 2)
```bash
set OPENAI_API_KEY=your-openai-api-key-here
python -m streamlit run frontend/app.py --server.port 8501
```

### Step 8 — Open Browser
| URL | Purpose |
|---|---|
| http://localhost:8501 | Chatbot UI |
| http://localhost:8000/docs | API Documentation |

### Login Credentials
```
Email:    demo@bankingchatbot.com
Password: Demo@1234
```

---

##  Example Questions

After uploading `data/transactions.csv` via the Files page:

| Question | What It Does |
|---|---|
| What is my total transaction amount? | SQL SUM aggregation |
| Calculate EMI for ₹5 lakh at 8.5% for 5 years | Deterministic EMI formula |
| What are the current FD interest rates? | RAG retrieval from rates.xlsx |
| Show me transactions above ₹10,000 | SQL filter query |
| What is the USD to INR exchange rate? | RAG retrieval from rates data |

---

##  Project Structure

```
banking-rag-chatbot/
├── backend/
│   ├── agents/          # LangGraph agents
│   │   ├── orchestrator.py   # Router — classifies intent
│   │   ├── account.py        # Transaction queries
│   │   ├── loan_agent.py     # EMI, eligibility
│   │   ├── rates_agent.py    # Interest rates, forex
│   │   ├── notify.py         # Gmail, Calendar, Slack
│   │   ├── compliance.py     # AML, flagging
│   │   ├── scheduling.py     # Appointments
│   │   └── graph.py          # LangGraph assembly
│   ├── rag/             # RAG pipeline
│   │   ├── ingest.py         # File ingestion (xlsx/csv only)
│   │   ├── embedder.py       # OpenAI embeddings
│   │   ├── retriever.py      # Hybrid search (Dense+BM25+RRF)
│   │   ├── reranker.py       # Cross-encoder reranking
│   │   ├── sql_engine.py     # SQLite aggregations
│   │   └── metrics.py        # RAG metrics logging
│   ├── routers/         # FastAPI endpoints
│   │   ├── auth.py           # Login, register, OAuth2
│   │   ├── chat.py           # Chat endpoint
│   │   └── files.py          # File upload
│   ├── db/              # Database layer
│   │   ├── session.py        # SQLite connection
│   │   └── crud.py           # DB operations
│   ├── auth/            # OAuth2 security
│   │   ├── oauth.py          # Google + Slack OAuth2
│   │   └── pkce.py           # PKCE security
│   └── main.py          # FastAPI entry point
├── frontend/
│   ├── app.py           # Main Streamlit app + login
│   └── pages/
│       ├── chat.py      # Chat interface
│       ├── files.py     # File upload page
│       ├── oauth.py     # Connect Gmail/Slack
│       └── log.py       # Audit logs
├── data/                # Sample banking data
│   ├── transactions.csv
│   ├── sample_loans.xlsx
│   └── rates.xlsx
├── scripts/
│   ├── init_db.py       # Create database tables
│   └── seed_data.py     # Load demo user + sample data
└── tests/
    ├── test_agents.py
    ├── test_auth.py
    └── test_ingest.py
```

---

##  Key Features

| Feature | Implementation |
|---|---|
| Multi-agent routing | LangGraph with GPT-4o intent classification |
| Dense search | ChromaDB + OpenAI text-embedding-3-small |
| Sparse search | BM25Okapi keyword matching |
| Result fusion | Reciprocal Rank Fusion (RRF, k=60) |
| Reranking | Cross-Encoder ms-marco-MiniLM-L-6-v2 |
| Aggregations | SQLite in-memory (never LLM estimation) |
| Authentication | JWT tokens (python-jose) |
| OAuth2 | Gmail + Google Calendar + Slack |
| Token security | Fernet encryption at rest |
| File validation | .xlsx and .csv only — PDF rejected |

---

##  Tech Stack

| Component | Technology | Version |
|---|---|---|
| Backend | FastAPI | 0.109 |
| Agents | LangGraph + GPT-4o | latest |
| Vector DB | ChromaDB | 0.5.23 |
| Embeddings | OpenAI text-embedding-3-small | latest |
| Reranker | sentence-transformers CrossEncoder | 2.4.0 |
| Database | SQLite (aiosqlite) | built-in |
| Frontend | Streamlit | 1.31 |
| Auth | JWT + OAuth2 + Fernet | - |
| Language | Python | 3.12 |

---

##  Important Notes

- **Never commit your `.env` file** — it contains your API key
- After every backend restart, re-upload your data files via the Files page
- The SQLite data is in-memory and resets on restart (by design for assessment)
- ChromaDB data persists in `./chroma_store/` folder

---

*Built as a technical assessment for a Banking AI role.*
