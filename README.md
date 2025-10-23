# Fund Performance Analysis System - Coding Challenge

## Time Estimate: 1 Week (Senior Developer)

## Overview

Build an **AI-powered fund performance analysis system** that enables Limited Partners (LPs) to:
1. Upload fund performance PDF documents
2. Automatically parse and extract structured data (tables → SQL, text → Vector DB)
3. Ask natural language questions about fund metrics (DPI, IRR, etc.)
4. Get accurate answers powered by RAG (Retrieval Augmented Generation) and SQL calculations

---

## Business Context

As an LP, you receive quarterly fund performance reports in PDF format. These documents contain:
- **Capital Call tables**: When and how much capital was called
- **Distribution tables**: When and how much was distributed back to LPs
- **Adjustment tables**: Rebalancing entries (recallable distributions, capital call adjustments)
- **Text explanations**: Definitions, investment strategies, market commentary

**Your task**: Build a system that automatically processes these documents and answers questions like:
- "What is the current DPI of this fund?"
- "Has the fund returned all invested capital to LPs?"
- "What does 'Paid-In Capital' mean in this context?"
- "Show me all capital calls in 2024"

---

## What's Provided (Starting Point)

This repository contains a **project scaffold** to help you get started quickly:

### Infrastructure Setup
- Docker Compose configuration (PostgreSQL, Redis, Backend, Frontend)
- Database schema and models (SQLAlchemy)
- Basic API structure (FastAPI with endpoints)
- Frontend boilerplate (Next.js with TailwindCSS)
- Environment configuration

### Basic UI Components
- Upload page layout
- Chat interface layout
- Fund dashboard layout
- Navigation and routing

### Metrics Calculation (Provided)
- **DPI (Distributions to Paid-In)** - Fully implemented
- **IRR (Internal Rate of Return)** - Using numpy-financial
- **PIC (Paid-In Capital)** - With adjustments
- **Calculation breakdown API** - Shows all cash flows and transactions for debugging
- Located in: `backend/app/services/metrics_calculator.py`

**Debugging Features:**
- View all capital calls, distributions, and adjustments used in calculations
- See cash flow timeline for IRR calculation
- Verify intermediate values (total calls, total distributions, etc.)
- Trace calculation steps with detailed explanations

### Sample Data (Provided)
- **Reference PDF**: ILPA metrics explanation document
- **Sample Fund Report**: Generated with realistic data
- **PDF Generator Script**: `files/create_sample_pdf.py`
- **Expected Results**: Documented for validation

### What's NOT Implemented (Your Job)

The following **core functionalities are NOT implemented** and need to be built by you:

#### 1. Document Processing Pipeline (Phase 2) - **CRITICAL**
- [x] PDF parsing with pdfplumber (integrate and test)
- [x] Table detection and extraction logic
- [x] Intelligent table classification (capital calls vs distributions vs adjustments)
- [x] Data validation and cleaning
- [x] Error handling for malformed PDFs
- [x] Background task processing (Celery integration)

**Files to implement:**
- `backend/app/services/document_processor.py` (skeleton provided)
- `backend/app/services/table_parser.py` (needs implementation)

#### 2. Vector Store & RAG System (Phase 3) - **CRITICAL**
- [x] Text chunking strategy implementation
- [x] embedding generation
- [x] FAISS index creation and management
- [x] Semantic search implementation
- [x] Context retrieval for LLM
- [x] Prompt engineering for accurate responses

**Files to implement:**
- `backend/app/services/vector_store.py` (pgvector implementation with TODOs)
- `backend/app/services/rag_engine.py` (needs implementation)

**Note**: This project uses **pgvector** instead of FAISS. pgvector is a PostgreSQL extension that stores vectors directly in your database, eliminating the need for a separate vector database.

#### 3. Query Engine & Intent Classification (Phase 3-4) - **CRITICAL**
- [x] Intent classifier (calculation vs definition vs retrieval)
- [x] Query router logic
- [x] LLM integration 
- [x] Response formatting
- [x] Source citation
- [x] Conversation context management

**Files to implement:**
- `backend/app/services/query_engine.py` (needs implementation)

#### 4. Integration & Testing
- [ ] End-to-end document upload flow
- [ ] API integration tests
- [ ] Error handling and logging
- [ ] Performance optimization

**Note**: Metrics calculation is already implemented. You can focus on document processing and RAG!

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend (Next.js)                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Upload     │  │     Chat     │  │   Dashboard  │     │
│  │     Page     │  │  Interface   │  │     Page     │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└────────────────────────┬────────────────────────────────────┘
                         │ REST API
┌────────────────────────▼────────────────────────────────────┐
│                    Backend (FastAPI)                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Document Processor                     │   │
│  │  ┌──────────────┐         ┌──────────────┐        │   │
│  │  │   Docling    │────────▶│  Table       │        │   │
│  │  │   Parser     │         │  Extractor   │        │   │
│  │  └──────────────┘         └──────┬───────┘        │   │
│  │                                   │                 │   │
│  │  ┌──────────────┐         ┌──────▼───────┐        │   │
│  │  │   Text       │────────▶│  Embedding   │        │   │
│  │  │   Chunker    │         │  Generator   │        │   │
│  │  └──────────────┘         └──────────────┘        │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Query Engine (RAG)                     │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────┐ │   │
│  │  │   Intent     │─▶│   Vector     │─▶│   LLM    │ │   │
│  │  │  Classifier  │  │   Search     │  │ Response │ │   │
│  │  └──────────────┘  └──────────────┘  └──────────┘ │   │
│  │                                                      │   │
│  │  ┌──────────────┐  ┌──────────────┐               │   │
│  │  │  Metrics     │─▶│     SQL      │               │   │
│  │  │ Calculator   │  │   Queries    │               │   │
│  │  └──────────────┘  └──────────────┘               │   │
│  └─────────────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
┌───────▼────────┐ ┌────▼─────┐ ┌────────▼────────┐
│   PostgreSQL   │ │  FAISS   │ │     Redis       │
│  (Structured)  │ │ (Vectors)│ │  (Task Queue)   │
└────────────────┘ └──────────┘ └─────────────────┘
```

---

## Data Model

### PostgreSQL Schema

#### `funds` table
```sql
CREATE TABLE funds (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    gp_name VARCHAR(255),
    fund_type VARCHAR(100),
    vintage_year INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### `capital_calls` table
```sql
CREATE TABLE capital_calls (
    id SERIAL PRIMARY KEY,
    fund_id INTEGER REFERENCES funds(id),
    call_date DATE NOT NULL,
    call_type VARCHAR(100),
    amount DECIMAL(15, 2) NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### `distributions` table
```sql
CREATE TABLE distributions (
    id SERIAL PRIMARY KEY,
    fund_id INTEGER REFERENCES funds(id),
    distribution_date DATE NOT NULL,
    distribution_type VARCHAR(100),
    is_recallable BOOLEAN DEFAULT FALSE,
    amount DECIMAL(15, 2) NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### `adjustments` table
```sql
CREATE TABLE adjustments (
    id SERIAL PRIMARY KEY,
    fund_id INTEGER REFERENCES funds(id),
    adjustment_date DATE NOT NULL,
    adjustment_type VARCHAR(100),
    category VARCHAR(100),
    amount DECIMAL(15, 2) NOT NULL,
    is_contribution_adjustment BOOLEAN DEFAULT FALSE,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### `documents` table
```sql
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    fund_id INTEGER REFERENCES funds(id),
    file_name VARCHAR(255) NOT NULL,
    file_path VARCHAR(500),
    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    parsing_status VARCHAR(50) DEFAULT 'pending',
    error_message TEXT
);
```

---

## Required Features (Phase 1-4)

### Phase 1: Core Infrastructure
- [ ] Docker setup with PostgreSQL, Redis
- [ ] FastAPI backend with CRUD endpoints
- [ ] Next.js frontend with basic layout
- [ ] Database schema implementation
- [ ] Environment configuration

### Phase 2: Document Processing
- [ ] File upload API endpoint
- [ ] Docling integration for PDF parsing
- [ ] Table extraction and SQL storage
- [ ] Text chunking and embedding
- [ ] Parsing status tracking

### Phase 3: Vector Store & RAG
- [ ] pgvector setup (PostgreSQL extension)
- [ ] Embedding generation (OpenAI/local)
- [ ] Similarity search using pgvector operators
- [ ] LangChain integration
- [ ] Basic chat interface

### Phase 4: Fund Metrics Calculation
- [ ] DPI calculation function
- [ ] IRR calculation function
- [ ] Metrics API endpoints
- [ ] Query engine integration

---

## Bonus Features (Phase 5-6)

### Phase 5: Dashboard & Polish
- [ ] Fund list page with metrics
- [ ] Fund detail page with charts
- [ ] Transaction tables with pagination
- [ ] Error handling improvements
- [ ] Loading states

### Phase 6: Advanced Features
- [ ] Conversation history
- [ ] Multi-fund comparison
- [ ] Excel export
- [ ] Custom calculation formulas
- [ ] Test coverage (50%+)

---

## Getting Started

### Prerequisites
- Docker & Docker Compose
- Node.js 18+ (for local frontend development)
- Python 3.11+ (for local backend development)
- OpenAI API key (or use free alternatives - see below)

### Quick Start

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd fund-analysis-system
```

2. **Set up environment variables**
```bash
# Copy example env file
cp .env.example .env

# Edit .env and add your API keys
# OPENAI_API_KEY=sk-...
# DATABASE_URL=postgresql://user:password@localhost:5432/funddb
```

3. **Start with Docker Compose**
```bash
docker-compose up -d
```

4. **Access the application**
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

5. **Upload sample document**
- Navigate to http://localhost:3000/upload
- Upload the provided PDF: `files/ILPA based Capital Accounting and Performance Metrics_ PIC, Net PIC, DPI, IRR  .pdf`
- Wait for parsing to complete

6. **Start asking questions**
- Go to http://localhost:3000/chat
- Try: "What is DPI?"
- Try: "Calculate the current DPI for this fund"

---

## Project Structure

```
fund-analysis-system/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── endpoints/
│   │   │   │   ├── documents.py
│   │   │   │   ├── funds.py
│   │   │   │   ├── chat.py
│   │   │   │   └── metrics.py
│   │   │   └── deps.py
│   │   ├── core/
│   │   │   ├── config.py
│   │   │   └── security.py
│   │   ├── db/
│   │   │   ├── base.py
│   │   │   ├── session.py
│   │   │   └── init_db.py
│   │   ├── models/
│   │   │   ├── fund.py
│   │   │   ├── transaction.py
│   │   │   └── document.py
│   │   ├── schemas/
│   │   │   ├── fund.py
│   │   │   ├── transaction.py
│   │   │   ├── document.py
│   │   │   └── chat.py
│   │   ├── services/
│   │   │   ├── document_processor.py
│   │   │   ├── table_parser.py
│   │   │   ├── vector_store.py
│   │   │   ├── query_engine.py
│   │   │   └── metrics_calculator.py
│   │   └── main.py
│   ├── tests/
│   │   ├── test_document_processor.py
│   │   ├── test_metrics.py
│   │   └── test_api.py
│   ├── requirements.txt
│   ├── Dockerfile
│   └── alembic/
│       └── versions/
├── frontend/
│   ├── app/
│   │   ├── layout.tsx
│   │   ├── page.tsx
│   │   ├── upload/
│   │   │   └── page.tsx
│   │   ├── chat/
│   │   │   └── page.tsx
│   │   └── funds/
│   │       ├── page.tsx
│   │       └── [id]/
│   │           └── page.tsx
│   ├── components/
│   │   ├── ui/
│   │   │   ├── button.tsx
│   │   │   ├── card.tsx
│   │   │   ├── input.tsx
│   │   │   └── ...
│   │   ├── FileUpload.tsx
│   │   ├── ChatInterface.tsx
│   │   ├── FundMetrics.tsx
│   │   └── TransactionTable.tsx
│   ├── lib/
│   │   ├── api.ts
│   │   └── utils.ts
│   ├── package.json
│   ├── tsconfig.json
│   ├── next.config.js
│   ├── tailwind.config.ts
│   └── Dockerfile
├── docker-compose.yml
├── .env.example
├── README.md
└── docs/
    ├── API.md
    ├── ARCHITECTURE.md
    └── CALCULATIONS.md
```

---

## API Endpoints

### Documents
```
POST   /api/documents/upload
GET    /api/documents/{doc_id}/status
GET    /api/documents/{doc_id}
DELETE /api/documents/{doc_id}
```

### Funds
```
GET    /api/funds
POST   /api/funds
GET    /api/funds/{fund_id}
GET    /api/funds/{fund_id}/transactions
GET    /api/funds/{fund_id}/metrics
```

### Chat
```
POST   /api/chat/query
GET    /api/chat/conversations/{conv_id}
POST   /api/chat/conversations
```

See [API.md](docs/API.md) for detailed documentation.

---

## Fund Metrics Formulas

### Paid-In Capital (PIC)
```
PIC = Total Capital Calls - Adjustments
```

### DPI (Distribution to Paid-In)
```
DPI = Cumulative Distributions / PIC
```

### IRR (Internal Rate of Return)
```
IRR = Rate where NPV of all cash flows = 0
Uses numpy-financial.irr() function
```

See [CALCULATIONS.md](docs/CALCULATIONS.md) for detailed formulas.

---

## Testing

### Run Backend Tests
```bash
cd backend
pytest tests/ -v --cov=app
```

### Run Frontend Tests
```bash
cd frontend
npm test
```

### Test Document Upload
```bash
curl -X POST "http://localhost:8000/api/documents/upload" \
  -F "file=@files/sample_fund_report.pdf"
```

### Test Chat Query
```bash
curl -X POST "http://localhost:8000/api/chat/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the current DPI?",
    "fund_id": 1
  }'
```

---

## Implementation Guidelines

### Document Parsing Strategy
1. Use **Docling** to extract document structure
2. Identify tables by headers (e.g., "Capital Call", "Distribution")
3. Parse table rows and map to SQL schema
4. Extract text paragraphs for vector storage
5. Handle parsing errors gracefully

### RAG Pipeline
1. **Retrieval**: Vector similarity search (top-k=5)
2. **Augmentation**: Combine retrieved context with SQL data
3. **Generation**: LLM generates answer with citations

### Calculation Logic
- Always validate input data before calculation
- Handle edge cases (zero PIC, missing data)
- Return calculation breakdown for transparency
- Cache results for performance

---

## Sample Questions to Test

### Definitions
- "What does DPI mean?"
- "Explain Paid-In Capital"
- "What is a recallable distribution?"

### Calculations
- "What is the current DPI?"
- "Calculate the IRR for this fund"
- "Has the fund returned all capital to LPs?"

### Data Retrieval
- "Show me all capital calls in 2024"
- "What was the largest distribution?"
- "List all adjustments"

### Complex Queries
- "How is the fund performing compared to industry benchmarks?"
- "What percentage of distributions were recallable?"
- "Explain the trend in capital calls over time"

---

## Evaluation Criteria

### Must-Have (Pass/Fail)
- Document upload and parsing works
- Tables correctly stored in SQL
- Text stored in vector DB
- DPI calculation is accurate
- Basic RAG Q&A works
- Application runs via Docker

### Code Quality (40 points)
- **Structure**: Modular, separation of concerns (10pts)
- **Readability**: Clear naming, comments (10pts)
- **Error Handling**: Try-catch, validation (10pts)
- **Type Safety**: TypeScript, Pydantic (10pts)

### Functionality (30 points)
- **Parsing Accuracy**: Table recognition (10pts)
- **Calculation Accuracy**: DPI, IRR (10pts)
- **RAG Quality**: Relevant answers (10pts)

### UX/UI (20 points)
- **Intuitiveness**: Easy to use (10pts)
- **Feedback**: Loading, errors, success (5pts)
- **Design**: Clean, consistent (5pts)

### Documentation (10 points)
- **README**: Setup instructions (5pts)
- **API Docs**: Endpoint descriptions (3pts)
- **Architecture**: Diagrams (2pts)

### Bonus Points (up to 20 points)
- Dashboard implementation (+5pts)
- Charts/visualization (+3pts)
- Multi-fund support (+3pts)
- Test coverage (+5pts)
- Live deployment (+4pts)

---

## Submission Requirements

### What to Submit
1. **GitHub Repository** (public or private with access)
2. **Complete source code** (backend + frontend)
3. **Docker configuration** (docker-compose.yml)
4. **Documentation** (README, API docs, architecture)
5. **Sample data** (at least one test PDF)

### README Must Include
- Project overview
- Tech stack
- Setup instructions (Docker)
- Environment variables (.env.example)
- API testing examples
- Features implemented
- Known limitations
- Future improvements
- Screenshots (minimum 3)

### Timeline
- **Recommended**: 1 week (Phase 1-4)
- **Maximum**: 2 weeks (Phase 1-6)

### How to Submit
1. Push code to GitHub
2. Test that `docker-compose up` works
3. Send repository URL via email
4. Include any special instructions

---

## Tech Stack

### Backend
- **Framework**: FastAPI (Python 3.11+)
- **Document Parser**: Docling
- **Vector DB**: pgvector (PostgreSQL extension)
- **SQL DB**: PostgreSQL 15+
- **ORM**: SQLAlchemy
- **LLM Framework**: LangChain
- **LLM**: OpenAI GPT-4 or any LLM
- **Embeddings**: OpenAI text-embedding-3-small
- **Task Queue**: Celery + Redis

### Frontend
- **Framework**: Next.js 14 (App Router)
- **UI Library**: shadcn/ui + Tailwind CSS
- **State**: Zustand or React Context
- **Data Fetching**: TanStack Query
- **Charts**: Recharts
- **File Upload**: react-dropzone

### Infrastructure
- **Development**: Docker + Docker Compose
- **Deployment**: Your choice (Vercel, Railway, AWS, etc.)

---

## Troubleshooting

### Document Parsing Issues
**Problem**: Docling can't extract tables
**Solution**: 
- Check PDF format (ensure it's not scanned image)
- Add fallback parsing logic
- Manually define table structure patterns

### LLM API Costs
**Problem**: OpenAI API is expensive
**Solution**: Use free alternatives (see "Free LLM Options" section below)
- Use caching for repeated queries
- Use cheaper models (gpt-3.5-turbo)
- Use local LLM (Ollama) for development

### IRR Calculation Errors
**Problem**: IRR returns NaN or extreme values
**Solution**:
- Validate cash flow sequence
- Check for missing dates
- Handle edge cases (all positive/negative flows)

### CORS Issues
**Problem**: Frontend can't call backend API
**Solution**:
- Add CORS middleware in FastAPI
- Allow origin: http://localhost:3000
- Check network configuration in Docker

---

## Free LLM Options

You don't need to pay for OpenAI API! Here are free alternatives:

### Option 1: Ollama (Recommended for Development)

**Completely free, runs locally on your machine**

1. **Install Ollama**
```bash
# Mac
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows
# Download from https://ollama.com/download
```

2. **Download a model**
```bash
# Llama 3.2 (3B - fast, good for development)
ollama pull llama3.2

# Or Llama 3.1 (8B - better quality)
ollama pull llama3.1

# Or Mistral (7B - good balance)
ollama pull mistral
```

3. **Update your .env**
```bash
# Use Ollama instead of OpenAI
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
```

4. **Modify your code to use Ollama**
```python
# In backend/app/services/query_engine.py
from langchain_community.llms import Ollama

llm = Ollama(
    base_url="http://localhost:11434",
    model="llama3.2"
)
```

**Pros**: Free, private, no API limits, works offline
**Cons**: Requires decent hardware (8GB+ RAM), slower than cloud APIs

---

### Option 2: Google Gemini (Free Tier)

**Free tier: 60 requests per minute**

1. **Get free API key**
   - Go to https://makersuite.google.com/app/apikey
   - Click "Create API Key"
   - Copy your key

2. **Install package**
```bash
pip install langchain-google-genai
```

3. **Update .env**
```bash
GOOGLE_API_KEY=your-gemini-api-key
LLM_PROVIDER=gemini
```

4. **Use in code**
```python
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)
```

**Pros**: Free, fast, good quality
**Cons**: Rate limits, requires internet

---

### Option 3: Groq (Free Tier)

**Free tier: Very fast inference, generous limits**

1. **Get free API key**
   - Go to https://console.groq.com
   - Sign up and get API key

2. **Install package**
```bash
pip install langchain-groq
```

3. **Update .env**
```bash
GROQ_API_KEY=your-groq-api-key
LLM_PROVIDER=groq
```

4. **Use in code**
```python
from langchain_groq import ChatGroq

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="mixtral-8x7b-32768"  # or "llama3-70b-8192"
)
```

**Pros**: Free, extremely fast, good quality
**Cons**: Rate limits, requires internet

---

### Option 4: Hugging Face (Free)

**Free inference API**

1. **Get free token**
   - Go to https://huggingface.co/settings/tokens
   - Create a token

2. **Update .env**
```bash
HUGGINGFACE_API_TOKEN=your-hf-token
LLM_PROVIDER=huggingface
```

3. **Use in code**
```python
from langchain_community.llms import HuggingFaceHub

llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_TOKEN")
)
```

**Pros**: Free, many models available
**Cons**: Can be slow, rate limits

---

### Comparison Table

| Provider | Cost | Speed | Quality | Setup Difficulty |
|----------|------|-------|---------|------------------|
| **Ollama** | Free | Medium | Good | Easy |
| **Gemini** | Free | Fast | Very Good | Very Easy |
| **Groq** | Free | Very Fast | Good | Very Easy |
| **Hugging Face** | Free | Slow | Varies | Easy |
| OpenAI | Paid | Fast | Excellent | Very Easy |

### Recommended Setup for This Project

**For Development/Testing:**
- Use **Ollama** with `llama3.2` (free, no limits)

**For Production/Demo:**
- Use **Groq** or **Gemini** (free tier is generous)

**If you have budget:**
- Use **OpenAI GPT-4** (best quality)

---

## Sample Data

### Provided Sample Files

Located in `files/` directory:

1. **`ILPA based Capital Accounting and Performance Metrics_ PIC, Net PIC, DPI, IRR.pdf`**
   - Reference document explaining fund metrics
   - Contains definitions of PIC, DPI, IRR, TVPI
   - Use this to test text extraction and RAG

### Sample Data You Should Create

For comprehensive testing, you should create **mock fund performance reports** with:

#### Example Capital Call Table
```
Date       | Call Number | Amount      | Description
-----------|-------------|-------------|------------------
2023-01-15 | Call 1      | $5,000,000  | Initial Capital
2023-06-20 | Call 2      | $3,000,000  | Follow-on
2024-03-10 | Call 3      | $2,000,000  | Bridge Round
```

#### Example Distribution Table
```
Date       | Type        | Amount      | Recallable | Description
-----------|-------------|-------------|------------|------------------
2023-12-15 | Return      | $1,500,000  | No         | Exit: Company A
2024-06-20 | Income      | $500,000    | No         | Dividend
2024-09-10 | Return      | $2,000,000  | Yes        | Partial Exit: Company B
```

#### Example Adjustment Table
```
Date       | Type                | Amount    | Description
-----------|---------------------|-----------|------------------
2024-01-15 | Recallable Dist     | -$500,000 | Recalled distribution
2024-03-20 | Capital Call Adj    | $100,000  | Fee adjustment
```

### Expected Test Results

For the sample data above:
- **Total Capital Called**: $10,000,000
- **Total Distributions**: $4,000,000
- **Net PIC**: $10,100,000 (after adjustments)
- **DPI**: 0.40 (4M / 10M)
- **IRR**: ~8-12% (depends on exact dates)

### Creating Test PDFs

#### Option 1: Use Provided Script (Recommended)

We've included a Python script to generate sample PDFs:

```bash
cd files/
pip install reportlab
python create_sample_pdf.py
```

This creates `Sample_Fund_Performance_Report.pdf` with:
- Capital calls table (4 entries)
- Distributions table (4 entries)
- Adjustments table (3 entries)
- Performance summary with definitions

#### Option 2: Create Your Own

You can create PDFs using:
- Google Docs/Word → Export as PDF
- Python libraries (reportlab, fpdf)
- Online PDF generators

**Tip**: Start with simple, well-structured tables before handling complex layouts.

---

## Reference Materials

- **Docling**: https://github.com/DS4SD/docling
- **LangChain RAG**: https://python.langchain.com/docs/use_cases/question_answering/
- **FAISS**: https://faiss.ai/
- **ILPA Guidelines**: https://ilpa.org/
- **PE Metrics**: https://www.investopedia.com/terms/d/dpi.asp

---

## Tips for Success

1. **Start Simple**: Get Phase 1-4 working before adding features
2. **Test Early**: Test document parsing with sample PDF immediately
3. **Use Tools**: Leverage LangChain, shadcn/ui to save time
4. **Focus on Core**: Perfect the RAG pipeline and calculations first
5. **Document Well**: Clear README helps evaluators understand your work
6. **Handle Errors**: Graceful error handling shows maturity
7. **Ask Questions**: If requirements are unclear, document your assumptions

---

## Support

For questions about this coding challenge:
- Open an issue in this repository
- Email: [your-contact-email]

---

**Good luck! Build something amazing!**

---

## Appendix: Calculation Formulas (from PDF)

### Paid-In Capital (PIC)
```
PIC = Capital Contributions (Gross) - Adjustments
```

### DPI (Distribution to Paid-In)
```
DPI = Cumulative Distributions / PIC
```

### Cumulative Distributions
```
Cumulative Distributions = 
  Return of Capital + 
  Dividends Paid + 
  Interest Paid + 
  Realized Gains Distributed - 
  (Fees & Carried Interest Withheld)
```

### Adjustments
```
Adjustments = Σ (Rebalance of Distribution + Rebalance of Capital Call)
```

#### Rebalance of Distribution
- **Nature**: Clawback of over-distributed amounts
- **Recording**: Contribution (-)
- **DPI Impact**: Numerator ↓, Denominator ↑ → DPI ↓

#### Rebalance of Capital Call
- **Nature**: Refund of over-called capital
- **Recording**: Distribution (+)
- **DPI Impact**: Denominator ↓, Numerator unchanged → Requires flag to prevent DPI inflation

---

**Version**: 1.0  
**Last Updated**: 2025-10-06  
**Author**: InterOpera-Apps Hiring Team
