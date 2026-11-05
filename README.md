# ✦ Aurora AI – Intelligent Multimodal Chatbot

Aurora AI is an intelligent conversational chatbot that combines **LLM-powered conversations, Multimodal Retrieval-Augmented Generation (RAG), and external API integration** in a unified application.

Built using LangGraph, Gemini, and Streamlit, Aurora AI supports contextual conversations, PDF-based question answering, and real-time train information retrieval through the RailRadar API.

## 🚀 Key Features

- **Conversational AI:** Context-aware conversations powered by Google Gemini and LangGraph.
- **Multimodal RAG:** Upload PDF documents and ask questions using text and image-based retrieval.
- **Document-Grounded Responses:** Retrieves relevant document content to generate contextually grounded answers.
- **Session-Isolated Documents:** Keeps uploaded document retrieval scoped to the current user session.
- **RailRadar API Integration:** Retrieves train information through external API integration when users ask train-related questions.
- **Streaming Responses:** Displays AI-generated responses progressively.
- **Unified Chatbot:** Combines conversational AI, document question answering, and external API capabilities in one interface.

## 🏗️ Architecture

Aurora AI uses a unified architecture in which LangGraph manages conversational flow and routes requests to the appropriate functionality.

```text
                    User
                     |
                     v
              Streamlit UI
                     |
                     v
              LangGraph Workflow
                     |
          +----------+----------+
          |                     |
          v                     v
   General AI Chat         Query Routing
          |                     |
          v              +------+------+
       Gemini             |             |
                          v             v
                     Multimodal    RailRadar API
                        RAG
                          |
                   +------+------+
                   |             |
                   v             v
              Text Retrieval  Image Retrieval
                   |             |
                   +------+------+
                          |
                          v
                   Context Fusion
                          |
                          v
                  Gemini Response
                          |
                          v
                    User Response
```

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Frontend | Streamlit |
| LLM | Google Gemini |
| Agent / Workflow Orchestration | LangGraph |
| LLM Integration | LangChain |
| Vector Database | ChromaDB |
| Document Processing | PyMuPDF |
| Embeddings | Google Gemini Embeddings |
| External API Integration | RailRadar API |
| Deployment | Render |

## 📂 Project Structure

```text
aurora_ai/
│
├── chatbot_app.py
├── chatbot_backend.py
├── railway_service.py
├── aurora-favicon.svg
├── requirements.txt
│
├── multimodal_rag/
│   ├── __init__.py
│   ├── api/
│   └── app/
│       ├── config.py
│       ├── generation/
│       ├── indexing/
│       ├── ingestion/
│       ├── retrieval/
│       └── workflow/
│
└── data/
    ├── indexes/
    ├── metadata/
    ├── processed/
    └── uploads/
```

## ⚙️ Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/aurora-ai.git
cd aurora-ai
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
GOOGLE_API_KEY=your_gemini_api_key
```

Never commit your `.env` file or expose API credentials publicly.

### 5. Run the application

```bash
streamlit run chatbot_app.py
```

Open the local URL displayed in your terminal to access Aurora AI.

## ☁️ Deployment

Aurora AI is deployed as a Streamlit application on **Render**.

Deployment configuration:

| Setting | Value |
|---|---|
| Service Type | Web Service |
| Runtime | Python 3 |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `streamlit run chatbot_app.py --server.address 0.0.0.0 --server.port $PORT` |

Configure the following environment variables in Render:

- `GEMINI_API_KEY`
- `GOOGLE_API_KEY`
- `PYTHON_VERSION` (3.12.4)

**Deployment Note:** The application uses temporary local storage for uploaded documents and generated indexes. On Render's free tier, the filesystem is ephemeral, so uploaded documents and indexes may not survive service restarts, spin-downs, or redeployments.

## 🔮 Future Enhancements

- Persistent document and vector storage.
- Enhanced retrieval and reranking strategies.
- Improved multimodal retrieval and response evaluation.
- Expanded external API and tool integrations.
- Advanced conversational memory and personalization.

## 👨‍💻 Author

Developed as an AI engineering project exploring conversational AI, LangGraph workflows, Multimodal RAG, and real-world API integration.