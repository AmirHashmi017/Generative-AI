## Genrative AI - LangChain And Agnetic AI - LangGraph

This repository documents my **4+ month deep dive into Generative AI and Agentic AI**, covering foundational concepts, advanced frameworks, and production-grade projects built using **LangChain, LangGraph, MCP, and modern AI architectures**.

The focus of this journey was not just *using LLMs*, but designing **autonomous, traceable, multi-agent systems** that can reason, plan, and act with minimal human intervention.

---

## 🧩 LangChain — Generative AI Topics

- LangChain Tools
- LangChain Agents
- LangChain Chains
- LangChain Document Loaders
- RAG using LangChain
- LangChain Models
- LangChain Output Parsers
- LangChain Prompts
- LangChain Retrievers
- LangChain Runnables
- LangChain Structured Output
- LangChain Text Splitters
- LangChain Tool Calling
- Vector Stores:
  - FAISS
  - Chroma DB

---

## 🤖 LangGraph — Agentic AI Topics

- Sequential Workflows
- Conditional Workflows
- Parallel Workflows
- Iterative Workflows
- Sub-Graphs
- Human-in-the-Loop (HITL)
- Short-Term Memory (STM)
- STM Summarization & Context Trimming
- Long-Term Memory
- Persistence & Checkpointers
- Streaming Agent Execution
- MCP Client using LangGraph
- MCP Server using Fast MCP

---

###  LangSmith — Observability, Debugging & Tracing
- End-to-end **traceability of agent executions**
- Debugging **multi-agent decision paths**
- Monitoring **tool calls, retries, and failures**
- Inspecting **prompt → model → output chains**
- Evaluating RAG quality and retrieval relevance
- Analyzing latency and execution flow in LangGraph pipelines

---

## 🛠️ Projects

---

## 🧪 Project 1 — **Volvox**

### Intelligent Research Assistant & Knowledge Vault

**Volvox** is a production-grade, RAG-powered AI research platform with ChatGPT-like chat sessions and persistent memory.

### 🔗 Live Demo
👉 **Live Website**: https://volvox-alpha-frontend-suit.vercel.app

👉 **Github Repo**: https://github.com/AmirHashmi017/Volvox-Backend

### 🚀 Features

#### 🤖 RAG-Based Intelligent Chatbot
- LangChain + FAISS Vector Store
- Context-aware conversations
- Persistent chat sessions
- View, continue & delete chat history
- Full conversation memory per session
- Attach documents for grounded responses
- Powered by **Gemini 2.5 Flash**

#### 📄 Research & Document Management
- Upload any research document
- Edit & delete documents
- Large file storage via **MongoDB GridFS**
- Attach documents to chatbot for RAG
- Advanced search:
  - Name
  - Date
  - Time filters

#### 📝 Research Summarization
- Single or multi-document summarization
- Optimized for large research material
- LLM-powered summarization pipelines

#### ▶️ YouTube Video Summarization
- Input YouTube URL
- Fetch transcripts using YouTube Transcript API
- AI-generated summaries
- Attach video knowledge to chatbot (RAG)

#### 🌐 Progressive Web App (PWA)
- Installable on mobile, laptop & desktop
- Near-native experience

### 🛠️ Tech Stack

**Backend**
- Python
- FastAPI
- LangChain
- FAISS
- Uvicorn

**Frontend**
- Next.js
- PWA

**Database & Storage**
- MongoDB
- GridFS

**Deployment**
- HuggingFace
- Vercel

**Testing**
- Postman

---

## 🎥 Project 2 — YouTube Video Q&A Chatbot
- Streamlit-based interface
- Transcript ingestion
- RAG-powered question answering
- Context-aware responses

---

## 🧠 Project 3 — MCP & Agentic Workflows Platform

### 🔗 Live Demo
👉 **Live Website**: https://research-mcp-frontend-suit-alpha.vercel.app

👉 **Github Repo**: https://github.com/AmirHashmi017/MCP-Server-And-LangGraph-Agent

### 📌 Overview
A **multi-system, agent-based AI platform** designed for:
- Research discovery
- Business intelligence
- Market analysis
- Automated proposal generation

Built using **LangGraph + MCP Server**, executed under a **Scrum of Scrums (SoS)** architecture.

---

### 🧩 Integrated AI Systems

- **Volvox** — RAG-based research assistant & knowledge vault
- **Smart Research Answering System** — Web + deep research
- **Innoscope** — Roadmaps, feasibility & market analysis
- **Kickstart** — Automated business proposal generation

Each system acts as an independent AI service unified via MCP.

---

### 🔗 MCP Server (Central Backbone)

Responsibilities:
- Unified tool exposure
- Standardized I/O schemas
- Context passing between agents
- Cross-system orchestration
- Contract-first development
- Decoupling agents from services

---

### 🤖 Agentic Workflows (LangGraph)

#### Workflow 1 — Research → Proposal Pipeline
- Upload research paper
- Summarization
- Roadmap generation
- Market & feasibility analysis
- Auto proposal generation
- Storage in Volvox Knowledge Vault

#### Workflow 2 — Research Intelligence Loop
- Query correction
- Deep research
- Internal paper retrieval
- Summarization
- PDF export
- Knowledge storage

#### Workflow 3 — Competitor & Market Intelligence
- Idea input
- Competitor & patent search
- Query expansion
- Web + RAG search
- Market trend analysis
- Feasibility matrix
- Proposal generation

---

## 🧪 Project 4 — Advanced Streamlit Chatbot

Features:
- RAG-powered chatbot
- Human-in-the-Loop (HITL)
- Short-term memory persistence (PostgreSQL)
- Long-term memory
- Streaming responses
- Weather tool
- Currency conversion tool
- Calculator tool
- Expense management (via MCP)
- Third-party tool integration

---

## 🧠 Key Takeaways

- Designed **autonomous agentic systems**
- Built **graph-based reasoning workflows**
- Implemented **persistent memory architectures**
- Orchestrated **multi-agent systems using MCP**
- Applied **RAG at scale**
- Focused on **real-world, production-ready AI**

---

## 📌 Final Note

This repository reflects my transition from **LLM user → AI system architect**, focusing on **Agentic AI**, **scalable workflows**, and **intelligent automation**.

⭐ If this repo helps or inspires you, feel free to star it and connect!
