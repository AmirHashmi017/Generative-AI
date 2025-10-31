# YouTube Video Q&A Assistant 

A GenAI-powered tool to ask questions about any YouTube video transcript. Built using **LangChain**, **Google Generative AI** and **Streamlit**, , this app converts video transcripts into searchable embeddings and answers your questions in natural language.

## 🔹 Features

- Automatically fetches YouTube video transcripts using `youtube-transcript-api`.
- Splits transcript into chunks with `RecursiveCharacterTextSplitter` for efficient processing.
- Generates embeddings using **GoogleGenerativeAIEmbeddings**.
- Stores and retrieves data with **FAISS vector store**.
- Provides answers using **ChatGoogleGenerativeAI** in context of the video transcript.
- Maintains **chat history** for multiple queries.
- Built with **Streamlit** for interactive and responsive UI.

## 🔹 Tech Stack

- **Gen-AI & NLP:** LangChain, Google Chat Generative AI, Google Generative AI Embedding 
- **Vector Database:** FAISS  
- **Frontend & Deployement:** Streamlit   

## 🔹 Installation

```bash
git clone https://github.com/AmirHashmi017/YouTube-Video-Q-A-Assistant-RAG
cd Generative-AI/Youtube-Video-Q&A-Chatbot-RAG
python -m venv venv
source venv/bin/activate
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
