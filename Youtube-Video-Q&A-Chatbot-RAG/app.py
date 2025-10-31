import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from dotenv import load_dotenv
import re

load_dotenv()

st.set_page_config(
    page_title="YouTube Video Q&A",
    page_icon="🎥",
    layout="wide"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
        color: #FF0000;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .answer-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #FF0000;
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

def extract_video_id(url):
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/|youtube\.com\/v\/)([a-zA-Z0-9_-]{11})',
        r'youtube\.com\/watch\?.*v=([a-zA-Z0-9_-]{11})'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    if re.match(r'^[a-zA-Z0-9_-]{11}$', url):
        return url
    
    raise ValueError("Invalid YouTube URL or Video ID")

def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

@st.cache_resource
def process_video(video_url):
    try:
        video_id = extract_video_id(video_url)

        transcript_list = YouTubeTranscriptApi().fetch(video_id)
        transcript = " ".join([item.text for item in transcript_list.snippets])

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        chunks = splitter.create_documents([transcript])

        embedding = GoogleGenerativeAIEmbeddings(model="text-embedding-004")
        vector_store = FAISS.from_documents(
            documents=chunks,
            embedding=embedding
        )
        
        retriever = vector_store.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": 4}
        )
        
        return retriever, video_id, None
    
    except Exception as e:
        return None, None, str(e)

def get_answer(retriever, question):
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
        
        prompt = PromptTemplate(
            template='''You are a helpful assistant.
            Answer only from the provided transcript context.
            If the context is insufficient, just say it is not in this video.

            {context}

            Question: {question}

            Answer:''',
            input_variables=['context', 'question']
        )
        
        parallel_chain = RunnableParallel({
            'context': retriever | RunnableLambda(format_docs),
            'question': RunnablePassthrough()
        })
        
        parser = StrOutputParser()
        llm_chain = prompt | llm | parser
        final_chain = parallel_chain | llm_chain
        
        response = final_chain.invoke(question)
        return response, None
    
    except Exception as e:
        return None, str(e)


st.markdown('<div class="main-header"> YouTube Video Q&A Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Ask questions about any YouTube video transcript</div>', unsafe_allow_html=True)

if 'current_video_id' not in st.session_state:
    st.session_state.current_video_id = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Layout
col1, col2 = st.columns([2, 1])

with col1:
    video_url = st.text_input(
        "Enter YouTube Video URL",
        placeholder="https://www.youtube.com/watch?v=...",
        help="Paste any YouTube video URL here"
    )

with col2:
    st.write("")
    st.write("")
    if st.button("🔄 Clear Cache", use_container_width=True):
        st.cache_resource.clear()
        st.session_state.current_video_id = None
        st.session_state.chat_history = []
        st.success("Cache cleared!")
        st.rerun()

if video_url:
    try:
        video_id = extract_video_id(video_url)

        if st.session_state.current_video_id != video_id:
            st.session_state.current_video_id = video_id
            st.session_state.chat_history = []
        
        st.video(video_url)

        with st.spinner("🔄 Processing video transcript and creating embeddings..."):
            retriever, processed_id, error = process_video(video_url)
        
        if error:
            st.error(f" Error processing video: {error}")
        elif retriever:
            st.success(f" Video processed successfully! (ID: {processed_id})")

            st.markdown("---")
            st.subheader("💬 Ask a Question")
            
            question = st.text_input(
                "Your question",
                placeholder="What are the major topics discussed in the video?",
                key="question_input"
            )
            
            col_ask, col_clear = st.columns([4, 1])
            
            with col_ask:
                ask_button = st.button("🚀 Ask Question", use_container_width=True, type="primary")
            
            with col_clear:
                if st.button("🗑️ Clear Chat", use_container_width=True):
                    st.session_state.chat_history = []
                    st.rerun()
            
            if ask_button and question:
                with st.spinner("🤔 Thinking..."):
                    answer, error = get_answer(retriever, question)
                
                if error:
                    st.error(f"❌ Error getting answer: {error}")
                elif answer:
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "question": question,
                        "answer": answer
                    })

            if st.session_state.chat_history:
                st.markdown("---")
                st.subheader("📝 Chat History")
                
                for i, chat in enumerate(reversed(st.session_state.chat_history)):
                    with st.container():
                        st.markdown(f"**Q{len(st.session_state.chat_history) - i}:** {chat['question']}")
                        st.markdown(f'<div class="answer-box"><strong>Answer:</strong><br>{chat["answer"]}</div>', unsafe_allow_html=True)
                        st.markdown("")
    
    except ValueError as e:
        st.error(f" {str(e)}")

else:
    st.info(" Enter a YouTube video URL to get started!")
    
    with st.expander(" How to use"):
        st.markdown("""
        1. **Paste a YouTube URL** in the input field above
        2. **Wait** for the video to be processed (this happens only once per video)
        3. **Ask questions** about the video content
        4. **View answers** based on the video transcript
        
        **Note:** The video transcript is cached, so you can ask multiple questions without reprocessing!
        """)
    
    with st.expander(" Example URLs"):
        st.code("https://www.youtube.com/watch?v=KwzytY32xlk")
        st.code("https://youtu.be/KwzytY32xlk")

st.markdown("---")
