import streamlit as st
from langgraph_backend import chatbot
from langchain_core.messages import HumanMessage, AIMessage
import uuid

def generate_thread_id():
    thread_id= uuid.uuid4()
    return thread_id

def reset_chat():
    thread_id= generate_thread_id()
    st.session_state['thread_id']=thread_id
    st.session_state['message_history']=[]

def add_thread(thread_id,chat_title):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)
        st.session_state['chat_titles'][thread_id]=chat_title
         

def load_conversation(thread_id):
    state = chatbot.get_state(
        config={"configurable": {'thread_id': thread_id}}
    )
    return state.values.get("messages", [])

def generate_chat_title(question):
    return question[:25] if len(question)>25 else question

is_streaming=False
if 'message_history' not in st.session_state:
    st.session_state['message_history']=[]

if 'thread_id' not in st.session_state:
    st.session_state['thread_id']=generate_thread_id()

if 'chat_titles' not in st.session_state:
    st.session_state['chat_titles']={}

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads']=[]

    

config= {"configurable":{"thread_id":st.session_state['thread_id']}}

user_input= st.chat_input("Type Here")

if user_input and st.session_state['thread_id'] not in st.session_state['chat_threads']:
    chat_title=generate_chat_title(user_input)
    add_thread(st.session_state['thread_id'],chat_title)
    

st.sidebar.title("LangGraph Chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()
    st.rerun()

st.sidebar.header("My Conversations")

for thread_id in st.session_state['chat_threads']:
    chat_title = st.session_state['chat_titles'][thread_id]
    if st.sidebar.button(chat_title, key=str(thread_id)):
        st.session_state['thread_id'] = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []
        for message in messages:
            if isinstance(message, HumanMessage):
                role = 'user'
            else:
                role = 'assistant'
            temp_messages.append({'role': role, 'content': message.content})

        st.session_state['message_history'] = temp_messages
        st.rerun()

if(not is_streaming):
    for message in st.session_state['message_history']:
        with st.chat_message(message["role"]):
            st.text(message["content"])

if user_input:
    is_streaming=True
    st.session_state['message_history'].append({'role':'user','content':user_input})
    with st.chat_message('user'):
        st.text(user_input)   
    # with st.chat_message('assistant'):
    #     response= chatbot.invoke(
    #         {'messages':[HumanMessage(content=user_input)]},
    #         config=config
    #     )
    #     ai_message = ""
    #     if 'messages' in response:
    #         ai_messages = [msg.content for msg in response['messages'] if isinstance(msg, AIMessage)]
    #         if ai_messages:
    #             ai_message = ai_messages[-1]
    #     with st.chat_message('assistant'):
    #         st.text(ai_message)
        # ai_message= st.write_stream(
        #     message_chunk.content for message_chunk,metadata in chatbot.stream(
        #         {'messages':[HumanMessage(content=user_input)]},
        #         config=config,
        #         stream_mode= "messages"
        #     )
        # )
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        last_len = 0
    
        for chunk, _ in chatbot.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
            stream_mode="messages",
        ):
            if isinstance(chunk, AIMessage):
                if len(chunk.content) <= last_len:
                    continue
                
                full_response = chunk.content
                last_len = len(full_response)
                placeholder.markdown(full_response)

    st.session_state['message_history'].append({'role':'assistant','content':full_response})
    is_streaming=False

    