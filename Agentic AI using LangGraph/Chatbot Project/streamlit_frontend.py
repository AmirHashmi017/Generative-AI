import streamlit as st
from langgraph_backend import chatbot, initialize_chatbot
from langgraph_backend import retrieve_all_threads, ingest_pdf
from langgraph_backend import save_chat_title, get_chat_title, get_messages_for_thread
from langgraph_backend import set_thread_context
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.types import Command
from langgraph.errors import GraphInterrupt
import uuid
import asyncio
import sys

st.set_page_config(page_title="LangGraph Chatbot", page_icon="🤖", layout="wide")

@st.cache_resource
def get_event_loop():
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("Event loop is closed")
    except RuntimeError:
        if sys.platform == 'win32':
            loop = asyncio.ProactorEventLoop()
        else:
            loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop

loop = get_event_loop()

def run_async(coro):
    """Run async code using the persistent event loop"""
    return loop.run_until_complete(coro)

async def init_app():
    if st.session_state.get('chatbot_initialized', False) is False:
        with st.spinner("🚀 Initializing chatbot with MCP tools..."):
            await initialize_chatbot()
            st.session_state['chatbot_initialized'] = True

if 'chatbot_initialized' not in st.session_state:
    run_async(init_app())

async def collect_stream_events(user_input, config):
    thread_id = config.get('configurable', {}).get('thread_id')
    set_thread_context(thread_id)
    
    events = []
    interrupt_value = None
    
    try:
        async for event in chatbot.astream(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
            stream_mode="values",
        ):
            events.append(event)
    except GraphInterrupt as e:
        interrupt_value = e.value
        return events, interrupt_value
    
    state = await chatbot.aget_state(config)
    interrupt_value = None
    if state.tasks:
        for task in state.tasks:
            if task.interrupts:
                interrupt_value = task.interrupts[0].value
                break
    
    return events, interrupt_value

async def resume_after_interrupt(decision, config):
    thread_id = config.get('configurable', {}).get('thread_id')
    set_thread_context(thread_id)
    
    events = []
    interrupt_value = None

    
    try:
        async for event in chatbot.astream(
            Command(resume=decision),
            config=config,
            stream_mode="values",
        ):
            events.append(event)
    except GraphInterrupt as e:
        interrupt_value = e.value
        return events, interrupt_value

    return events, interrupt_value
  
def generate_thread_id():
    return str(uuid.uuid4())

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    st.session_state['message_history'] = []

def add_thread(thread_id, chat_title):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)
        run_async(save_chat_title(thread_id, chat_title))

def load_conversation(thread_id):
    state = chatbot.get_state(
        config={"configurable": {'thread_id': thread_id}}
    )
    return state.values.get("messages", [])

def generate_chat_title(question):
    return question[:30] + "..." if len(question) > 30 else question

def get_tool_icon(tool_name):
    """Return an icon for each tool"""
    icons = {
        "duckduckgo_search": "🔍",
        "calculator": "🧮",
        "get_stock_price": "📈",
        "get_weather_data": "🌤️",
        "get_conversion_factor": "💱",
        "convert_currency": "💰"
    }
    return icons.get(tool_name, "🔧")

def format_tool_call(tool_name, args):
    """Format tool call for display"""
    icon = get_tool_icon(tool_name)
    formatted_args = ", ".join([f"{k}={v}" for k, v in args.items()])
    return f"{icon} **{tool_name}**({formatted_args})"

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = list(run_async(retrieve_all_threads()))

if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}

if 'interrupt_data' not in st.session_state:
    st.session_state['interrupt_data'] = None

thread_key = str(st.session_state["thread_id"])
thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})

st.title("💬 Chat Assistant")

user_input = st.chat_input("Type your message here...")

if user_input:
    if st.session_state['thread_id'] not in st.session_state['chat_threads']:
        chat_title = generate_chat_title(user_input)
        add_thread(st.session_state['thread_id'], chat_title)

st.sidebar.title("🤖 LangGraph Chatbot")

if st.sidebar.button("➕ New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

if thread_docs:
    latest_doc = list(thread_docs.values())[-1]
    st.sidebar.success(
        f"Using `{latest_doc.get('filename')}` "
        f"({latest_doc.get('chunks')} chunks from {latest_doc.get('documents')} pages)"
    )
else:
    st.sidebar.info("No PDF indexed yet.")

uploaded_pdf = st.sidebar.file_uploader("Upload a PDF for this chat", type=["pdf"])
if uploaded_pdf:
    if uploaded_pdf.name in thread_docs:
        st.sidebar.info(f"`{uploaded_pdf.name}` already processed for this chat.")
    else:
        with st.sidebar.status("Indexing PDF…", expanded=True) as status_box:
            summary = ingest_pdf(
                uploaded_pdf.getvalue(),
                thread_id=thread_key,
                filename=uploaded_pdf.name,
            )
            thread_docs[uploaded_pdf.name] = summary
            status_box.update(label="✅ PDF indexed", state="complete", expanded=False)

st.sidebar.divider()
st.sidebar.header("💬 My Conversations")

for thread_id in st.session_state['chat_threads']:
    chat_title = run_async(get_chat_title(str(thread_id)))
    if chat_title:
        is_current = (str(thread_id) == str(st.session_state['thread_id']))
        button_type = "primary" if is_current else "secondary"
        
        if st.sidebar.button(
            f"{'📌 ' if is_current else ''}{chat_title}", 
            key=str(thread_id),
            use_container_width=True,
            type=button_type
        ):
            st.session_state['thread_id'] = thread_id
            try:
                with st.spinner("📂 Loading conversation..."):
                    messages = run_async(get_messages_for_thread(thread_id))

                    temp_messages = []
                    for message in messages:
                        if isinstance(message, HumanMessage):
                            temp_messages.append({
                                'role': 'user', 
                                'content': message.content,
                                'type': 'message'
                            })
                        elif isinstance(message, AIMessage):
                            if hasattr(message, 'tool_calls') and message.tool_calls:
                                for tool_call in message.tool_calls:
                                    temp_messages.append({
                                        'role': 'tool_call',
                                        'tool_name': tool_call.get('name', 'unknown'),
                                        'args': tool_call.get('args', {}),
                                        'type': 'tool_call'
                                    })
                            
                            if message.content:
                                temp_messages.append({
                                    'role': 'assistant', 
                                    'content': message.content,
                                    'type': 'message'
                                })
                        elif isinstance(message, ToolMessage):
                            pass

                    st.session_state['message_history'] = temp_messages
            except Exception as e:
                st.error(f"❌ Error loading conversation: {str(e)}")
                
            st.rerun()




for message in st.session_state['message_history']:
    if message.get('type') == 'tool_call':
        with st.status(f"{get_tool_icon(message['tool_name'])} Using {message['tool_name']}", state="complete"):
            st.code(str(message['args']), language="json")
    else:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

config = {
    "configurable": {"thread_id": st.session_state['thread_id']},
    "metadata": {"thread_id": st.session_state['thread_id']},
    "run_name": "chat_turn"
}


if user_input:
    st.session_state['message_history'].append({
        'role': 'user', 
        'content': user_input,
        'type': 'message'
    })
    with st.chat_message('user'):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        status_container = st.container()
        response_placeholder = st.empty()
        
        full_response = ""
        current_tool_calls = []
        active_statuses = {}
        
        with st.spinner("🤖 Thinking..."):
            events, interrupt_value = run_async(collect_stream_events(user_input, config))
            
            if interrupt_value:
                st.session_state['interrupt_data'] = interrupt_value
            else:
                st.session_state['interrupt_data'] = None

if st.session_state.get('interrupt_data'):
    interrupt_msg = st.session_state['interrupt_data']
    interrupt_msg = interrupt_msg if isinstance(interrupt_msg, str) else str(interrupt_msg)

    with st.chat_message("assistant"):
        st.warning(f"⏸️ **Human Review Required**")
        st.write(f"**Request:** {interrupt_msg}")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("✅ Yes - Allow", use_container_width=True, key="hitl_yes"):
                with st.spinner("Resuming with approval..."):
                    resume_events, new_interrupt = run_async(
                        resume_after_interrupt({"approved": "yes"}, config)
                    )

                    for event in resume_events:
                        if "messages" in event:
                            messages_list = event["messages"]

                            for message in messages_list:
                                if isinstance(message, AIMessage):
                                    if message.content:
                                        content = message.content
                                        if isinstance(content, list):
                                            content = content[0].get('text', str(content)) if content else ""

                                        # Add to history and display
                                        st.session_state['message_history'].append({
                                            'role': 'assistant', 
                                            'content': content,
                                            'type': 'message'
                                        })

                    st.session_state['interrupt_data'] = new_interrupt
                    st.rerun()

        with col2:
            if st.button("❌ No - Deny", use_container_width=True, key="hitl_no"):
                with st.spinner("Resuming with denial..."):
                    resume_events, new_interrupt = run_async(
                        resume_after_interrupt({"approved": "no"}, config)
                    )

                    for event in resume_events:
                        if "messages" in event:
                            messages_list = event["messages"]

                            for message in messages_list:
                                if isinstance(message, AIMessage):
                                    if message.content:
                                        content = message.content
                                        if isinstance(content, list):
                                            content = content[0].get('text', str(content)) if content else ""

                                        st.session_state['message_history'].append({
                                            'role': 'assistant', 
                                            'content': content,
                                            'type': 'message'
                                        })

                    st.session_state['interrupt_data'] = new_interrupt
                    st.rerun()
    st.stop()

if user_input:

        for event in events:
            if "messages" in event:
                last_message = event["messages"][-1]

                if isinstance(last_message, AIMessage):
                    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                        for tool_call in last_message.tool_calls:
                            tool_id = tool_call.get('id', str(len(current_tool_calls)))

                            if tool_id not in [tc.get('id') for tc in current_tool_calls]:
                                current_tool_calls.append(tool_call)
                                tool_name = tool_call.get('name', 'unknown')
                                tool_args = tool_call.get('args', {})

                                with status_container:
                                    tool_status = st.status(
                                        f"{get_tool_icon(tool_name)} Using {tool_name}...",
                                        state="running"
                                    )
                                    tool_status.code(str(tool_args), language="json")
                                    active_statuses[tool_id] = tool_status

                                st.session_state['message_history'].append({
                                    'role': 'tool_call',
                                    'tool_name': tool_name,
                                    'args': tool_args,
                                    'type': 'tool_call'
                                })
                    

                    if last_message.content:
                        new_content = last_message.content

                        if isinstance(new_content, list):
                            
                            if new_content and isinstance(new_content[0], dict):
                                new_content = new_content[0].get('text', str(new_content))
                            else:
                                new_content = str(new_content)
                        elif not isinstance(new_content, str):
                            new_content = str(new_content)
                        
                        
                        lines = new_content.split('\n')
                        clean_content = lines[0].strip() if lines else new_content
                        
                        if clean_content and clean_content != full_response:
                            full_response = clean_content
                            response_placeholder.markdown(full_response + " ▌")

                elif isinstance(last_message, ToolMessage):
                    tool_call_id = getattr(last_message, 'tool_call_id', None)
                    if tool_call_id and tool_call_id in active_statuses:
                        active_statuses[tool_call_id].update(state="complete")
 
        for status in active_statuses.values():
            status.update(state="complete")

        if full_response:
            response_placeholder.markdown(full_response)
            st.session_state['message_history'].append({
                'role': 'assistant', 
                'content': full_response,
                'type': 'message'
            })
        else:
            response_placeholder.info("Tool execution completed. Check the tool outputs above.")

st.markdown("""
<style>
    .stButton button {
        border-radius: 8px;
    }
    .stChatMessage {
        border-radius: 12px;
    }
    .element-container:has(> .stStatus) {
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)