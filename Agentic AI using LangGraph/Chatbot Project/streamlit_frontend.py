import streamlit as st
from langgraph_backend import chatbot
from langgraph_backend import retrieve_all_threads
from langgraph_backend import save_chat_title, get_chat_title, get_all_chat_titles
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import uuid

st.set_page_config(page_title="LangGraph Chatbot", page_icon="🤖", layout="wide")

def generate_thread_id():
    return str(uuid.uuid4())

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    st.session_state['message_history'] = []

def add_thread(thread_id, chat_title):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)
        save_chat_title(thread_id, chat_title)

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
    st.session_state['chat_threads'] = list(retrieve_all_threads())

st.sidebar.title("🤖 LangGraph Chatbot")

if st.sidebar.button("➕ New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

st.sidebar.divider()
st.sidebar.header("💬 My Conversations")

for thread_id in st.session_state['chat_threads']:
    chat_title = get_chat_title(str(thread_id))
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
            messages = load_conversation(thread_id)

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
            st.rerun()


st.title("💬 Chat Assistant")

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

user_input = st.chat_input("Type your message here...")

if user_input:
    if st.session_state['thread_id'] not in st.session_state['chat_threads']:
        chat_title = generate_chat_title(user_input)
        add_thread(st.session_state['thread_id'], chat_title)

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
        
        for event in chatbot.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
            stream_mode="values",
        ):
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
                            new_content = str(new_content)
                        elif not isinstance(new_content, str):
                            new_content = str(new_content)
                        
                        if new_content != full_response:
                            full_response = new_content
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