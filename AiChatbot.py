import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Load environment variables (HuggingFace API key etc.)
load_dotenv()

# Initialize HuggingFace LLM
llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
    max_new_tokens=200   # increased for better responses
)

model = ChatHuggingFace(llm=llm)

# --- Streamlit App ---
st.set_page_config(page_title="Synapsepg", page_icon="🤖")
st.title("🤖 Synapse")

# Initialize chat history in Streamlit session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [SystemMessage(content="You are a helpful assistant.")]

# Display previous chat messages
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)

# User input box
user_input = st.chat_input("Type your message...")

if user_input:
    # Add user input to history
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    # Generate response
    result = model.invoke(st.session_state.chat_history)
    st.session_state.chat_history.append(AIMessage(content=result.content))

    # Display response
    st.chat_message("user").write(user_input)
    st.chat_message("assistant").write(result.content)
