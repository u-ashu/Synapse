import os
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# =========================
# PAGE CONFIG & THEME
# =========================
st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="wide")

NAVBAR_CSS = """
<style>
.navbar {
    background: linear-gradient(90deg, #4e54c8, #8f94fb);
    padding: 1rem;
    text-align: center;
    border-radius: 12px;
    margin-bottom: 20px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.08);
}
.navbar h1 {
    color: #fff;
    font-size: 1.75rem;
    margin: 0;
    letter-spacing: .3px;
}
.input-row {
    display: flex;
    align-items: center;
    gap: 8px;
    background: #f0f2f6;
    border-radius: 12px;
    padding: 8px 10px;
}
.mic-btn, .send-btn {
    border: none;
    background: transparent;
    cursor: pointer;
    font-size: 20px;
}
.helper {
    font-size: 0.9rem;
    color: #666;
}
</style>
"""
st.markdown(NAVBAR_CSS, unsafe_allow_html=True)
st.markdown('<div class="navbar"><h1>🤖 AI Chatbot</h1></div>', unsafe_allow_html=True)

# =========================
# MODEL SETUP
# =========================
load_dotenv()  # optional if you use Streamlit Secrets instead
# If you need a token: token=os.getenv("HUGGINGFACE_API_TOKEN")
llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",  # change as you like
    task="text-generation",
    max_new_tokens=200,
)
chat_model = ChatHuggingFace(llm=llm)

# =========================
# SESSION STATE
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": "user"/"assistant", "content": "..."}
if "text" not in st.session_state:
    st.session_state.text = ""      # bound to the text_input widget
if "clear_text_next" not in st.session_state:
    st.session_state.clear_text_next = False  # flag to clear input on next rerun
if "user_profile" not in st.session_state:
    st.session_state.user_profile = {"name": "Guest", "email": "Not provided"}

# IMPORTANT: Clear the input BEFORE the widget is created (if requested)
if st.session_state.clear_text_next:
    st.session_state.text = ""
    st.session_state.clear_text_next = False

# =========================
# SIDEBAR (Profile + History)
# =========================
with st.sidebar:
    st.header("👤 Profile")
    st.session_state.user_profile["name"] = st.text_input("Your Name", st.session_state.user_profile["name"])
    st.session_state.user_profile["email"] = st.text_input("Email (optional)", st.session_state.user_profile["email"])
    st.markdown("---")
    st.subheader("💬 Chat History (this session)")
    if st.session_state.messages:
        for i, m in enumerate(st.session_state.messages, start=1):
            st.write(f"**{m['role'].capitalize()} {i}:** {m['content']}")
    else:
        st.info("No chats yet.")
    st.markdown("---")
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

# =========================
# DISPLAY CHAT
# =========================
for msg in st.session_state.messages:
    with st.chat_message("user" if msg["role"] == "user" else "assistant"):
        bubble_class = "chat-bubble-user" if msg["role"] == "user" else "chat-bubble-bot"
        st.markdown(f"<div class='{bubble_class}'>{msg['content']}</div>", unsafe_allow_html=True)

st.caption("You can type your message or use the mic. Press **Enter** or click **Send**.")

# =========================
# INPUT AREA (FORM prevents double runs & enables Enter submit)
# =========================
with st.form("chat_form", clear_on_submit=False):
    # The Streamlit widget we read in Python:
    text = st.text_input("Type your message", value=st.session_state.text, key="text", label_visibility="collapsed")

    # Mic + Send row (mic is HTML/JS that fills the text input; Send is the actual submit)
    c1, c2 = st.columns([1, 5])
    with c1:
        # This renders a mic button that uses Web Speech API to fill the Streamlit text input.
        # It does NOT submit; it only writes into the existing input.
        st.components.v1.html(
            """
            <div class="input-row">
              <button class="mic-btn" type="button" title="Speak" onclick="startDictation()">🎤</button>
              <span class="helper">Speak → text will appear in the box above</span>
            </div>
            <script>
            function startDictation() {
              try {
                if (window.hasOwnProperty('webkitSpeechRecognition')) {
                  var recognition = new webkitSpeechRecognition();
                  recognition.continuous = false;
                  recognition.interimResults = false;
                  recognition.lang = "en-US";
                  recognition.start();
                  recognition.onresult = function(e) {
                    const txt = e.results[0][0].transcript;
                    // Find the first text input (Streamlit uses baseweb input)
                    const input = window.parent.document.querySelector('input[type="text"]');
                    if (input) {
                      input.value = txt;
                      input.dispatchEvent(new Event('input', { bubbles: true }));
                    }
                    recognition.stop();
                  };
                  recognition.onerror = function(e){ recognition.stop(); };
                } else {
                  alert("Speech recognition not supported in this browser.");
                }
              } catch (err) {
                console.error(err);
              }
            }
            </script>
            """,
            height=60,
        )
    with c2:
        submitted = st.form_submit_button("📨 Send", use_container_width=True)

# =========================
# PROCESS MESSAGE (single-shot, no loops)
# =========================
if submitted:
    # Only handle if there's actual text (after trim)
    user_query = (st.session_state.text or "").strip()
    if user_query:
        # Append user message
        st.session_state.messages.append({"role": "user", "content": user_query})

        # Get model reply
        try:
            reply = chat_model.invoke(user_query)
            bot_text = getattr(reply, "content", str(reply))
        except Exception as e:
            bot_text = f"(Error calling model: {e})"

        st.session_state.messages.append({"role": "assistant", "content": bot_text})

        # Clear the input on the NEXT rerun (safe place to modify widget state)
        st.session_state.clear_text_next = True

        # Rerun to refresh UI & clear the box
        st.rerun()

