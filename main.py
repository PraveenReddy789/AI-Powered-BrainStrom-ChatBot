import os
from langchain_huggingface import HuggingFaceEndpoint
import streamlit as st

# Streamlit app configuration
st.set_page_config(page_title="Mistral AI Brainstorm", layout="centered")

# Custom CSS for styling
st.markdown("""
    <style>
    .chat-container {
        background-color:rgb(241, 242, 245);
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .user-message {
        background-color: #ffedd5;
        padding: 10px;
        border-radius: 15px;
        text-align: left;
        margin: 5px 0;
        font-size: 16px;
        line-height: 1.5;
    }
    .ai-message {
        background-color: #d9f9d9;
        padding: 10px;
        border-radius: 15px;
        text-align: left;
        margin: 5px 0;
        font-size: 16px;
        line-height: 1.5;
    }
    .chat-input {
        margin-top: 20px;
    }
    .ask-button {
        background-color: #4caf50;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        font-size: 16px;
        cursor: pointer;
    }
    .ask-button:hover {
        background-color: #45a049;
    }
    .brainstorm-section {
        background-color: #fce4ec;
        border-radius: 15px;
        padding: 15px;
        margin-top: 20px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
    }
    .emoji {
        font-size: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.title("🧠 Mistral AI Brainstorm")
st.markdown("Generate ideas, solve problems, and explore creative solutions with **Mistral-7B-Instruct**!")

# Hugging Face token input
if "HF_TOKEN" not in st.session_state:
    st.session_state["HF_TOKEN"] = "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"

if "history" not in st.session_state:
    st.session_state["history"] = []  # To store question-answer pairs

if st.session_state["HF_TOKEN"]:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.session_state["HF_TOKEN"]

    # Model repository ID
    repo_id = "mistralai/Mistral-7B-Instruct-v0.3"

    # Initialize the model
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        temperature=0.7,
        max_length=128,
        token=st.session_state["HF_TOKEN"],
    )

    # Chat history container
    st.write("### Chat History 💬")
    chat_container = st.container()
    with chat_container:
        for qa in st.session_state["history"]:
            st.markdown(
                f"<div class='chat-container'>"
                f"<div class='user-message'><span class='emoji'>🙋‍♂️</span> <strong>You:</strong> {qa['question']}</div>"
                f"<div class='ai-message'><span class='emoji'>🤖</span> <strong>Mistral:</strong> {qa['answer']}</div>"
                f"</div>", unsafe_allow_html=True
            )

    # Brainstorming section
    st.write("### Brainstorming Section 🌀")
    with st.expander("Start Brainstorming with AI 💡"):
        st.markdown("""
        **How it works:**
        - Share your topic or problem.
        - Get creative ideas or solutions from Mistral AI.
        """)
        brainstorm_topic = st.text_input("Enter a topic or problem to brainstorm:")
        brainstorm_button = st.button("Brainstorm 🚀")

        if brainstorm_button and brainstorm_topic.strip():
            with st.spinner("Generating ideas..."):
                try:
                    # Generate ideas
                    brainstorm_response = llm.invoke(f"Brainstorm ideas on: {brainstorm_topic.strip()}")
                    st.markdown(f"""
                    <div class='brainstorm-section'>
                        <span class='emoji'>✨</span> **Brainstorming Ideas:**
                        <div>{brainstorm_response}</div>
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error: {e}")

    # Input form for regular chat
    st.write("### Ask a Question 🤔")
    with st.form("chat_form", clear_on_submit=True):
        question = st.text_input("Enter your question:", placeholder="Type your question here...")
        submit_button = st.form_submit_button("Ask")

        if submit_button and question.strip():  # Only proceed if the button is clicked and question is valid
            with st.spinner("Thinking..."):
                try:
                    # Generate the response
                    answer = llm.invoke(question.strip())
                    # Append the question and answer to the history
                    st.session_state["history"].append({"question": question.strip(), "answer": answer})
                except Exception as e:
                    st.error(f"Error: {e}")

else:
    st.warning("Please enter your Hugging Face API Token to proceed.")
