
import streamlit as st
from rag_backend import chat_with_hr

# Page configuration
st.set_page_config(
    page_title="Zyro Dynamics HR Assistant",
    page_icon="🤖",
    layout="centered"
)

# Application heading
st.title("🤖 Zyro Dynamics HR Assistant")
st.caption("Your AI-powered assistant for HR policies and employee information.")

st.divider()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Clear chat button
if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()

st.sidebar.title("About")
st.sidebar.write(
    "Ask questions about company HR policies, "
    "leave, work from home, benefits, conduct and more."
)

st.sidebar.info(
    "Answers are generated using the available Zyro Dynamics HR policy documents."
)

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept employee question
if question := st.chat_input("Ask an HR policy question..."):

    # Display user question
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    # Generate chatbot response
    with st.chat_message("assistant"):
        with st.spinner("Searching HR policies..."):
            try:
                answer = chat_with_hr(question)
                st.markdown(answer)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

            except Exception as error:
                st.error("Something went wrong while generating the answer.")
                st.exception(error)
