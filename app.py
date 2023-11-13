import os
import pickle
import streamlit as st
from PyPDF2 import PdfReader
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain

with st.sidebar:
    st.title('🗨️ PDF Based Chatbot')
    st.markdown("## Conversation History: ")

    if "chat_sessions" not in st.session_state:
        st.session_state.chat_sessions = {}

    if "active_session" not in st.session_state or st.sidebar.button("New Chat"):
        chat_id = len(st.session_state.chat_sessions) + 1
        session_key = f"Chat {chat_id}"
        st.session_state.chat_sessions[session_key] = []
        st.session_state.active_session = session_key

    for session in st.session_state.chat_sessions:
        if st.sidebar.button(session, key=session):
            st.session_state.active_session = session
    st.markdown('''
    ## About App:

    The app's primary resource is utilized to create:

    - [Streamlit](https://streamlit.io/)
    - [Langchain](https://docs.langchain.com/docs/)
    - [OpenAI](https://openai.com/)

    ## About me:

    - [Linkedin](https://www.linkedin.com/in/kunal-pamu-710674230/)

    ''')
    st.write("Made by Kunal Shripati Pamu")

def save_vector_store(vector_store, file_path):
    # Save
    with open(file_path, "wb") as f:
        pickle.dump(vector_store, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_vector_store(file_path):
    # Load
    with open(file_path, "rb") as f:
        vector_store = pickle.load(f)
    return vector_store

def main():
    api_key = st.text_input("Enter your OpenAI API Key", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

        st.header("Chat with your PDF File")

        # PDF upload and processing
        pdf = st.file_uploader("Upload your PDF:", type='pdf')

        # extract the text
        if pdf is not None:
            pdf_reader = PdfReader(pdf)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()

            # split text into chunks
            text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_text(text)
            
            # check cache for pdf name and if present use the previous embeddings else create new ones 
            store_name = pdf.name[:-4]

            if os.path.exists(f"{store_name}.pkl"):
                vector_store = load_vector_store(f"{store_name}.pkl")
            else:
                embeddings=OpenAIEmbeddings()
                vector_store=FAISS.from_texts(chunks, embedding=embeddings)
                save_vector_store(vector_store, f"{store_name}.pkl")
                
            llm = OpenAI(temperature=0)
            qa_chain = ConversationalRetrievalChain.from_llm(llm, vector_store.as_retriever())

            if "active_session" in st.session_state:
                for message in st.session_state.chat_sessions[st.session_state.active_session]:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

            query = st.chat_input("Ask your questions from PDF ")

            if query:
                st.session_state.chat_sessions[st.session_state.active_session].append({"role": "user", "content": query})
                with st.chat_message("user"):
                    st.markdown(query)

                result = qa_chain({"question": query, "chat_history": [(message["role"], message["content"]) for message in st.session_state.chat_sessions[st.session_state.active_session]]})
                response = result["answer"]

                with st.chat_message("assistant"):
                    st.markdown(response)
                st.session_state.chat_sessions[st.session_state.active_session].append({"role": "assistant", "content": response})

if __name__ == '__main__':
    main()
