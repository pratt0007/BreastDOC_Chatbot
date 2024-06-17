import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import logging
import requests
from bs4 import BeautifulSoup

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Configure the generative AI with the API key
genai.configure(api_key=api_key)

# Initialize logging
logging.basicConfig(level=logging.INFO)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
            else:
                logging.warning(f"Could not extract text from page {page}")
    return text

def get_web_text(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text()
        return text
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching the URL {url}: {e}")
        return ""

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in the provided context, just say, "answer is not available in the context", don't provide the wrong answer.\n\n
    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

def main():
    # st.markdown(
    #     """
    #     <style>
    #     .stApp {
    #         background-color: white;
    #     }
    #     .stTextInput>div>div>input {
    #         background-color: lightpink;
    #     }
    #     </style>
    #     """,
    #     unsafe_allow_html=True
    # )    
    st.set_page_config(page_title="BreastDOC", layout="wide")
    st.header("Chat with our BreastDOC chatbot!!")

    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
    if 'input_key' not in st.session_state:
        st.session_state.input_key = 0

    def handle_input():
        user_question = st.session_state[f"user_input_{st.session_state.input_key}"]
        response = user_input(user_question)
        st.session_state.conversation.append((user_question, response))
        st.session_state.input_key += 1

    if st.session_state.conversation:
        for question, response in st.session_state.conversation:
            st.write(f"**You:** {question}")
            st.write(f"**ChatBot:** {response}")

    st.text_input("Ask a Question", key=f"user_input_{st.session_state.input_key}", on_change=handle_input)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True, type=['pdf'])
        url_input = st.text_area("Enter URLs to scrape (one per line)")

        if st.button("Submit & Process PDFs"):
            if pdf_docs:
                with st.spinner("Processing PDFs..."):
                    try:
                        raw_text = get_pdf_text(pdf_docs)
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("PDFs processed successfully")
                    except Exception as e:
                        st.error(f"Error: {e}")
                        logging.error(f"Error during PDF processing: {e}")
            else:
                st.error("Please upload at least one PDF file.")

        if st.button("Scrape Websites"):
            if url_input:
                urls = [url.strip() for url in url_input.split('\n') if url.strip()]
                if urls:
                    all_text = ""
                    with st.spinner(f"Scraping {len(urls)} websites..."):
                        for url in urls:
                            try:
                                raw_text = get_web_text(url)
                                all_text += raw_text + "\n"
                            except Exception as e:
                                st.error(f"Error scraping {url}: {e}")
                                logging.error(f"Error scraping {url}: {e}")

                    if all_text:
                        text_chunks = get_text_chunks(all_text)
                        get_vector_store(text_chunks)
                        st.success(f"Websites scraped and processed successfully")
                else:
                    st.error("Please enter at least one valid URL.")
            else:
                st.error("Please enter URLs to scrape.")

if __name__ == "__main__":
    main()
