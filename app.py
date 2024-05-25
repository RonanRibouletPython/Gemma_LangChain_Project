
import os
from dotenv import load_dotenv

import streamlit as st

from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
# Vector Store DB
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

## load the GROQ And OpenAI API KEY 
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Title of the StreamLite app
st.title("Langchain Q&A with Gemma LLM")

Gemma_llm = ChatGroq(groq_api_key=groq_api_key,
                     model_name="gemma-7b-it",
                     temperature=0,
                     max_tokens=1000,
                     verbose=True
            )

# Check the Gemma llm parameters
#print(Gemma_llm)

# Create the prompt with zero shot
prompt = ChatPromptTemplate.from_template(
    """
    <start_of_turn>user
    You are an assistant capable of answering any questions from users.
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    Do not be harmful in any way to the user.
    Here is the context: {context}
    Query of the user: {input}
    <end_of_turn>
    <start_of_turn>model
    """
)

# With this function we will read all the PDF files
def vector_embedding():

    if "vectors" not in st.session_state:
        # Load the embeddings
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            task_type="retrieval_query"
        )

        # Data ingestion
        st.session_state.loader = PyPDFDirectoryLoader(path="./PDF",
                                                       recursive=False
                                            )
        
        # Document loading
        st.session_state.documents = st.session_state.loader.load()

        # Text splitter
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=64
        )

        # Aplly the split to the documents
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.documents[:20])
        #print(f"Size: {len(st.session_state.documents)}")

        # Vector DB
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        
prompt_input = st.text_input("What do you want to ask to the document?")

if st.button("Vector Embedding"):

    vector_embedding()
    
    st.write("Vector DB is ready")

if prompt_input:
    document_chain = create_stuff_documents_chain(Gemma_llm, prompt)
    # Use the interface to retrieve the informations from the vectors
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever=retriever,
                                             combine_docs_chain=document_chain
                                    )
    


    response = retrieval_chain.invoke({"input":prompt_input})
    # Display the answer
    st.write(response["answer"])

    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant context that made the model answers that question correctly
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")


