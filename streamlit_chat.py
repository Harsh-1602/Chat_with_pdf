import os
import re
import time
import base64
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.schema.messages import AIMessage, HumanMessage
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import ServiceContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.schema import IndexNode
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import RecursiveRetriever, VectorIndexRetriever
from llama_index.core.storage.storage_context import StorageContext
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader

# Load environment variables
load_dotenv()
Groq_api_key = os.getenv("GROQ_API_KEY")

# Function to get models
@st.cache_resource(show_spinner=False)
def get_models(model_name):
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5")
    
    system_prompt = """
    You are a Q&A assistant. Your goal is to answer questions as accurately as possible based on the instructions and context provided.
    If the answer is not present in the context, warn the user and then generate the answer based on the query.
    """
    Settings._prompt_helper = system_prompt

    llm = Groq(model=model_name, api_key=Groq_api_key, temperature=0.5, system_prompt=system_prompt)
    return llm, embed_model

# Function to generate nodes from documents
def gen_nodes(doc):
    parser = LlamaParse(api_key=os.getenv("LLAMA-PARSE"), result_type="markdown", verbose=True)
    file_extractor = {".pdf": parser}
    documents = SimpleDirectoryReader(input_files=[f"{doc}"], file_extractor=file_extractor).load_data()

    node_parser = SentenceSplitter(chunk_size=1024)
    base_nodes = node_parser.get_nodes_from_documents(documents)

    sub_chunk_sizes = [256, 512]
    sub_node_parsers = [SentenceSplitter(chunk_size=c) for c in sub_chunk_sizes]
    
    all_nodes = []
    for base_node in base_nodes:
        for n in sub_node_parsers:
            sub_nodes = n.get_nodes_from_documents([base_node])
            sub_inodes = [IndexNode.from_text_node(sn, base_node.node_id) for sn in sub_nodes]
            all_nodes.extend(sub_inodes)

        original_node = IndexNode.from_text_node(base_node, base_node.node_id)
        all_nodes.append(original_node)

    return all_nodes

# Function to create a query engine
@st.cache_data(show_spinner=False)
def get_query_engine(path):
    service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)
    all_nodes = gen_nodes(path)
    
    storage_context = StorageContext.from_defaults()
    vector_ind = VectorStoreIndex(all_nodes, service_context=service_context, storage_context=storage_context)
    
    vector_r = vector_ind.as_retriever(similarity_top_k=2)
    query_engine_chunk = RetrieverQueryEngine.from_args(vector_r, service_context=service_context)
    
    return query_engine_chunk

# Function to get a response based on user query
def get_response(prompt):
    response = st.session_state.query_engine.query(prompt)
    return response

# Function to save the uploaded file to a temporary directory
def save_uploadedfile(uploadedfile):
    with open(os.path.join("tempDir", uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())

# Main Streamlit app
st.title("Chat with PDFs")

# Sidebar for settings
left_column, right_column = st.columns(2)
with st.sidebar:
    st.header("Settings")
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    
    options = ["Mixtral 8x7b", "LLaMA3 70b", "LLaMA3 8b"]
    selected_option = st.selectbox("Choose a Model:", options)
    if selected_option == "LLaMA3 8b":
        option = "llama3-8b-8192"
    elif selected_option == "Mixtral 8x7b":
        option = "mixtral-8x7b-32768"
    elif selected_option == "LLaMA3 70b":
        option = "llama3-70b-8192"

    st.write(f"You selected: {selected_option}")

# Load the model
llm, embed_model = get_models(option)

if pdf is None:
    for key in st.session_state.keys():
        del st.session_state[key]
    st.info("Please upload a PDF")
else:
    folder_name = "tempDir"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    save_uploadedfile(pdf)
    Settings.llm = llm
    Settings.embed_model = embed_model
    
    path = f"tempDir/{pdf.name}"

    # Embed and display PDF in Streamlit
    with open(path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = F'<embed id="pdfViewer" src="data:application/pdf;base64,{base64_pdf}" width="500" height="1000" type="application/pdf">'
    
    if "query_engine" not in st.session_state:
        with st.spinner("Analyzing Data!"):
            st.session_state.query_engine = get_query_engine(path)
    
    user_query = st.chat_input("Type your message here...")
    
    with right_column:
        st.markdown(pdf_display, unsafe_allow_html=True)
    
    with left_column:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if user_query:
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)
            with st.spinner("Retrieving Answer"):
                response = get_response(user_query)
                # page = 1  # For now, the page is set to 1; modify as needed
                # st.markdown(f"Answer is from Page no: {page}")
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

            # Clear tempDir after response
            for f in os.listdir("tempDir"):
                os.remove(os.path.join("tempDir", f))
