import streamlit as st
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.agents import initialize_agent, Tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_groq import ChatGroq


# Load environment
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

os.environ['GROQ_API_KEY']=os.getenv("GROQ_API_KEY")

groq_api_key=os.getenv("GROQ_API_KEY")


# Init Streamlit
st.set_page_config(page_title="Search Engine with LangChain Agents")
st.title("🔍 LangChain-Powered Search Engine (Docs + Web)")
st.write("Ask questions, and the agent will choose between local documents or web search to answer.")

# Upload documents
if "vectorstore" not in st.session_state:
    with st.spinner("Loading documents and building vector index..."):
        loader = PyPDFDirectoryLoader("docs")
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        st.session_state.vectorstore = FAISS.from_documents(split_docs, embeddings)

# Setup tools
retriever_tool = Tool(
    name="DocumentSearch",
    func=st.session_state.vectorstore.as_retriever().get_relevant_documents,
    description="Use this to search through uploaded research papers."
)

web_search_tool = Tool(
    name="WebSearch",
    func=DuckDuckGoSearchRun().run,
    description="Use this to search the web for recent or general info."
)

tools = [retriever_tool, web_search_tool]

# Choose model (OpenAI, Groq, Ollama)
# llm = ChatOpenAI(temperature=0)  # Replace with ChatGroq(...) or ChatOllama(...) if desired

llm=ChatGroq(groq_api_key=groq_api_key,model_name="Llama3-8b-8192")

agent_executor = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True
)

# Query input
query = st.text_input("Ask a question (e.g., 'What does the research say about Llama 3?')")

if query:
    with st.spinner("Thinking..."):
        answer = agent_executor.run(query)
        st.success("Answer:")
        st.write(answer)