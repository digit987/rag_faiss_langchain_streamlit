import os
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import streamlit as st

# Load environment variables
os.environ["OPENAI_API_KEY"] == st.secrets["OPENAI_API_KEY"]

def faiss_rag(filename, prompt):
    loader = TextLoader(filename)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embeddings)
    retriever = db.as_retriever()
    template = """You are an assistant for question-answering tasks including the 
    context of news that includes presidents of USA etc. Also, you have to provide 
    answers to questions outside the context. See the context and question and answer.
    Question: {question} 
    Context: {context} 
    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    rag_chain = (
        {"context": retriever,  "question": RunnablePassthrough()} 
        | prompt 
        | llm
        | StrOutputParser() 
    )

    return rag_chain.invoke(prompt)

def main():
    st.title("Chatbot using your Text File. Implementing RAG using FAISS and LangChain")
    text_file = st.file_uploader("Choose a text file", type="txt")
    if text_file is not None:
        file_name = text_file.name
        prompt = st.text_input("Ask anything")
        if prompt:
            response = faiss_rag(file_name, prompt)
            st.write(response)

if __name__ == "__main__":
    main()
