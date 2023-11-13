import os
import time
import pickle
import streamlit as st
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import  HuggingFaceHub


HUGGINGFACEHUB_API_TOKEN = ''
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN


st.title("News Research tool")

st.sidebar.title("news articles")

urls=[]
for i in range(3):
    url=st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked=st.sidebar.button("Process URLs")
file_path="faiss_indexstore.pkl"
main_placefolder=st.empty()
if process_url_clicked:
    # load data
    loader=UnstructuredURLLoader(urls=urls)
    main_placefolder.text("Data loading started")
    data=loader.load()
    #split texts
    text_splitter=RecursiveCharacterTextSplitter(
        separators=['\n\n','\n','.',','],
        chunk_size=1000,
        chunk_overlap=200
    )
    main_placefolder.text("Text splitting started")
    docs=text_splitter.split_documents(data)
    #create embeddings
    model_name="sentence-transformers/all-mpnet-base-v2"
    hf=HuggingFaceEmbeddings(model_name=model_name)
    vector_index=FAISS.from_documents(docs,hf)
    main_placefolder.text("Embedding vector started")
    time.sleep(2)

    with open(file_path,"wb") as f:
        pickle.dump(vector_index,f)

query=main_placefolder.text_input("Question:")

if query:
    if os.path.exists(file_path):
        with open(file_path,'rb') as f:
            vector_store=pickle.load(f)

    llm = HuggingFaceHub(repo_id="HuggingFaceH4/zephyr-7b-beta", model_kwargs={"temperature":0.9, "max_length": 500})
    chain=RetrievalQAWithSourcesChain.from_llm(llm=llm,retriever=vector_store.as_retriever())
    result=chain({"question":query},return_only_outputs=True)
    st.header("Answer:")
    st.write(result['answer'])

    sources=result.get("sources","")
    if sources:
        st.subheader("Sources:")
        st.write(sources)
