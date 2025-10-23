from langchain_community.chat_models import ChatZhipuAI
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

def get_contents(url):
    from bs4 import BeautifulSoup
    import requests

    html = requests.get(url).text

    soup = BeautifulSoup(html, "html.parser")

    # Remove unwanted sections (like nav, header, footer, script, style)
    for tag in soup(["nav", "header", "footer", "script", "style", "aside"]):
        tag.decompose()

    # Extract main readable tags only
    content_tags = soup.find_all(["h1", "h2", "h3", "p", "li", "blockquote", "pre", "code"])

    clean_text = "\n".join(tag.get_text(strip=True, separator=" ") for tag in content_tags)

    return clean_text

def text_chuking(docs):
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,    
        chunk_overlap=200   
    )
    docs_split = splitter.split_text(docs)
    docs_split = [Document(page_content=chunk) for chunk in docs_split]
    return docs_split


def get_answer(urls, query):
    docs = []
    for url in urls:
        docs.append(get_contents(url))
    all_documents = "\n".join(docs)
    documents = text_chuking(all_documents)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma.from_documents(
    documents,
    embedding=embeddings,
    # persist_directory="./chroma_store"
    )
    # retriever = db.as_retriever(search_kwargs={"k": 3})
    # query = "What are AI agents and how do they work?"
    results = db.similarity_search_with_score(query, k=3)
    for i, (doc, score) in enumerate(results):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Raw similarity score: {score:.4f}")
        print(f"Confidence (1 - score): {1 - score:.4f}")  
        print(doc.page_content[:500])


get_answer(["https://lilianweng.github.io/posts/2023-06-23-agent/"], 'Agent System Overview')



    
