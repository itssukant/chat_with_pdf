from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFMinerLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
import os
 
def main():
    documents = []
    for root, dirs, files in os.walk("docs"):
        for file in files:
            if file.endswith(".pdf"):
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")
 
                # Create an instance of PDFMinerLoader with the file path
                loader = PDFMinerLoader(file_path)
               
                # Load documents
                loaded_docs = loader.load()
                documents.extend(loaded_docs)
 
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
 
    # Create embeddings
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
   
    # Create and persist the vector store
    db = Chroma.from_documents(texts, embeddings, persist_directory="db")
 
if __name__ == "__main__":
    main()
