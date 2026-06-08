import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from bs4 import BeautifulSoup

DATA_DIR = "data"
FAISS_INDEX_PATH = "faiss_index"

def load_wikipedia_clean(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')
        
    content = soup.find('div', {'id': 'mw-content-text'})
    if not content:
        return []
    
    # Makni sav nepotrebni šum
    for tag in content.find_all(['table', 'sup', 'style', 'script', 'figure', 'navbox', 'reflist']):
        tag.decompose()
        
    # NOVO: Makni i sekciju "References", "See also" i "External links"
    for heading in content.find_all(['h2', 'h3']):
        heading_text = heading.get_text().lower()
        if 'references' in heading_text or 'external links' in heading_text or 'see also' in heading_text:
            # Makni sve elemente nakon tog naslova
            for sibling in heading.find_next_siblings():
                sibling.decompose()
            heading.decompose()
            break # Stani jer smo obrisali sve do kraja
            
    text = content.get_text(separator='\n', strip=True)
    return [Document(page_content=text, metadata={"source": os.path.basename(file_path)})]

def ingest_documents():
    documents = []
    
    # 1. Učitavanje FIA PDF-ova
    pdf_dir = os.path.join(DATA_DIR, "2026")
    if os.path.exists(pdf_dir):
        for filename in os.listdir(pdf_dir):
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(pdf_dir, filename))
                documents.extend(loader.load())
                print(f"Učitan PDF: {filename}")
    else:
        print(f"Upozorenje: Mapa {pdf_dir} ne postoji!")
            
    # 2. Učitavanje i čišćenje Wikipedia HTML-ova
    if os.path.exists(DATA_DIR):
        for filename in os.listdir(DATA_DIR):
            if filename.endswith(".html"):
                docs = load_wikipedia_clean(os.path.join(DATA_DIR, filename))
                documents.extend(docs)
                print(f"Učitan i očišćen HTML: {filename}")
    else:
        print(f"Upozorenje: Mapa {DATA_DIR} ne postoji!")
            
    print(f"Ukupno učitano dokumenata: {len(documents)}")
    
    if not documents:
        print("Nema dokumenata za procesuiranje. Provjeri strukturu mapa.")
        return
    
    # 3. Chunkanje (dijeljenje na manje dijelove)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Broj chunkova: {len(chunks)}")
    
    # 4. Embedding i spremanje u FAISS
    print("Generiram embedde (ovo može potrajati na CPU-u)...")
    # NOVO (puno bolja pretraga, a i dalje radi glatko na CPU-u):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Spremi lokalno
    vectorstore.save_local(FAISS_INDEX_PATH)
    print(f"FAISS index spremljen u {FAISS_INDEX_PATH}")

if __name__ == "__main__":
    ingest_documents()