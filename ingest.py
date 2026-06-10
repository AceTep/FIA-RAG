import os
import re
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from bs4 import BeautifulSoup

# Gasi "Ignoring wrong pointing object" upozorenja od pypdf biblioteke
logging.getLogger("pypdf").setLevel(logging.ERROR)

DATA_DIR = "data"
FAISS_INDEX_PATH = "faiss_index"
EDIT_DATA_DIR = "edit_data"

# ---------------------------------------------------------------------------
# PDF čišćenje — makni FIA header/footer šum koji se ponavlja na svakoj stranici
# ---------------------------------------------------------------------------
def clean_pdf_text(text: str) -> str:
    text = re.sub(r"FIA FORMULA ONE WORLD CHAMPIONSHIP.*?\n", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"FORMULA 1.*?REGULATIONS.*?\n", " ", text, flags=re.IGNORECASE)
    
    text = re.sub(r"^\s*\d{1,3}\s*$", " ", text, flags=re.MULTILINE)
    
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    text = re.sub(r"\t+", "  ", text)
    
    return text.strip()

# ---------------------------------------------------------------------------
# Wikipedia HTML čišćenje
# ---------------------------------------------------------------------------
def load_wikipedia_clean(file_path: str) -> list[Document]:
    with open(file_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")
        
    candidates = soup.find_all("div", class_="mw-parser-output")
    if candidates:
        content = max(candidates, key=lambda d: len(d.get_text(strip=True)))
    else:
        content = (
            soup.find("div", {"id": "mw-content-text"})
            or soup.find("main")
            or soup.find("article")
            or soup.find("div", {"id": "bodyContent"})
            or soup.body
        )
    if not content:
        print(f"UPOZORENJE: Nije pronađen content container u '{file_path}'. Preskačem.")
        return []
    print(f"Content container pronađen: <{content.name} class='{content.get('class','')}'>  [{file_path}]  ({len(content.get_text(strip=True))} znakova)")

    for toc in content.find_all("div", {"id": "toc"}):
        toc.decompose()

    for catlinks in content.find_all("div", {"id": "catlinks"}):
        catlinks.decompose()

    bad_classes = ["ambox", "tmbox", "navbox", "metadata", "sistersitebox", "refbegin", "refend", "noprint"]

    for tag in list(content.find_all(["table", "div"])):
        if tag is None or getattr(tag, "attrs", None) is None:
            continue
        classes = tag.attrs.get("class", [])
        if any(cls in classes for cls in bad_classes):
            tag.decompose()

    for reflist in content.find_all(["ol", "div"], class_=lambda x: x and ("references" in x or "reflist" in x)):
        reflist.decompose()

    for tag in content.find_all(["script", "style", "math"]):
        tag.decompose()
        

    SKIP_SECTIONS = ("references", "external links", "see also", "notes",
                     "bibliography", "further reading", "sources")
    for heading in list(content.find_all(["h2", "h3", "h4"])):
        if getattr(heading, "attrs", None) is None:
            continue
        heading_text = heading.get_text().lower().strip()
        heading_text = re.sub(r"\[.*?\]", "", heading_text).strip()
        if any(kw == heading_text or heading_text.startswith(kw) for kw in SKIP_SECTIONS):
            for sibling in list(heading.find_next_siblings()):
                sibling.decompose()
            heading.decompose()

    text = content.get_text(separator=" ", strip=True)

    text = re.sub(r"\[edit\]", " ", text)
    text = re.sub(r"\[ *edit *\]", " ", text)
    text = re.sub(r"Jump to.*?\n", " ", text)
    text = re.sub(r"Main article:.*?\n", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text) 

    if len(text.strip()) < 100:
        print(f"UPOZORENJE: Tekst iz '{file_path}' je gotovo prazan nakon čišćenja "
              f"({len(text.strip())} znakova)! Provjeri HTML strukturu.")
        return []

    print(f"Ekstrahirano {len(text.strip())} znakova iz '{os.path.basename(file_path)}'")
    return [Document(page_content=text, metadata={"source": os.path.basename(file_path)})]

# ---------------------------------------------------------------------------
# Glavni ingest
# ---------------------------------------------------------------------------
def ingest_documents():
    documents = []
    
    os.makedirs(EDIT_DATA_DIR, exist_ok=True)
    print(f"Mapa '{EDIT_DATA_DIR}' je spremna za spremanje očišćenih tekstova.\n")

    pdf_dir = os.path.join(DATA_DIR, "2026")
    if os.path.exists(pdf_dir):
        for filename in sorted(os.listdir(pdf_dir)):
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(pdf_dir, filename))
                raw_docs = loader.load()

                for doc in raw_docs:
                    doc.page_content = clean_pdf_text(doc.page_content)

                raw_docs = [d for d in raw_docs if len(d.page_content.strip()) > 50]

                documents.extend(raw_docs)
                print(f"PDF učitan i očišćen ({len(raw_docs)} stranica): {filename}")
    else:
        print(f"Mapa {pdf_dir} ne postoji!")

    if os.path.exists(DATA_DIR):
        for filename in sorted(os.listdir(DATA_DIR)):
            if filename.endswith(".html"):
                docs = load_wikipedia_clean(os.path.join(DATA_DIR, filename))
                
                if docs:
                    clean_filename = filename.replace(".html", "_cleaned.txt")
                    output_path = os.path.join(EDIT_DATA_DIR, clean_filename)
                    
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(docs[0].page_content)
                        
                    print(f"HTML učitan, očišćen i spremljen u '{EDIT_DATA_DIR}': {clean_filename}")
                
                documents.extend(docs)
    else:
        print(f"Mapa {DATA_DIR} ne postoji!")

    print(f"\nUkupno učitano dokumenata / stranica: {len(documents)}")

    if not documents:
        print("Nema dokumenata za procesuiranje. Provjeri strukturu mapa.")
        return

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", " "],
    )
    chunks = text_splitter.split_documents(documents)

    chunks = [c for c in chunks if len(c.page_content.strip()) > 30]

    print(f"Broj chunkova nakon dijeljenja: {len(chunks)}")

    print("Generiram embedde...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(FAISS_INDEX_PATH)
    
    print(f"\nFAISS index uspješno spremljen u '{FAISS_INDEX_PATH}'")
    print(f"Ukupno vektora u indexu: {vectorstore.index.ntotal}")

if __name__ == "__main__":
    ingest_documents()