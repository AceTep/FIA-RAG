
# F1 2026 Regulations RAG Chatbot

**Lokalni specijalizirani chatbot s RAG arhitekturom za FIA Formula 1 2026 pravilnike.**

Projekt izrađen u sklopu kolegija **Prikaz znanja i rezoniranje o znanju** (FIDIT, Sveučilište u Rijeci, ak. god. 2025./26.).

---

## O projektu

Ovaj projekt demonstrira primjenu **RAG (Retrieval-Augmented Generation)** arhitekture za izradu lokalnog, specijaliziranog chatbota. Sustav odgovara na kompleksna tehnička, sportska i financijska pitanja iz službenih FIA Formula 1 pravilnika za sezonu 2026. 

Sve komponente (LLM inferencija, vektorska baza, GUI) rade **potpuno lokalno na CPU-u**, bez ikakvog oslanjanja na vanjske API-je ili internetsku vezu.

### Ključne značajke
-  **Lokalni LLM-ovi:** Podrška za 4 različita kvantizirana (Q4) modela optimizirana za CPU.
-  **RAG Pipeline:** Dinamički dohvat konteksta iz vlastite baze znanja (FAISS).
-  **Streamlit GUI:** Moderno sučelje s prikazom izvora i streamingom odgovora u realnom vremenu.
-  **PCA Vizualizacija:** Interaktivni graf koji prikazuje semantički prostor dokumenata i poziciju korisničkog upita.
-  **Automatska evaluacija:** Usporedba modela koristeći BLEU, ROUGE i BERTScore metrike.

---

## 🛠️ Tehnološki stack

| Komponenta | Korišteni alati / Biblioteke |
| :--- | :--- |
| **RAG Framework** | LangChain, langchain-community |
| **Lokalni LLM pogon** | llama-cpp-python (GGUF format) |
| **Vektorska baza** | FAISS-CPU |
| **Embedding model** | sentence-transformers (all-MiniLM-L6-v2) |
| **Parsing dokumenata** | PyPDF, BeautifulSoup4 |
| **GUI** | Streamlit |
| **Vizualizacija** | Plotly, scikit-learn (PCA) |
| **Evaluacija** | NLTK (BLEU), rouge-score, bert-score, pandas |

---

##  Struktura projekta

```text
projekt2/
│
├── data/                  # Izvorni dokumenti (6x PDF + 4x HTML)
│   ├── 2026/              # FIA pravilnici
│   └── *.html             # Wikipedia stranice
│
├── edit_data/             # Očišćeni tekstovi (auto-generirano)
├── faiss_index/           # FAISS vektorska baza (auto-generirano)
├── models/                # Lokalni GGUF modeli (preuzeti zasebno)
│
├── app.py                 # Streamlit GUI aplikacija
├── ingest.py              # Ingestion pipeline (chunking, embedding, FAISS)
├── evaluate.py            # Evaluacijska skripta (BLEU, ROUGE, BERTScore)
├── requirements.txt       # Python ovisnosti
├── preuzimanje_modela.md  # Upute i linkovi za preuzimanje modela
├── preuzimanje_data_html.md    # Upute i linkovi za preuzimanje podataka
└── README.md              # Ovaj dokument
```

---

##  Instalacija i pokretanje

### 1. Preduvjeti
- Python 3.10+
- Preporučeno: 8GB+ RAM-a (za nesmetan rad s 7B/3B modelima)

### 2. Postavljanje okruženja
```bash
# Kreiranje virtualnog okruženja
python -m venv venv
source venv/bin/activate  # (Linux/Mac)
# ili: venv\Scripts\activate (Windows)

# Instalacija ovisnosti
pip install -r requirements.txt
```
*(Napomena: Ako instalacija `llama-cpp-python` ne uspije zbog kompajliranja, koristite precompiled wheel: `pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu`)*

### 3. Preuzimanje modela
GGUF modele potrebno je ručno preuzeti s Hugging Facea i spremiti u mapu `models/`. Detaljne upute i `wget` naredbe nalaze se u datoteci **`preuzimanje_modela.md`**.

Podržani modeli:
- `Phi-3-mini-4k-instruct-Q4_K_M.gguf` (3.8B)
- `Llama-3.2-3B-Instruct-Q4_K_M.gguf` (3B)
- `gemma-2-2b-it-Q4_K_M.gguf` (2B)
- `Qwen2.5-1.5B-Instruct-Q4_K_M.gguf` (1.5B)

### 4. Priprema baze znanja (Ingestion)
Pokretanjem ove skripte učitavaju se PDF i HTML dokumenti, čiste se od šuma, dijele na chunkove i spremaju u FAISS bazu.
```bash
python ingest.py
```

### 5. Pokretanje chatbota
```bash
streamlit run app.py
```
Aplikacija će se otvoriti u pregledniku na adresi `http://localhost:8501`.

### 6. Evaluacija (Opcionalno)
Za pokretanje automatizirane usporedbe sva 4 modela na testnim pitanjima:
```bash
python evaluate.py
```
Rezultati se spremaju u `evaluacija_usporedba_modela.csv`.

---

##  Evaluacija i rezultati

U sklopu projekta provedena je detaljna evaluacija 4 lokalna modela na 5 testnih pitanja iz domene. 

**Ključni nalazi:**
- **Gemma 2 2B** i **Qwen 2.5 1.5B** pokazali su iznenađujuće dobre rezultate za svoju veličinu, postižući visoke BERTScore vrijednosti (>0.88).
- Veći broj parametara ne garantira bolji RAG odgovor; kvaliteta prompt templatea i chunking strategije često su presudni faktori.
- Svi modeli uspješno koriste kontekst iz FAISS baze umjesto oslanjanja na parametarsko znanje (sprječavanje *knowledge leakage*-a).

Detaljna analiza metrika i ručne ocjene dostupna je u CSV datoteci i završnom izvještaju.

