### 📄 Sadržaj datoteke: `preuzimanje_modela.md`

```markdown
# 📥 Skripta za preuzimanje LLM modela (GGUF format)

Ova skripta preuzima optimizirane (kvantizirane) GGUF modele spremne za rad isključivo na CPU-u pomoću `llama.cpp` / `langchain`.

## 🚀 Opcija 1: Preuzmi SVE modele odjednom (Copy-Paste u terminal)
Ako imaš dovoljno prostora na disku (~15 GB), kopiraj i zalijepi ovo u terminal unutar mape `projekt2`:

```bash
# 1. Qwen 2.5 3B (NAJBOLJI OMJER: brzina + kvaliteta) ~2.0 GB
wget -O qwen2.5-3b-instruct-q4_k_m.gguf "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf"

# 2. Mistral 7B v0.2 Q3 (PREPORUKA S VJEŽBI, ali olakšana verzija za CPU) ~3.4 GB
wget -O mistral-7b-instruct-v0.2.Q3_K_M.gguf "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q3_K_M.gguf"

# 3. Gemma 2 2B (ULTRA LAGANI: munjevit, minimalni RAM) ~1.7 GB
wget -O gemma-2-2b-it-Q4_K_M.gguf "https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf"

# 4. Phi-3 Mini 3.8B (MICROSOFTOV "MALI DIV": pametniji od 7B, a zauzima upola manje) ~2.5 GB
wget -O Phi-3-mini-4k-instruct-Q4_K_M.gguf "https://huggingface.co/bartowski/Phi-3-mini-4k-instruct-GGUF/resolve/main/Phi-3-mini-4k-instruct-Q4_K_M.gguf"

# 5. Llama 3.2 3B (META-IN NOVI EDGE MODEL: optimiziran za CPU) ~2.0 GB
wget -O Llama-3.2-3B-Instruct-Q4_K_M.gguf "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
```

---

## 📊 Opis modela (Zašto baš ovi?)

| Model | Veličina datoteke | Potrošnja RAM-a | Brzina na CPU-u | Zašto ga uključiti u projekt? |
| :--- | :---: | :---: | :---: | :--- |
| **Qwen 2.5 3B** | ~2.0 GB | ~2.5 GB | ⚡⚡⚡⚡ | **Defaultni izbor.** Najbolji balans brzine i točnosti za RAG na CPU-u. |
| **Mistral 7B (Q3)** | ~3.4 GB | ~4.0 GB | ⚡⚡ | **Obavezan za izvještaj** jer ga spominju upute zadatka. Q3 verzija štedi RAM. |
| **Gemma 2 2B** | ~1.7 GB | ~2.2 GB | ⚡⚡⚡⚡⚡ | Dokaz da RAG radi čak i na ultra-slabim konfiguracijama. |
| **Phi-3 Mini 3.8B**| ~2.5 GB | ~3.0 GB | ⚡⚡⚡ | "Mali div". Često nadmašuje Mistral 7B u logici, a zauzima upola manje RAM-a. |
| **Llama 3.2 3B** | ~2.0 GB | ~2.5 GB | ⚡⚡⚡⚡ | Najnoviji Meta-in model specifično treniran za "edge" uređaje i CPU. |

---

## ⚙️ VAŽNO: Dodavanje u Python kod (`app.py` i `evaluate.py`)

Nakon preuzimanja, moraš dodati ove modele u rječnike u svom kodu. Kopiraj ove dijelove u svoje datoteke:

### 1. U `app.py` (dodaj u `AVAILABLE_MODELS` i `PROMPT_TEMPLATES`):
```python
AVAILABLE_MODELS = {
    "Qwen 2.5 3B (NAJBRŽI - Preporučeno)": "qwen2.5-3b-instruct-q4_k_m.gguf",
    "Phi-3 Mini (Mali div)": "Phi-3-mini-4k-instruct-Q4_K_M.gguf",
    "Llama 3.2 3B (Meta Edge)": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
    "Gemma 2 2B (Ultra lagani)": "gemma-2-2b-it-Q4_K_M.gguf",
    "Mistral 7B v0.2 Q3 (Standard)": "mistral-7b-instruct-v0.2.Q3_K_M.gguf"
}

PROMPT_TEMPLATES = {
    "Qwen 2.5 3B (NAJBRŽI - Preporučeno)": "<|im_start|>system\nYou are an expert F1 assistant. Use ONLY the context to answer. If you don't know, say 'I don't know'.<|im_end|>\n<|im_start|>user\nContext:\n{context}\n\nQuestion: {question}<|im_end|>\n<|im_start|>assistant\n",
    "Phi-3 Mini (Mali div)": "<|user|>\nYou are an expert F1 assistant. Use ONLY the context to answer. If you don't know, say 'I don't know'.\n\nContext:\n{context}\n\nQuestion: {question}<|end|>\n<|assistant|>\n",
    "Llama 3.2 3B (Meta Edge)": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are an expert F1 assistant. Use ONLY the context to answer. If you don't know, say 'I don't know'.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nContext:\n{context}\n\nQuestion: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    "Gemma 2 2B (Ultra lagani)": "<bos><start_of_turn>user\nYou are an expert F1 assistant. Use ONLY the context to answer. If you don't know, say 'I don't know'.\n\nContext:\n{context}\n\nQuestion: {question}<end_of_turn>\n<start_of_turn>model\n",
    "Mistral 7B v0.2 Q3 (Standard)": "[INST] You are an expert F1 assistant. Use ONLY the context to answer. If you don't know, say 'I don't know'.\n\nContext:\n{context}\n\nQuestion: {question} [/INST]"
}
```

### 2. U `evaluate.py` (dodaj u `MODELS_TO_TEST` i `PROMPT_TEMPLATES`):
*(Koristi iste nazive datoteka i prompt templateove kao gore, samo ih preslikaj u rječnik `MODELS_TO_TEST`)*

---

## 💡 Savjet za uštedu prostora
Ako ti ponestaje prostora na disku, **obavezno zadrži samo ova 3 modela** za savršenu evaluaciju:
1. `qwen2.5-3b-instruct-q4_k_m.gguf` (Tvoj glavni, brzi model)
2. `mistral-7b-instruct-v0.2.Q3_K_M.gguf` (Za zadovoljavanje uvjeta zadatka)
3. `Phi-3-mini-4k-instruct-Q4_K_M.gguf` (Za impresivnu usporedbu "novi mali vs. stari veliki")

Ostale možeš slobodno izbrisati nakon testiranja.


Otvori terminal u korijenu projekta i pokreni:

# Kreiraj data mapu ako ne postoji
mkdir -p data

# Preuzmi Wikipedia HTML datoteke
wget -O data/f1_car.html "https://en.wikipedia.org/wiki/Formula_One_car"
wget -O data/f1_2026_season.html "https://en.wikipedia.org/wiki/2026_Formula_One_World_Championship"
wget -O data/power_unit.html "https://en.wikipedia.org/wiki/Formula_One_engines"
wget -O data/safety_car.html "https://en.wikipedia.org/wiki/Safety_car"

(Napomena: FIA PDF-ove u mapu data/2026/ moraš ručno preuzeti sa službene FIA stranice jer zahtijevaju specifične URL-ove ili registraciju, ali ovi HTML-ovi su ti sada spremni za ingest.py)