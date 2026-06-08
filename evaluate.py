import os
import gc
import nltk
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp

nltk.download('punkt')
nltk.download('wordnet')

FAISS_INDEX_PATH = "faiss_index"

MODELS_TO_TEST = {
    "Gemma 2 2B (Q4)": "gemma-2-2b-it-Q4_K_M.gguf",
    "Qwen 2.5 3B (Q4)": "qwen2.5-3b-instruct-q4_k_m.gguf",
    "Llama 3 8B (Q4)": "Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
    "Mistral 7B v0.2 (Q4)": "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
}

# Dinamički prompti (isti kao u app.py)
PROMPT_TEMPLATES = {
    "Gemma 2 2B (Q4)": "<bos><start_of_turn>user\nYou are an expert F1 assistant. Use ONLY the context to answer. If you don't know, say 'I don't know'.\n\nContext:\n{context}\n\nQuestion: {question}<end_of_turn>\n<start_of_turn>model\n",
    "Qwen 2.5 3B (Q4)": "<|im_start|>system\nYou are an expert F1 assistant. Use ONLY the context to answer. If you don't know, say 'I don't know'.<|im_end|>\n<|im_start|>user\nContext:\n{context}\n\nQuestion: {question}<|im_end|>\n<|im_start|>assistant\n",
    "Llama 3 8B (Q4)": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are an expert F1 assistant. Use ONLY the context to answer. If you don't know, say 'I don't know'.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nContext:\n{context}\n\nQuestion: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    "Mistral 7B v0.2 (Q4)": "[INST] You are an expert F1 assistant. Use ONLY the context to answer. If you don't know, say 'I don't know'.\n\nContext:\n{context}\n\nQuestion: {question} [/INST]"
}

test_cases = [
    {"question": "How many forward gears must the 2026 F1 gearbox have?", "expected": "The gearbox must have exactly 8 forward gears."},
    {"question": "What is the minimum weight of the 2026 F1 car without fuel?", "expected": "The minimum weight of the car without fuel is 768 kg."},
    {"question": "What are the power split details for the 2026 Power Unit?", "expected": "The ICE is limited to around 400kW, and the electrical power from the MGU-K is increased to 350kW."},
    {"question": "What happens if a team exceeds the cost cap?", "expected": "Teams face sporting and financial penalties, ranging from fines to deductions in wind tunnel time or constructor points."},
    {"question": "What is the role of the Safety Car according to the sporting regulations?", "expected": "The Safety Car is deployed to neutralize the race in case of an accident or unsafe conditions, and all cars must queue up behind it."}
]

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_rag_chain(vectorstore, llm, model_name):
    prompt_template = PROMPT_TEMPLATES[model_name]
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

def run_evaluation():
    print("Učitavam FAISS index...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    smoother = SmoothingFunction().method1
    all_results = []
    
    for model_name, model_path in MODELS_TO_TEST.items():
        if not os.path.exists(model_path):
            print(f"\n⚠️ Model '{model_name}' nije pronađen. Preskačem.")
            continue
            
        print(f"\n{'='*50}")
        print(f"🚀 EVALUACIJA: {model_name}")
        print(f"{'='*50}")
        
        llm = LlamaCpp(
            model_path=model_path,
            temperature=0.1,
            max_tokens=512,       # Smanjeno za brzinu
            n_ctx=4096,
            n_batch=512,
            n_threads=4,
            verbose=False
        )
        
        rag_chain = get_rag_chain(vectorstore, llm, model_name)
        
        for i, case in enumerate(test_cases):
            print(f"  -> Pitanje {i+1}/5...")
            llm_answer = rag_chain.invoke(case["question"])
            
            reference = [case["expected"].lower().split()]
            candidate = llm_answer.lower().split()
            
            try:
                bleu_score = sentence_bleu(reference, candidate, smoothing_function=smoother)
            except:
                bleu_score = 0.0
                
            rouge_scores = scorer.score(case["expected"], llm_answer)
            
            all_results.append({
                "Model": model_name,
                "Pitanje": case["question"],
                "Odgovor Chatbota": llm_answer.replace('\n', ' '),
                "Očekivano (Ground Truth)": case["expected"],
                "BLEU Score": round(bleu_score, 4),
                "ROUGE-1 F1": round(rouge_scores['rouge1'].fmeasure, 4),
                "ROUGE-L F1": round(rouge_scores['rougeL'].fmeasure, 4),
                "Ocjena (0-5)": "",
                "Komentar": ""
            })
            
        print(f"  ✅ Završeno. Oslobađam RAM...")
        del llm
        gc.collect()

    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv("evaluacija_usporedba_modela.csv", index=False)
        print("\n🎉 GOTVO! Rezultati spremljeni u 'evaluacija_usporedba_modela.csv'")

if __name__ == "__main__":
    run_evaluation()