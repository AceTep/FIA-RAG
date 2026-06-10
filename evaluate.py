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

nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)


try:
    from bert_score import score as bert_score_fn
    BERTSCORE_AVAILABLE = True
    print("BERTScore dostupan.")
except ImportError:
    BERTSCORE_AVAILABLE = False
    print("BERTScore nije instaliran (pip install bert-score). Preskačem tu metriku.")

FAISS_INDEX_PATH = "faiss_index"
MODELS_TO_TEST = {
    "Gemma 2 2B (Q4)": "models/gemma-2-2b-it-Q4_K_M.gguf",
    "Llama 3.2 3B (Q4)": "models/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
    "Phi-3 Mini 4K (Q4)": "models/Phi-3-mini-4k-instruct-Q4_K_M.gguf",
    "Qwen 2.5 1.5B (Q4)": "models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
}

PROMPT_TEMPLATES = {
    "Gemma 2 2B (Q4)": (
        "<start_of_turn>user\n"
        "You are an expert F1 assistant. Use ONLY the context to answer. If you don't know, say 'I don't know'.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}<end_of_turn>\n"
        "<start_of_turn>model\n"
    ),
    "Llama 3.2 3B (Q4)": (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "You are an expert F1 assistant. Use ONLY the context to answer. If you don't know, say 'I don't know'.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    ),
    "Phi-3 Mini 4K (Q4)": (
        "<|system|>\n"
        "You are an expert F1 assistant. Use ONLY the context to answer. If you don't know, say 'I don't know'.<|end|>\n"
        "<|user|>\n"
        "Context:\n{context}\n\n"
        "Question: {question}<|end|>\n"
        "<|assistant|>\n"
    ),
    "Qwen 2.5 1.5B (Q4)": (
        "<|im_start|>system\n"
        "You are an expert F1 assistant. Use ONLY the context to answer. If you don't know, say 'I don't know'.<|im_end|>\n"
        "<|im_start|>user\n"
        "Context:\n{context}\n\n"
        "Question: {question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    ),
}

STOP_TOKENS = {
    "Gemma 2 2B (Q4)": ["<end_of_turn>"],
    "Llama 3.2 3B (Q4)": ["<|eot_id|>", "<|start_header_id|>"],
    "Phi-3 Mini 4K (Q4)": ["<|end|>", "<|user|>"],
    "Qwen 2.5 1.5B (Q4)": ["<|im_end|>", "<|im_start|>"],
}
# ---------------------------------------------------------------------------
# Test pitanja s ground truth odgovorima
# ---------------------------------------------------------------------------
test_cases = [
    {
        "question": "How many forward gears must the 2026 F1 gearbox have?",
        "expected":  "The number of forward gear ratios must be 8.",
    },
    {
        "question": "What is the minimum weight of the 2026 F1 car without fuel?",
        "expected":  "The minimum mass is 724 kg plus the Nominal Tyre Mass, or 726 kg plus Nominal Tyre Mass during Qualifying and Sprint Qualifying sessions.",
    },
    {
        "question": "What are the power split details for the 2026 Power Unit?",
        "expected":  "The new 2026 power units produce over 1000 bhp (750 kW), with the power coming from a turbocharged 1.6-litre V6 internal combustion engine combined with an electric motor.",
    },
    {
        "question": "What happens if a team exceeds the cost cap?",
        "expected":  "A Material Overspend Breach results in a Constructors Championship points deduction and a financial penalty.",
    },
    {
        "question": "What is the role of the Safety Car according to the sporting regulations?",
        "expected":  "The Safety Car is deployed during a caution period to limit the speed of competing cars, leading the field at a safe speed while competitors are not allowed to overtake.",
    },
]


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def get_rag_chain(vectorstore, llm, model_name):
    prompt_template = PROMPT_TEMPLATES[model_name]
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


def run_evaluation():
    print("Učitavam FAISS index...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True
    )

    rouge = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    smoother = SmoothingFunction().method1
    all_results = []

    for model_name, model_path in MODELS_TO_TEST.items():
        if not os.path.exists(model_path):
            print(f"\nModel '{model_name}' nije pronađen na putanji '{model_path}'. Preskačem.")
            continue

        print(f"\n{'='*55}")
        print(f"EVALUACIJA: {model_name}")
        print(f"{'='*55}")

        llm = LlamaCpp(
            model_path=model_path,
            temperature=0.1,
            max_tokens=512,
            n_ctx=4096,
            n_batch=512,
            n_threads=4,
            stop=STOP_TOKENS[model_name],
            verbose=False,
        )

        rag_chain = get_rag_chain(vectorstore, llm, model_name)

        for i, case in enumerate(test_cases):
            print(f"  -> Pitanje {i+1}/{len(test_cases)}: {case['question'][:60]}...")
            llm_answer = rag_chain.invoke(case["question"])

            reference = [case["expected"].lower().split()]
            candidate = llm_answer.lower().split()

            # BLEU
            try:
                bleu = sentence_bleu(reference, candidate, smoothing_function=smoother)
            except Exception:
                bleu = 0.0

            r_scores = rouge.score(case["expected"], llm_answer)

            bert_f1 = ""
            if BERTSCORE_AVAILABLE:
                try:
                    _, _, F1 = bert_score_fn(
                        [llm_answer], [case["expected"]], lang="en", verbose=False
                    )
                    bert_f1 = round(float(F1[0]), 4)
                except Exception:
                    bert_f1 = ""

            all_results.append(
                {
                    "Model":                    model_name,
                    "Pitanje":                  case["question"],
                    "Odgovor Chatbota":         llm_answer.replace("\n", " ").strip(),
                    "Očekivano (Ground Truth)": case["expected"],
                    "BLEU Score":               round(bleu, 4),
                    "ROUGE-1 F1":               round(r_scores["rouge1"].fmeasure, 4),
                    "ROUGE-L F1":               round(r_scores["rougeL"].fmeasure, 4),
                    "BERTScore F1":             bert_f1,
                    "Ocjena (0-5)":             "",
                    "Komentar":                 "",
                }
            )

        del llm
        gc.collect()

    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv("evaluacija_usporedba_modela.csv", index=False, encoding="utf-8-sig")
        print("\nRezultati spremljeni u 'evaluacija_usporedba_modela.csv'")

        print("\nPROSJEČNE METRIKE PO MODELU:")
        summary = (
            df[df["BLEU Score"] != ""]
            .groupby("Model")[["BLEU Score", "ROUGE-1 F1", "ROUGE-L F1"]]
            .mean()
            .round(4)
        )
        print(summary.to_string())
    else:
        print("\nNema rezultata — provjeri putanje modela.")


if __name__ == "__main__":
    run_evaluation()
