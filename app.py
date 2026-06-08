import streamlit as st
import os
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp

FAISS_INDEX_PATH = "faiss_index"

# 1. Qwen je postavljen kao DEFAULT jer je najbrži i najpouzdaniji na CPU-u
AVAILABLE_MODELS = {
    "Qwen 2.5 3B (NAJBRŽI - Preporučeno)": "qwen2.5-3b-instruct-q4_k_m.gguf",
    "Gemma 2 2B (Ultra lagani)": "gemma-2-2b-it-Q4_K_M.gguf",
    "Mistral 7B v0.2 (Standard)": "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    "Llama 3 8B (Najpametniji, ali sporiji)": "Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
}

# Dinamički prompti za svaki model
PROMPT_TEMPLATES = {
    "Qwen 2.5 3B (NAJBRŽI - Preporučeno)": "<|im_start|>system\nYou are an expert Formula 1 regulations assistant. Use ONLY the provided context to answer. If you don't know, say 'I don't know based on the context.' Cite the specific regulation if possible.<|im_end|>\n<|im_start|>user\nContext:\n{context}\n\nQuestion: {question}<|im_end|>\n<|im_start|>assistant\n",
    "Gemma 2 2B (Ultra lagani)": "<bos><start_of_turn>user\nYou are an expert Formula 1 regulations assistant. Use ONLY the provided context to answer. If you don't know, say 'I don't know based on the context.' Cite the specific regulation if possible.\n\nContext:\n{context}\n\nQuestion: {question}<end_of_turn>\n<start_of_turn>model\n",
    "Mistral 7B v0.2 (Standard)": "[INST] You are an expert Formula 1 regulations assistant. Use ONLY the provided context to answer. If you don't know, say 'I don't know based on the context.' Cite the specific regulation if possible.\n\nContext:\n{context}\n\nQuestion: {question} [/INST]",
    "Llama 3 8B (Najpametniji, ali sporiji)": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are an expert Formula 1 regulations assistant. Use ONLY the provided context to answer. If you don't know, say 'I don't know based on the context.' Cite the specific regulation if possible.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nContext:\n{context}\n\nQuestion: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
}

@st.cache_resource
def load_models(selected_model_key):
    model_path = AVAILABLE_MODELS[selected_model_key]
    prompt_template = PROMPT_TEMPLATES[selected_model_key]
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    
    # OPTIMIZACIJA ZA BRZINU NA CPU-u
    llm = LlamaCpp(
        model_path=model_path,
        temperature=0.1,
        max_tokens=512,       # Smanjeno s 1024 na 512 za 2x brži odgovor
        n_ctx=4096,
        n_batch=512,
        n_threads=4,          # Prilagodi broju fizičkih jezgri svog CPU-a (npr. 4 ili 8)
        verbose=False
    )
    
    return vectorstore, llm, embeddings, prompt_template

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_rag_chain(vectorstore, llm, prompt_template):
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain, retriever

# --- STREAMLIT UI ---
st.set_page_config(page_title="F1 2026 RAG Chatbot", layout="wide")
st.title("🏎️ F1 2026 Regulations Expert (Local RAG)")

selected_model = st.selectbox("Odaberi LLM model:", list(AVAILABLE_MODELS.keys()), index=0) # index=0 = Qwen default

vectorstore, llm, embeddings, prompt_template = load_models(selected_model)
rag_chain, retriever = get_rag_chain(vectorstore, llm, prompt_template)

tab1, tab2 = st.tabs(["💬 Chat with FIA Rules", "📊 Embedding Visualization"])

with tab1:
    st.markdown("Ask anything about the 2026 F1 Sporting, Technical, or Financial regulations.")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What are the new power unit rules for 2026?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner(f"Thinking with {selected_model}..."):
                answer = rag_chain.invoke(prompt)
                docs = retriever.invoke(prompt)
                
                st.markdown(answer)
                with st.expander("View Source Documents"):
                    for i, doc in enumerate(docs):
                        st.write(f"**Source {i+1}:** {doc.metadata.get('source', 'Unknown')}")
                        st.caption(doc.page_content[:400] + "...")
                        
        st.session_state.messages.append({"role": "assistant", "content": answer})

with tab2:
    st.markdown("### PCA Visualization of Document Embeddings")
    if st.button("Generate Visualization"):
        with st.spinner("Extracting vectors and running PCA..."):
            index = vectorstore.index
            num_vectors = index.ntotal
            sample_size = min(2000, num_vectors) 
            
            embeddings_matrix = index.reconstruct_n(0, num_vectors)
            if num_vectors > sample_size:
                indices = np.random.choice(num_vectors, sample_size, replace=False)
                embeddings_matrix = embeddings_matrix[indices]
            
            pca = PCA(n_components=2, random_state=42)
            pca_result = pca.fit_transform(embeddings_matrix)
            
            fig = px.scatter(x=pca_result[:, 0], y=pca_result[:, 1], opacity=0.6, color_discrete_sequence=['#E10600'])
            fig.update_layout(height=500, title="2D PCA Projection of F1 2026 Document Chunks")
            st.plotly_chart(fig, use_container_width=True)