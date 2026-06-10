import uuid
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from sklearn.decomposition import PCA

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
FAISS_INDEX_PATH = "faiss_index"

st.set_page_config(
    page_title="F1 2026 RAG Assistant",
    layout="wide"
)

# ------------------------------------------------------------------
# MODELS
# ------------------------------------------------------------------
MODEL_ORDER = [
    "Phi-3 Mini 4K (Q4)",
    "Llama 3.2 3B (Q4)",
    "Gemma 2 2B (Q4)",
    "Qwen 2.5 1.5B (Q4)"
]

MODEL_PATHS = {
    "Phi-3 Mini 4K (Q4)":  "models/Phi-3-mini-4k-instruct-Q4_K_M.gguf",
    "Llama 3.2 3B (Q4)":   "models/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
    "Gemma 2 2B (Q4)":     "models/gemma-2-2b-it-Q4_K_M.gguf",
    "Qwen 2.5 1.5B (Q4)":  "models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
}

PROMPTS = {
    "Phi-3 Mini 4K (Q4)": (
        "<|system|>You are an F1 regulations expert. Use only context.<|end|>\n"
        "<|user|>Context:\n{context}\n\nQuestion: {question}<|end|>\n"
        "<|assistant|>\n"
    ),
    "Llama 3.2 3B (Q4)": (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        "Use only context.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n{context}\n{question}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    ),
    "Gemma 2 2B (Q4)": (
        "<start_of_turn>user\nContext:\n{context}\n\nQuestion:{question}<end_of_turn>\n<start_of_turn>model\n"
    ),
    "Qwen 2.5 1.5B (Q4)": (
        "<|im_start|>user\n{context}\n{question}<|im_end|>\n<|im_start|>assistant\n"
    ),
}

STOP_TOKENS = {
    "Phi-3 Mini 4K (Q4)":  ["<|end|>"],
    "Llama 3.2 3B (Q4)":   ["<|eot_id|>"],
    "Gemma 2 2B (Q4)":     ["<end_of_turn>"],
    "Qwen 2.5 1.5B (Q4)":  ["<|im_end|>"],
}

# ------------------------------------------------------------------
# LIGHT MODE CSS
# ------------------------------------------------------------------
st.markdown("""
<style>
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div {
    background-color: #ebebeb !important;
    border-right: 1px solid #999999 !important;
}
[data-testid="stChatMessage"] {
    background-color: #f5f5f5 !important;
    border-radius: 8px !important;
    margin: 4px 0 !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    border-left: 2px solid #444444 !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    border-left: 2px solid #888888 !important;
}
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-thumb { background: #999999; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# LOAD VECTORSTORE
# ------------------------------------------------------------------
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

vectorstore = load_vectorstore()

# ------------------------------------------------------------------
# PCA
# ------------------------------------------------------------------
@st.cache_resource
def build_pca_projection(_vectorstore):
    index = _vectorstore.index
    n = index.ntotal
    all_vectors = index.reconstruct_n(0, n)
    all_texts = []
    for i in range(n):
        doc_id = _vectorstore.index_to_docstore_id[i]
        doc    = _vectorstore.docstore.search(doc_id)
        text   = doc.page_content if doc else ""
        all_texts.append(text[:200] + ("…" if len(text) > 200 else ""))
    pca    = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(all_vectors)
    return pca, coords, all_texts

pca_model, all_coords, all_texts = build_pca_projection(vectorstore)

# ------------------------------------------------------------------
# EMBED QUERY
# ------------------------------------------------------------------
def embed_query(text: str) -> np.ndarray:
    return np.array(
        vectorstore.embedding_function.embed_query(text),
        dtype=np.float32
    ).reshape(1, -1)

# ------------------------------------------------------------------
# EMBEDDING FIGURE
# ------------------------------------------------------------------
def build_embedding_figure(query_text: str, retrieved_docs: list) -> go.Figure:
    retrieved_set = set(d.page_content[:200] for d in retrieved_docs)

    colors, sizes = [], []
    for txt in all_texts:
        if txt.rstrip("…")[:200] in retrieved_set or txt[:200] in retrieved_set:
            colors.append("#333333")
            sizes.append(9)
        else:
            colors.append("rgba(150,150,150,0.4)")
            sizes.append(4)

    hover_chunks = [f"<b>Chunk</b><br>{t}<extra></extra>" for t in all_texts]
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=all_coords[:, 0], y=all_coords[:, 1],
        mode="markers",
        marker=dict(color=colors, size=sizes, opacity=0.75),
        hovertemplate=hover_chunks,
        showlegend=False,
    ))

    q_coord = pca_model.transform(embed_query(query_text))[0]

    fig.add_trace(go.Scatter(
        x=[q_coord[0]], y=[q_coord[1]],
        mode="markers",
        marker=dict(symbol="star", size=16, color="#666666",
                    line=dict(color="#111111", width=1)),
        hovertemplate=f"<b>Query</b><br>{query_text[:120]}<extra></extra>",
        showlegend=False,
    ))

    for doc in retrieved_docs:
        for i, txt in enumerate(all_texts):
            if txt.startswith(doc.page_content[:80]):
                fig.add_trace(go.Scatter(
                    x=[q_coord[0], all_coords[i, 0]],
                    y=[q_coord[1], all_coords[i, 1]],
                    mode="lines",
                    line=dict(color="rgba(100,100,100,0.3)", width=1.2, dash="dot"),
                    hoverinfo="skip",
                    showlegend=False,
                ))
                break

    fig.update_layout(
        paper_bgcolor="#f5f5f5",
        plot_bgcolor="#ffffff",
        font_color="#111111",
        margin=dict(l=40, r=20, t=30, b=40),
        height=400,
        xaxis=dict(title="PC1", showgrid=True, gridcolor="rgba(0,0,0,0.07)",
                   zeroline=False, color="#111111"),
        yaxis=dict(title="PC2", showgrid=True, gridcolor="rgba(0,0,0,0.07)",
                   zeroline=False, color="#111111"),
        hoverlabel=dict(bgcolor="#f0f0f0", font_size=12, font_color="#111111"),
    )

    fig.add_annotation(
        x=0.01, y=0.99, xref="paper", yref="paper",
        text=(
            '<span style="color:rgba(150,150,150,0.8);">●</span> svi chunkovi &nbsp;&nbsp;'
            '<span style="color:#333333;">●</span> retrieved &nbsp;&nbsp;'
            '<span style="color:#666666;">★</span> query'
        ),
        showarrow=False, font=dict(size=11, color="#111111"),
        align="left", bgcolor="rgba(245,245,245,0.9)",
        bordercolor="rgba(0,0,0,0.1)", borderwidth=1,
    )
    return fig

# ------------------------------------------------------------------
# MODEL LOADER
# ------------------------------------------------------------------
@st.cache_resource
def load_llm(model_name: str):
    return LlamaCpp(
        model_path=MODEL_PATHS[model_name],
        temperature=0.1,
        max_tokens=512,
        n_ctx=4096,
        n_batch=512,
        n_threads=4,
        stop=STOP_TOKENS[model_name],
        verbose=False,
        streaming=True,

    )

# ------------------------------------------------------------------
# RAG PIPELINE
# ------------------------------------------------------------------
def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

def build_chain(model_name: str):
    llm       = load_llm(model_name)
    prompt    = PromptTemplate(
        template=PROMPTS[model_name],
        input_variables=["context", "question"]
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )
    return chain, retriever

# ------------------------------------------------------------------
# HEADER
# ------------------------------------------------------------------
st.markdown("## F1 2026 Regulations RAG")
st.caption("Local RAG assistant · F1 2026 regulations")

# ------------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------------
st.sidebar.title("Model")

selected_model = st.sidebar.radio(
    "Choose model (heavy → light)",
    MODEL_ORDER
)

if "active_model" not in st.session_state:
    st.session_state.active_model = selected_model

if st.session_state.active_model != selected_model:
    st.session_state.messages = []
    st.session_state.active_model = selected_model
    st.toast(f"Model: {selected_model}", icon="🔄")

st.sidebar.markdown("---")
st.sidebar.markdown("**Embedding map**")
show_map = st.sidebar.toggle(
    "Show after each answer",
    value=True,
    help="PCA 2D projekcija svih chunkova"
)

n_total = vectorstore.index.ntotal
st.sidebar.caption(f"{n_total} chunkova · PCA 2D")

# ------------------------------------------------------------------
# BUILD CHAIN
# ------------------------------------------------------------------
chain, retriever = build_chain(selected_model)

# ------------------------------------------------------------------
# SESSION STATE
# ------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ------------------------------------------------------------------
# CHAT HISTORY
# ------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            if "sources" in msg:
                with st.expander("Sources used", expanded=False):
                    for i, src in enumerate(msg["sources"], 1):
                        st.caption(f"**Chunk {i}:** {src[:300]}{'...' if len(src) > 300 else ''}")
            if show_map and msg.get("embedding_fig"):
                with st.expander("Embedding map", expanded=False):
                    st.plotly_chart(
                        msg["embedding_fig"],
                        use_container_width=True,
                        key=f"map_{msg['id']}"
                    )

# ------------------------------------------------------------------
# INPUT & RESPONSE (SA STREAMINGOM)
# ------------------------------------------------------------------

user_input = st.chat_input("Ask about F1 2026 regulations...")
if user_input:
    # 1. Prikaži korisnikov upit
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # 2. Pripremi odgovor asistenta
    with st.chat_message("assistant"):
        # Prvo brzo dohvati dokumente (trebaju nam za izvore i PCA mapu)
        with st.spinner("Retrieving context..."):
            docs = retriever.invoke(user_input)
        
        # Definiraj generator koji će yield-ati tokene jedan po jedan
        def response_generator():
            # POPRAVAK: Prosljeđujemo string, a ne dict!
            # chain.stream će automatski proslijediti string i retrieveru i promptu.
            for chunk in chain.stream(user_input):
                yield chunk
        
        # Streamlit će ovo prikazivati u realnom vremenu i vratiti cijeli tekst na kraju
        full_answer = st.write_stream(response_generator())
        
        # 3. Prikaži izvore (nakon što je generiranje teksta gotovo)
        sources = [d.page_content for d in docs]
        with st.expander("Sources used", expanded=False):
            for i, src in enumerate(sources, 1):
                st.caption(f"**Chunk {i}:** {src[:300]}{'...' if len(src) > 300 else ''}")

        # 4. Prikaži Embedding mapu (ako je uključeno)
        fig = None
        if show_map:
            with st.expander("Embedding map – zašto ovi chunkovi?", expanded=False):
                with st.spinner("Projecting embeddings..."):
                    fig = build_embedding_figure(user_input, docs)
                st.plotly_chart(fig, use_container_width=True, key=f"map_live_{uuid.uuid4()}")

    # 5. Spremi u session state za povijest razgovora
    msg_id = str(uuid.uuid4())
    st.session_state.messages.append({
        "id":            msg_id,
        "role":          "assistant",
        "content":       full_answer,  # st.write_stream vraća potpuni sastavljeni string
        "sources":       sources,
        "embedding_fig": fig,
    })