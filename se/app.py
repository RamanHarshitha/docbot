import streamlit as st
import pdfplumber
import re
from sentence_transformers import SentenceTransformer, util
import faiss
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

st.set_page_config(page_title="Universal PDF Chatbot", page_icon="📚", layout="wide")
st.title("💬 Universal PDF Q&A Chatbot (Fast + Accurate)")

# ----------------------------------------------
# ✅ 1. Extract & clean text from ANY TEXT PDF
# ----------------------------------------------
def extract_text(pdf):
    text = ""
    with pdfplumber.open(pdf) as pdf_file:
        for page in pdf_file.pages:
            t = page.extract_text() or ""
            t = re.sub(r"\s+", " ", t)   # normalize spacing
            text += t + "\n\n"
    return text


# ----------------------------------------------
# ✅ 2. Load Fast Embeddings (Best for universal PDFs)
# ----------------------------------------------
@st.cache_resource
def load_embeddings():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embedder = load_embeddings()


# ----------------------------------------------
# ✅ 3. Load Fast LLM (Flan-T5-Base)
# ----------------------------------------------
@st.cache_resource
def load_llm():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return pipeline("text2text-generation", model=model, tokenizer=tokenizer)

llm = load_llm()


# ----------------------------------------------
# ✅ 4. Smart chunking (Adaptive for ANY text PDF)
# ----------------------------------------------
def chunk_text(text):
    chunks = []
    max_len = 1200

    for i in range(0, len(text), max_len):
        chunks.append(text[i:i+max_len])

    return chunks


# ----------------------------------------------
# ✅ 5. Build FAISS index
# ----------------------------------------------
def build_index(chunks):
    vectors = embedder.encode(chunks, convert_to_numpy=True)
    dim = vectors.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    return index, vectors


# =====================================================
# ✅ Chat Interface + PDF Upload
# =====================================================
if "history" not in st.session_state:
    st.session_state.history = []

uploaded_pdf = st.file_uploader("📄 Upload any text PDF", type="pdf")

if uploaded_pdf:
    st.success("✅ PDF Uploaded Successfully — Ready to Chat!")

    raw_text = extract_text(uploaded_pdf)
    chunks = chunk_text(raw_text)
    index, vectors = build_index(chunks)

    # -----------------------------
    # ✅ Chat UI
    # -----------------------------
    user_input = st.text_input("💬 Ask anything from the PDF:")

    if user_input:
        # Add user message to chat
        st.session_state.history.append(("user", user_input))

        q_vec = embedder.encode(user_input, convert_to_numpy=True)

        scores, results = index.search(q_vec.reshape(1, -1), 20)
        retrieved = [chunks[i] for i in results[0]]

        # ✅ Rerank using STS
        reranked = util.semantic_search(
            embedder.encode(user_input, convert_to_tensor=True),
            embedder.encode(retrieved, convert_to_tensor=True)
        )[0]

        # ✅ pick top 3 chunks
        top_chunks = [retrieved[item["corpus_id"]] for item in reranked[:3]]
        context = "\n\n".join(top_chunks)

        prompt = f"""
Answer only using this context.
If answer is not found, say "Not present in PDF".

CONTEXT:
{context}

QUESTION:
{user_input}

ANSWER:
"""

        answer = llm(prompt, max_length=300)[0]["generated_text"]

        # Add bot message
        st.session_state.history.append(("bot", answer))

    # -----------------------------
    # ✅ Display Chat
    # -----------------------------
    for sender, msg in st.session_state.history:
        if sender == "user":
            st.markdown(f"🧑 **You:** {msg}")
        else:
            st.markdown(f"🤖 **Bot:** {msg}")
