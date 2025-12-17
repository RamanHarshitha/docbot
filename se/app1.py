import streamlit as st
import fitz  # PyMuPDF
import re
from PIL import Image
import pytesseract
from sentence_transformers import SentenceTransformer, util
import faiss
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# ✅ Set correct Tesseract path
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# ✅ Page settings
st.set_page_config(page_title="Universal PDF Chatbot", page_icon="🤖", layout="wide")

# ---------------------------------------------------------
# ✅ User Database (Simple for now)
# ---------------------------------------------------------
if "USER_DB" not in st.session_state:
    st.session_state.USER_DB = {
        "admin": "admin",
        "student": "password",
    }

# ---------------------------------------------------------
# ✅ Unified PDF Extractor (Slides + OCR + Text)
# ---------------------------------------------------------
def extract_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    slides = []

    for page in doc:
        blocks = page.get_text("blocks")
        slide_text = ""

        for block in blocks:
            x0, y0, x1, y1, text, *_ = block

            # If empty → OCR the block
            if not text.strip():
                rect = fitz.Rect(x0, y0, x1, y1)
                pix = page.get_pixmap(clip=rect)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                text = pytesseract.image_to_string(img)

            # Clean unwanted text
            if any(k in text for k in ["RGUKT", "CS3102", "Professor", "Mallikarjuna"]):
                continue

            text = re.sub(r"\s+", " ", text)

            if len(text.strip()) > 2:
                slide_text += text + " "

        if len(slide_text.strip()) > 5:
            slides.append(slide_text.strip())

    return slides

# ---------------------------------------------------------
# ✅ Embedding Model (Fast + Accurate)
# ---------------------------------------------------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embed_model = load_embedder()

# ---------------------------------------------------------
# ✅ LLM Model (Fast)
# ---------------------------------------------------------
@st.cache_resource
def load_llm():
    model_name = "google/flan-t5-small"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return pipeline("text2text-generation", model=model, tokenizer=tok)

llm = load_llm()

# ---------------------------------------------------------
# ✅ Build Vector Index
# ---------------------------------------------------------
def build_index(slides):
    vectors = embed_model.encode(slides, convert_to_numpy=True)
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    return index

# ---------------------------------------------------------
# ✅ Login Page
# ---------------------------------------------------------
def login_page():
    st.title("🔐 Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login ✅"):
        if username in st.session_state.USER_DB and st.session_state.USER_DB[username] == password:
            st.session_state.logged_in = True
            st.session_state.user = username
            st.session_state.chat = []
            st.success("✅ Login successful!")
            st.rerun()
        else:
            st.error("❌ Incorrect username or password")

    if st.button("Create New Account ✨"):
        st.session_state.page = "signup"
        st.rerun()

# ---------------------------------------------------------
# ✅ Signup Page
# ---------------------------------------------------------
def signup_page():
    st.title("📝 Create New Account")

    username = st.text_input("Choose username")
    password = st.text_input("Choose password", type="password")

    if st.button("Signup ✅"):
        if username in st.session_state.USER_DB:
            st.error("❌ Username already exists")
        else:
            st.session_state.USER_DB[username] = password
            st.success("✅ Account created successfully!")
            st.session_state.page = "login"
            st.rerun()

    if st.button("⬅ Back to Login"):
        st.session_state.page = "login"
        st.rerun()

# ---------------------------------------------------------
# ✅ MAIN CHAT UI
# ---------------------------------------------------------
# ---------------------------------------------------------
# ✅ MAIN CHAT + MCQ UI
# ---------------------------------------------------------
def chat_ui():
    st.sidebar.title(f"👋 Welcome, {st.session_state.user}")
    choice = st.sidebar.radio("Navigation", ["Chat", "MCQs", "History", "Logout"])

    if choice == "Logout":
        st.session_state.logged_in = False
        st.session_state.chat = []
        st.success("✅ Logged out!")
        st.rerun()

    if choice == "History":
        st.title("📜 Chat History")
        if not st.session_state.chat:
            st.info("No history yet!")
        else:
            for sender, msg in st.session_state.chat:
                st.write(f"**{sender}:** {msg}")
        return

    # ------------------------------------------------------
    # ✅ MCQ Generator
    # ------------------------------------------------------
    if choice == "MCQs":
        st.title("📝 MCQ Generator")
        pdf = st.file_uploader("📄 Upload PDF for MCQs", type="pdf")

        if pdf:
            slides = extract_pdf(pdf)
            st.success("✅ PDF processed. Ready to generate MCQs!")

            if st.button("Generate MCQs"):
                # Take first few pages for context
                context = " ".join(slides[:4])
                prompt = f"""
Generate 5 multiple-choice questions from this context.
Each MCQ must have:
- Question
- 4 options (A, B, C, D)
- Correct answer clearly labeled

Context:
{context}

MCQs:
"""
                raw_mcqs = llm(prompt, max_length=500)[0]["generated_text"]

                # Split questions by numbering
                mcq_list = re.split(r'\n\d+\.', raw_mcqs)
                mcq_list = [m.strip() for m in mcq_list if m.strip()]

                st.write("### Generated MCQs")
                for i, mcq in enumerate(mcq_list, 1):
                    lines = mcq.split("\n")
                    question = lines[0]
                    options = [l for l in lines[1:] if re.match(r'[A-D]\s*[\.:]', l)]
                    answer_line = [l for l in lines if "Answer" in l or "Correct" in l]
                    answer = answer_line[0] if answer_line else "Not specified"

                    st.markdown(f"**Q{i}. {question}**")
                    cols = st.columns(2)
                    for j, opt in enumerate(options):
                        cols[j % 2].markdown(f"- {opt}")
                    st.markdown(f"**Answer:** {answer}")
                    st.markdown("---")
        return

    # ------------------------------------------------------
    # ✅ Chatbot Interface
    # ------------------------------------------------------
    if choice == "Chat":
        st.title("🤖 PDF Assistant")
        pdf = st.file_uploader("📄 Upload PDF (Text, Slides, Scanned)", type="pdf")

        if pdf:
            slides = extract_pdf(pdf)
            index = build_index(slides)
            st.success("✅ PDF processed successfully!")

            user_msg = st.chat_input("Ask something from the PDF...")
            if user_msg:
                st.session_state.chat.append((st.session_state.user, user_msg))

                qvec = embed_model.encode(user_msg, convert_to_numpy=True)
                scores, res = index.search(qvec.reshape(1, -1), 5)
                matched = [slides[i] for i in res[0]]
                context = " ".join(matched)

                prompt = f"""
Use ONLY this context:
{context}

Question: {user_msg}
Answer:
"""
                ans = llm(prompt, max_length=200)[0]["generated_text"]
                st.session_state.chat.append(("Bot", ans))

            for sender, msg in st.session_state.chat:
                st.chat_message("user" if sender != "Bot" else "assistant").markdown(msg)

# ---------------------------------------------------------
# ✅ ROUTER
# ---------------------------------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "page" not in st.session_state:
    st.session_state.page = "login"

if not st.session_state.logged_in:
    if st.session_state.page == "login":
        login_page()
    else:
        signup_page()
else:
    chat_ui()
