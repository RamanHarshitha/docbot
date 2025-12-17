import streamlit as st
import fitz  # PyMuPDF
import re
from PIL import Image
import pytesseract
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import random
import io
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import time

# ----------------------------
# Tesseract path for OCR
# ----------------------------
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# ----------------------------
# Streamlit page config
# ----------------------------
st.set_page_config(page_title="PDF Chatbot & MCQ Generator", page_icon="🤖", layout="wide")

# ----------------------------
# Simple user DB
# ----------------------------
if "USER_DB" not in st.session_state:
    st.session_state.USER_DB = {"admin": "admin", "student": "password"}

# ----------------------------
# PDF extraction with OCR
# ----------------------------
def extract_pdf(pdf_file):
    """Extract text from PDF with OCR fallback"""
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        slides = []
        for page_num, page in enumerate(doc):
            blocks = page.get_text("blocks")
            slide_text = ""
            for block in blocks:
                x0, y0, x1, y1, text, *_ = block
                if not text.strip():
                    rect = fitz.Rect(x0, y0, x1, y1)
                    pix = page.get_pixmap(clip=rect)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    text = pytesseract.image_to_string(img)
                text = re.sub(r"\s+", " ", text)
                if len(text.strip()) > 2:
                    slide_text += text + " "
            if len(slide_text.strip()) > 5:
                slides.append(slide_text.strip())
        return slides
    except Exception as e:
        st.error(f"Error extracting PDF: {str(e)}")
        return []

# ----------------------------
# Chunk text
# ----------------------------
def chunk_text(slides, max_chars=800):
    """Split text into manageable chunks"""
    chunks = []
    current = ""
    for s in slides:
        if len(current) + len(s) > max_chars:
            chunks.append(current.strip())
            current = s
        else:
            current += " " + s
    if current:
        chunks.append(current.strip())
    return chunks

# ----------------------------
# Load models
# ----------------------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def load_llm():
    model_name = "google/flan-t5-small"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return pipeline("text2text-generation", model=model, tokenizer=tok, max_length=512)

embed_model = load_embedder()
llm = load_llm()

# ----------------------------
# Build FAISS index for Q&A
# ----------------------------
def build_index(slides):
    """Build FAISS index from text slides"""
    try:
        if not slides:
            raise ValueError("No slides provided for indexing")
        
        vectors = embed_model.encode(slides, convert_to_numpy=True)
        
        if len(vectors) == 0:
            raise ValueError("No vectors generated from slides")
        
        dim = vectors.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(vectors)
        return index
    except Exception as e:
        st.error(f"Error building index: {str(e)}")
        return None

# ----------------------------
# IMPROVED Q&A FUNCTION WITH BETTER QUESTION TYPE DETECTION
# ----------------------------
def generate_answer(question, context):
    """Generate direct answers from context with improved question type detection"""
    
    question_lower = question.lower().strip()
    
    # Improved question type detection
    is_difference = any(word in question_lower for word in ['difference', 'different', 'compare', 'contrast', 'vs', 'versus', 'between'])
    is_definition = any(word in question_lower for word in ['what is', 'define', 'definition', 'meaning of', 'what does', 'explain what'])
    is_process = any(word in question_lower for word in ['how', 'process', 'steps', 'procedure', 'work', 'method'])
    is_list = any(word in question_lower for word in ['list', 'examples', 'types', 'categories', 'name the'])
    is_explanation = any(word in question_lower for word in ['why', 'reason', 'purpose', 'benefit', 'advantage', 'importance'])
    
    # Create specific prompts for each question type
    if is_difference:
        prompt = f"""
Based ONLY on the context below, explain the differences or comparisons mentioned.

Question: {question}

Context: {context[:1500]}

Provide clear differences in bullet points. Focus on contrasting features.
If no differences are found in context, say "No comparison information found in the document."

Answer in bullet points:
"""
    
    elif is_definition:
        prompt = f"""
Based ONLY on the context below, provide a clear definition.

Question: {question}

Context: {context[:1500]}

Provide a concise definition. Start with what it is and give key characteristics.
If not found, say "Definition not found in the document."

Definition:
"""
    
    elif is_process:
        prompt = f"""
Based ONLY on the context below, explain the process or steps.

Question: {question}

Context: {context[:1500]}

Explain step by step in a clear sequence. Use numbered steps if possible.
If not found, say "Process details not found in the document."

Steps:
"""
    
    elif is_list:
        prompt = f"""
Based ONLY on the context below, provide a list as requested.

Question: {question}

Context: {context[:1500]}

Provide a clear list with bullet points. Make each item distinct.
If not found, say "List information not found in the document."

List:
"""
    
    elif is_explanation:
        prompt = f"""
Based ONLY on the context below, explain the reasons or purposes.

Question: {question}

Context: {context[:1500]}

Explain the reasons, purposes or benefits clearly with logical points.
If not found, say "Explanation not found in the document."

Explanation:
"""
    
    else:
        prompt = f"""
Based ONLY on the context below, answer this question directly.

Question: {question}

Context: {context[:1500]}

Answer directly and factually. If the answer cannot be found, say "I cannot find this information in the document."

Answer:
"""
    
    try:
        response = llm(
            prompt,
            max_new_tokens=400,
            temperature=0.1,
            do_sample=False,
            repetition_penalty=1.1
        )[0]["generated_text"].strip()
        
        # Enhanced post-processing based on question type
        if is_difference:
            # Ensure bullet point format for differences
            if '•' not in response and '-' not in response and '1.' not in response:
                sentences = [s.strip() for s in response.split('. ') if len(s.strip()) > 20]
                if len(sentences) > 1:
                    response = "Key differences:\n\n• " + "\n• ".join(sentences)
                elif response:
                    response = "• " + response
        
        elif is_list:
            # Ensure bullet point format for lists
            if '•' not in response and '-' not in response and '1.' not in response:
                sentences = [s.strip() for s in response.split('. ') if len(s.strip()) > 10]
                if len(sentences) > 1:
                    response = "• " + "\n• ".join(sentences)
        
        elif is_process:
            # Ensure numbered format for processes
            if '1.' not in response and 'Step' not in response:
                sentences = [s.strip() for s in response.split('. ') if len(s.strip()) > 15]
                if len(sentences) > 1:
                    response = "Process:\n\n" + "\n".join([f"{i+1}. {sent}" for i, sent in enumerate(sentences)])
        
        # Validate response
        if not response or len(response) < 5:
            return "I don't have enough information to answer this question based on the document."
        
        return response
        
    except Exception as e:
        return f"I encountered an error while processing your question. Please try again."

# ----------------------------
# IMPROVED PDF SUMMARY FEATURE
# ----------------------------
def generate_pdf_summary(slides):
    """Generate a comprehensive summary of the PDF content"""
    
    # Combine all slides into one text
    full_text = " ".join(slides)
    
    # If text is too long, take first 3000 characters for summary
    if len(full_text) > 3000:
        summary_text = full_text[:3000]
    else:
        summary_text = full_text
    
    prompt = f"""
Create a comprehensive summary of the following document content:

{summary_text}

Provide a well-structured summary covering:
- Main topic and purpose
- Key points and concepts
- Important findings or conclusions
- Overall significance

Make it detailed but concise, around 200-300 words.
"""
    
    try:
        summary = llm(
            prompt,
            max_new_tokens=500,
            temperature=0.2,
            do_sample=False
        )[0]["generated_text"].strip()
        
        return summary if summary else "Unable to generate summary from the document content."
        
    except Exception as e:
        return f"Unable to generate summary due to error: {str(e)}"

# ----------------------------
# WORKING PDF SUMMARY DOWNLOAD
# ----------------------------
def create_summary_pdf(summary_text, title="Document Summary"):
    """Create a properly formatted PDF for the summary"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, 
                          rightMargin=72, leftMargin=72, 
                          topMargin=72, bottomMargin=72)
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=30,
        alignment=1,  # Center
        textColor=colors.darkblue,
        fontName='Helvetica-Bold'
    )
    
    summary_style = ParagraphStyle(
        'SummaryStyle',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=12,
        leading=14,  # Line height
        fontName='Helvetica'
    )
    
    footer_style = ParagraphStyle(
        'FooterStyle',
        parent=styles['Normal'],
        fontSize=9,
        spaceBefore=20,
        textColor=colors.gray,
        fontName='Helvetica'
    )
    
    story = []
    
    # Title
    story.append(Paragraph(title, title_style))
    story.append(Spacer(1, 20))
    
    # Summary content - split into paragraphs for better formatting
    summary_paragraphs = summary_text.split('\n')
    for para in summary_paragraphs:
        if para.strip():
            story.append(Paragraph(para.strip(), summary_style))
            story.append(Spacer(1, 8))
    
    # Footer
    story.append(Spacer(1, 20))
    story.append(Paragraph(f"Generated on: {time.strftime('%Y-%m-%d at %H:%M:%S')}", footer_style))
    story.append(Paragraph("Generated by PDF Assistant", footer_style))
    story.append(Paragraph(f"Summary length: {len(summary_text)} characters", footer_style))
    
    try:
        doc.build(story)
        buffer.seek(0)
        return buffer
    except Exception as e:
        st.error(f"Error creating PDF: {str(e)}")
        return None

# ----------------------------
# IMPROVED MCQ GENERATOR
# ----------------------------
def generate_mcqs_improved(text_chunk, num_questions=2):
    """Generate MCQs with better prompting and parsing"""
    
    clean_text = re.sub(r'\s+', ' ', text_chunk)[:400]
    
    prompt = f"""
Based EXACTLY on this text content, create {num_questions} multiple choice questions:

TEXT: {clean_text}

For EACH question, follow this EXACT format:
QUESTION: [clear question based on text]
A) [plausible option A]
B) [plausible option B] 
C) [plausible option C]
D) [plausible option D]
ANSWER: [single letter A-D]

Make sure:
- Questions test understanding of the text
- Only ONE correct answer per question
- Wrong options are plausible but incorrect
- Questions are diverse and cover different aspects
"""
    
    try:
        response = llm(
            prompt, 
            max_new_tokens=500,
            do_sample=True,
            temperature=0.8,
            num_return_sequences=1
        )[0]["generated_text"]
        
        return response
    except Exception as e:
        return f"Error: {str(e)}"

def parse_mcqs_robust(response):
    """More robust MCQ parsing"""
    mcqs = []
    
    # Split by various question indicators
    blocks = re.split(r'(?:QUESTION:|Q\d*[:.]|\nQUESTION:|\nQ\d*[:.]|\n\d+[\.)])', response)
    
    for block in blocks:
        if not block.strip():
            continue
            
        lines = [line.strip() for line in block.split('\n') if line.strip()]
        
        question = ""
        options = {}
        correct_answer = ""
        
        for line in lines:
            # Extract question
            if re.match(r'^(?:QUESTION:|Q\d*[:.]?)\s*(.+)', line) and not question:
                match = re.match(r'^(?:QUESTION:|Q\d*[:.]?)\s*(.+)', line)
                question = match.group(1).strip()
            # Extract options
            elif re.match(r'^[A-D][).]?\s*(.+)', line):
                match = re.match(r'^([A-D])[).]?\s*(.+)', line)
                if match:
                    opt_key = match.group(1)
                    opt_text = match.group(2).strip()
                    # Remove any answer indicators from options
                    opt_text = re.sub(r'\(correct\)|\(answer\)', '', opt_text, flags=re.IGNORECASE).strip()
                    options[opt_key] = opt_text
            # Extract answer
            elif re.match(r'^(?:ANSWER:|CORRECT:)\s*([A-D])', line, re.IGNORECASE):
                match = re.match(r'^(?:ANSWER:|CORRECT:)\s*([A-D])', line, re.IGNORECASE)
                correct_answer = match.group(1)
            # If no explicit question marker, first substantial line is question
            elif not question and len(line) > 15 and not re.match(r'^[A-D][).]', line):
                question = line
        
        # Validate we have a complete MCQ
        if (question and len(question) > 10 and 
            len(options) >= 3 and  # At least 3 options
            correct_answer and correct_answer in options):
            
            # Ensure we have exactly 4 options A-D
            full_options = {}
            for letter in ['A', 'B', 'C', 'D']:
                if letter in options:
                    full_options[letter] = options[letter]
                else:
                    # Create plausible filler option
                    filler_options = {
                        'A': 'None of the above',
                        'B': 'All of the above', 
                        'C': 'Both A and B',
                        'D': 'Cannot be determined from the text'
                    }
                    full_options[letter] = filler_options.get(letter, f"Option {letter}")
            
            mcqs.append({
                'question': question,
                'options': full_options,
                'answer': correct_answer
            })
    
    return mcqs

# ----------------------------
# MANUAL MCQ GENERATOR (Fallback)
# ----------------------------
def create_manual_mcqs(text_chunk, num_questions=2):
    """Create MCQs manually from text content when LLM fails"""
    sentences = [s.strip() for s in re.split(r'[.!?]', text_chunk) if len(s.strip()) > 20]
    
    if len(sentences) < 2:
        return []
    
    mcqs = []
    
    for i in range(min(num_questions, len(sentences))):
        sentence = sentences[i]
        words = sentence.split()
        
        if len(words) > 6:
            # Create fill-in-the-blank question
            blank_idx = random.randint(2, len(words)-2)
            key_word = words[blank_idx]
            question_text = " ".join(words[:blank_idx] + ["______"] + words[blank_idx+1:])
            question = f"Complete the sentence: {question_text}"
            
            # Create options
            correct_option = key_word
            wrong_options = [
                random.choice(['process', 'method', 'technique', 'approach']),
                random.choice(['important', 'essential', 'critical', 'vital']),
                random.choice(['system', 'framework', 'structure', 'model'])
            ]
            random.shuffle(wrong_options)
            
            options = {
                'A': correct_option,
                'B': wrong_options[0],
                'C': wrong_options[1],
                'D': wrong_options[2]
            }
            
            # Shuffle options but track correct answer
            option_keys = ['A', 'B', 'C', 'D']
            random.shuffle(option_keys)
            shuffled_options = {}
            correct_key = None
            
            for j, key in enumerate(option_keys):
                shuffled_options[key] = options[list(options.keys())[j]]
                if list(options.keys())[j] == 'A':  # 'A' was the correct answer
                    correct_key = key
            
            mcqs.append({
                'question': question,
                'options': shuffled_options,
                'answer': correct_key
            })
    
    return mcqs

# ----------------------------
# PDF DOWNLOAD FUNCTIONALITY FOR MCQs
# ----------------------------
def create_mcq_pdf(mcqs, include_answers=True, title="Generated Multiple Choice Questions"):
    """Create a professional PDF with MCQs"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, 
                          rightMargin=72, leftMargin=72, 
                          topMargin=72, bottomMargin=72)
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=30,
        alignment=1,  # Center
        textColor=colors.darkblue,
        fontName='Helvetica-Bold'
    )
    
    question_style = ParagraphStyle(
        'QuestionStyle',
        parent=styles['Normal'],
        fontSize=12,
        spaceAfter=12,
        leftIndent=0,
        fontName='Helvetica-Bold'
    )
    
    option_style = ParagraphStyle(
        'OptionStyle',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=6,
        leftIndent=20,
        fontName='Helvetica'
    )
    
    answer_style = ParagraphStyle(
        'AnswerStyle',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=20,
        leftIndent=20,
        textColor=colors.green,
        fontName='Helvetica-Bold'
    )
    
    story = []
    
    # Title
    story.append(Paragraph(title, title_style))
    story.append(Spacer(1, 20))
    
    # Add MCQs
    for i, mcq in enumerate(mcqs, 1):
        # Question
        story.append(Paragraph(f"Q{i}. {mcq['question']}", question_style))
        
        # Options
        for letter in ['A', 'B', 'C', 'D']:
            option_text = mcq['options'].get(letter, '')
            story.append(Paragraph(f"{letter}. {option_text}", option_style))
        
        # Answer
        if include_answers:
            correct_text = mcq['options'].get(mcq['answer'], '')
            story.append(Paragraph(f"<b>Answer: {mcq['answer']}. {correct_text}</b>", answer_style))
        else:
            story.append(Paragraph("Answer: ___________________", option_style))
        
        story.append(Spacer(1, 15))
        
        # Page break after every 5 questions
        if i % 5 == 0 and i < len(mcqs):
            story.append(PageBreak())
    
    # Footer
    story.append(Spacer(1, 20))
    story.append(Paragraph(f"Total Questions: {len(mcqs)}", styles['Normal']))
    story.append(Paragraph(f"Generated on: {time.strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
    story.append(Paragraph("Generated by PDF MCQ Generator", styles['Normal']))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

# ----------------------------
# Login Page
# ----------------------------
def login_page():
    st.title("🔐 Login")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", placeholder="Enter your password")
        
        if st.button("🚀 Login", use_container_width=True):
            if username and password:
                if username in st.session_state.USER_DB and st.session_state.USER_DB[username] == password:
                    st.session_state.logged_in = True
                    st.session_state.user = username
                    st.session_state.chat = []
                    st.success("✅ Login successful!")
                    st.rerun()
                else:
                    st.error("❌ Incorrect username or password")
            else:
                st.warning("⚠️ Please enter both username and password")
    
    with col2:
        st.info("""
        **Demo Accounts:**
        - Username: `admin` | Password: `admin`
        - Username: `student` | Password: `password`
        """)
        
        if st.button("✨ Create New Account", use_container_width=True):
            st.session_state.page = "signup"
            st.rerun()

# ----------------------------
# Signup Page
# ----------------------------
def signup_page():
    st.title("📝 Create New Account")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        username = st.text_input("Choose username", placeholder="Enter unique username")
        password = st.text_input("Choose password", type="password", placeholder="Create strong password")
        confirm_password = st.text_input("Confirm password", type="password", placeholder="Re-enter password")
        
        if st.button("✅ Create Account", use_container_width=True):
            if not username or not password:
                st.error("❌ Please fill all fields")
            elif username in st.session_state.USER_DB:
                st.error("❌ Username already exists")
            elif password != confirm_password:
                st.error("❌ Passwords do not match")
            else:
                st.session_state.USER_DB[username] = password
                st.success("✅ Account created successfully!")
                st.session_state.page = "login"
                st.rerun()
    
    with col2:
        st.info("""
        **Account Requirements:**
        - Unique username
        - Secure password
        - Passwords must match
        """)
        
        if st.button("⬅ Back to Login", use_container_width=True):
            st.session_state.page = "login"
            st.rerun()

# ----------------------------
# IMPROVED SUMMARY PAGE
# ----------------------------
def summary_page():
    st.title("📄 PDF Summary Generator")
    st.markdown("Upload a PDF to generate a comprehensive summary of its content")
    
    pdf = st.file_uploader("📄 Upload PDF for Summary", type="pdf", key="summary_uploader")
    
    if pdf:
        with st.spinner("📖 Extracting text from PDF..."):
            slides = extract_pdf(pdf)
        
        if slides:
            st.success(f"✅ PDF processed! Found {len(slides)} text sections.")
            
            # Display PDF info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Text Sections", len(slides))
            with col2:
                total_chars = sum(len(slide) for slide in slides)
                st.metric("Total Characters", f"{total_chars:,}")
            with col3:
                st.metric("Content Density", f"{total_chars//len(slides):,} chars/section")
            
            if st.button("📝 Generate Summary", type="primary", use_container_width=True):
                with st.spinner("🤔 Generating comprehensive summary..."):
                    summary = generate_pdf_summary(slides)
                
                if summary and not summary.startswith("Unable to generate"):
                    st.success("✅ Summary generated successfully!")
                    
                    # Display summary in an expandable section
                    with st.expander("📋 View Document Summary", expanded=True):
                        st.markdown("#### Document Overview")
                        st.markdown("---")
                        st.write(summary)
                    
                    # Download options
                    st.subheader("📥 Download Summary")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Text download
                        st.download_button(
                            label="📝 Download as Text",
                            data=summary,
                            file_name="document_summary.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
                    
                    with col2:
                        # PDF download - FIXED
                        pdf_buffer = create_summary_pdf(summary)
                        if pdf_buffer:
                            st.download_button(
                                label="📄 Download as PDF",
                                data=pdf_buffer,
                                file_name="document_summary.pdf",
                                mime="application/pdf",
                                use_container_width=True
                            )
                        else:
                            st.error("Could not generate PDF download")
                    
                else:
                    st.error("❌ Could not generate summary. Please try a different PDF.")
        else:
            st.error("❌ No text could be extracted from the PDF. Please try a different file.")

# ----------------------------
# MCQ PAGE
# ----------------------------
def mcq_page():
    st.title("📝 MCQ Generator")
    st.markdown("Upload a PDF to generate multiple choice questions from its content")
    
    pdf = st.file_uploader("📄 Upload PDF for MCQs", type="pdf", key="mcq_uploader")
    
    if pdf:
        with st.spinner("📖 Extracting text from PDF..."):
            slides = extract_pdf(pdf)
        
        if slides:
            st.success(f"✅ PDF processed! Found {len(slides)} text sections.")
            
            # Display PDF info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Text Sections", len(slides))
            with col2:
                total_chars = sum(len(slide) for slide in slides)
                st.metric("Total Characters", f"{total_chars:,}")
            with col3:
                chunks = chunk_text(slides)
                st.metric("Content Chunks", len(chunks))
            
            # Configuration
            with st.expander("⚙️ Generation Settings"):
                col1, col2 = st.columns(2)
                with col1:
                    total_questions = st.slider("Questions to generate", 5, 20, 10)
                    use_manual_fallback = st.checkbox("Enhanced generation", value=True)
                with col2:
                    max_chunk_size = st.slider("Max chunk size", 500, 1500, 800)
                    questions_per_chunk = st.slider("Questions per chunk", 1, 3, 2)
            
            if st.button("🎯 Generate MCQs", type="primary", use_container_width=True):
                chunks = chunk_text(slides, max_chars=max_chunk_size)
                all_mcqs = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, chunk in enumerate(chunks):
                    if len(all_mcqs) >= total_questions:
                        break
                    
                    status_text.text(f"📝 Processing chunk {i+1}/{len(chunks)}...")
                    
                    # Try LLM generation
                    response = generate_mcqs_improved(chunk, num_questions=questions_per_chunk)
                    mcqs = parse_mcqs_robust(response)
                    
                    # Fallback if needed
                    if len(mcqs) < questions_per_chunk and use_manual_fallback:
                        extra_mcqs = create_manual_mcqs(chunk, num_questions=questions_per_chunk - len(mcqs))
                        mcqs.extend(extra_mcqs)
                    
                    # Add unique MCQs
                    for mcq in mcqs:
                        if (mcq['question'] not in [m['question'] for m in all_mcqs] and 
                            len(all_mcqs) < total_questions):
                            all_mcqs.append(mcq)
                    
                    progress_bar.progress(min((i + 1) / len(chunks), 1.0))
                
                status_text.text("✅ Generation complete!")
                st.session_state.generated_mcqs = all_mcqs
                
                # Display results
                if all_mcqs:
                    st.subheader(f"📋 Generated MCQs ({len(all_mcqs)} questions)")
                    
                    # Download section
                    st.subheader("📥 Download Options")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        pdf_with_answers = create_mcq_pdf(all_mcqs, include_answers=True)
                        st.download_button(
                            label="📄 PDF (With Answers)",
                            data=pdf_with_answers,
                            file_name="mcq_with_answers.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                    
                    with col2:
                        pdf_without_answers = create_mcq_pdf(all_mcqs, include_answers=False)
                        st.download_button(
                            label="📝 PDF (Practice)",
                            data=pdf_without_answers,
                            file_name="mcq_practice.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                    
                    with col3:
                        # Also provide text version
                        text_content = "Generated Multiple Choice Questions\n\n"
                        for i, mcq in enumerate(all_mcqs, 1):
                            text_content += f"Q{i}. {mcq['question']}\n"
                            for letter in ['A', 'B', 'C', 'D']:
                                text_content += f"  {letter}. {mcq['options'][letter]}\n"
                            text_content += f"  Answer: {mcq['answer']}\n\n"
                        
                        st.download_button(
                            label="📝 Text File",
                            data=text_content,
                            file_name="mcq_questions.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
                    
                    st.markdown("---")
                    
                    # Display questions
                    for i, mcq in enumerate(all_mcqs, 1):
                        with st.container():
                            st.markdown(f"**Q{i}. {mcq['question']}**")
                            
                            # Options in two columns
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**A.** {mcq['options']['A']}")
                                st.write(f"**C.** {mcq['options']['C']}")
                            with col2:
                                st.write(f"**B.** {mcq['options']['B']}")
                                st.write(f"**D.** {mcq['options']['D']}")
                            
                            # Answer with expander
                            with st.expander("🎯 Show Answer & Explanation", expanded=False):
                                st.success(f"**Correct Answer: {mcq['answer']}**")
                                st.info(f"**Explanation:** {mcq['options'][mcq['answer']]}")
                            
                            st.markdown("---")
                else:
                    st.error("❌ No MCQs could be generated. Please try a different PDF.")
        else:
            st.error("❌ No text could be extracted from the PDF. Please try a different file.")

# ----------------------------
# CHAT UI
# ----------------------------
def chat_ui():
    st.sidebar.title(f"👋 Welcome, {st.session_state.user}")
    choice = st.sidebar.radio("Navigation", ["Chat", "MCQs", "Summary", "History", "Logout"])

    if choice == "Logout":
        st.session_state.logged_in = False
        st.session_state.chat = []
        st.session_state.current_index = None
        st.session_state.current_slides = None
        st.success("✅ Logged out successfully!")
        st.rerun()

    if choice == "History":
        st.title("📜 Chat History")
        if not st.session_state.chat:
            st.info("💬 No conversation history yet!")
        else:
            for sender, msg in st.session_state.chat:
                if sender == "Bot":
                    st.markdown(f"**🤖 Assistant:** {msg}")
                else:
                    st.markdown(f"**👤 You:** {msg}")
        return

    if choice == "MCQs":
        mcq_page()
        return
        
    if choice == "Summary":
        summary_page()
        return

    if choice == "Chat":
        st.title("🤖 PDF Assistant")
        st.markdown("Upload a PDF and ask questions about its content")
        
        pdf = st.file_uploader("📄 Upload PDF (Text, Slides, Scanned)", type="pdf", key="chat_uploader")
        
        if pdf:
            with st.spinner("🔍 Processing PDF content..."):
                slides = extract_pdf(pdf)
                if slides:
                    index = build_index(slides)
                    if index is not None:
                        st.session_state.current_index = index
                        st.session_state.current_slides = slides
                        st.success("✅ PDF processed successfully! You can now ask questions.")
                    else:
                        st.error("❌ Failed to build search index. Please try another PDF.")
                        return
                else:
                    st.error("❌ Could not extract text from PDF. Please try a different file.")
                    return

            # Initialize chat
            if "chat" not in st.session_state:
                st.session_state.chat = []

            # Display chat history
            for sender, msg in st.session_state.chat:
                if sender == "Bot":
                    with st.chat_message("assistant"):
                        st.markdown(msg)
                else:
                    with st.chat_message("user"):
                        st.markdown(msg)

            # Chat input
            user_msg = st.chat_input("💭 Ask something from the PDF...")
            if user_msg:
                # Add user message
                st.session_state.chat.append((st.session_state.user, user_msg))
                with st.chat_message("user"):
                    st.markdown(user_msg)

                # Search for relevant context
                with st.spinner("🔍 Searching for relevant information..."):
                    try:
                        qvec = embed_model.encode(user_msg, convert_to_numpy=True)
                        scores, res = st.session_state.current_index.search(qvec.reshape(1, -1), 3)
                        matched = [st.session_state.current_slides[i] for i in res[0] if i < len(st.session_state.current_slides)]
                        context = " ".join(matched) if matched else ""
                    except Exception as e:
                        st.error(f"Search error: {str(e)}")
                        context = ""

                # Generate answer using the IMPROVED function
                with st.spinner("🤔 Generating answer..."):
                    if context:
                        ans = generate_answer(user_msg, context)
                    else:
                        ans = "❌ I couldn't find relevant information in the PDF to answer your question. Please try rephrasing or ask about a different topic."
                
                # Add bot response
                st.session_state.chat.append(("Bot", ans))
                with st.chat_message("assistant"):
                    st.markdown(ans)

# ----------------------------
# Main App
# ----------------------------
def main():
    # Initialize session state
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "page" not in st.session_state:
        st.session_state.page = "login"
    if "chat" not in st.session_state:
        st.session_state.chat = []
    if "generated_mcqs" not in st.session_state:
        st.session_state.generated_mcqs = []
    if "current_index" not in st.session_state:
        st.session_state.current_index = None
    if "current_slides" not in st.session_state:
        st.session_state.current_slides = None

    if not st.session_state.logged_in:
        if st.session_state.page == "login":
            login_page()
        else:
            signup_page()
    else:
        chat_ui()

if __name__ == "__main__":
    main()