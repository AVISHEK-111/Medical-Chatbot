# app.py (Streamlit version)
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import requests

# Set page config
st.set_page_config(
    page_title="Medical Textbook Chatbot",
    page_icon="🩺",
    layout="wide"
)

# Load environment variables
load_dotenv()

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'db_loaded' not in st.session_state:
    st.session_state.db_loaded = False
if 'db' not in st.session_state:
    st.session_state.db = None

# Sidebar for configuration
with st.sidebar:
    st.title("🩺 Medical Textbook Chatbot")
    st.markdown("---")
    
    st.subheader("About")
    st.markdown("""
    This chatbot answers medical questions using a 637-page medical textbook.
    
    Features:
    - Search 4201 text sections
    - AI-generated answers
    - Source references with page numbers
    """)
    
    st.markdown("---")
    
    # Database status
    st.subheader("Database Status")
    
    DB_FAISS_PATH = "vectorstore/db_faiss"
    if os.path.exists(DB_FAISS_PATH):
        st.success("✓ Medical textbook database found")
        
        # Load database if not already loaded
        if not st.session_state.db_loaded:
            with st.spinner("Loading medical knowledge..."):
                try:
                    embedding_model = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2"
                    )
                    
                    st.session_state.db = FAISS.load_local(
                        DB_FAISS_PATH, 
                        embedding_model, 
                        allow_dangerous_deserialization=True
                    )
                    st.session_state.db_loaded = True
                    st.success("Database loaded successfully!")
                except Exception as e:
                    st.error(f"Error loading database: {e}")
    else:
        st.error("Database not found. Run process_pdfs.py first.")
    
    st.markdown("---")
    
    # Clear chat button
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# Main app
st.title("📚 Medical Textbook Chatbot")
st.markdown("Ask questions about medical topics from a 637-page textbook")

# Check if database is loaded
if not st.session_state.db_loaded:
    st.warning("⚠️ Medical database not loaded. Please check the sidebar.")
    st.stop()

# Chat interface
col1, col2 = st.columns([3, 1])

with col1:
    question = st.text_input(
        "Ask a medical question:",
        placeholder="e.g., What is diabetes? Explain cardiovascular disease...",
        key="question_input"
    )

with col2:
    if st.button("Ask", type="primary", use_container_width=True):
        if question.strip():
            # Add to chat history
            st.session_state.chat_history.append({
                "role": "user",
                "content": question
            })

# Function to get LLM response
def get_llm_response(prompt):
    """Get response from LLM"""
    HF_TOKEN = os.environ.get("HF_TOKEN")
    
    if not HF_TOKEN:
        return "API token not configured"
    
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 200,
            "temperature": 0.3,
            "do_sample": False,
            "top_p": 0.9,
            "return_full_text": False
        }
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                answer = result[0].get('generated_text', '').strip()
                # Clean the answer
                for prefix in ["ANSWER:", "Answer:", "RESPONSE:", "Response:"]:
                    if answer.startswith(prefix):
                        answer = answer[len(prefix):].strip()
                return answer
    except Exception as e:
        return f"Error generating answer: {str(e)}"
    
    return None

# Function to generate medical answer
def generate_medical_answer(question, textbook_content, page_num):
    """Generate a medical answer using LLM"""
    
    prompt = f"""MEDICAL QUESTION: {question}

TEXTBOOK REFERENCE (Page {page_num}):
{textbook_content}

INSTRUCTIONS:
Provide a clear, complete medical answer based ONLY on the textbook.
Start with a definition and include key information.
Use professional medical language.

MEDICAL ANSWER:"""
    
    answer = get_llm_response(prompt)
    
    if not answer:
        # Fallback: create a summary from the textbook
        sentences = [s.strip() for s in textbook_content.split('.') if s.strip()]
        if sentences:
            answer = '. '.join(sentences[:2]) + '.'
        else:
            answer = textbook_content[:200] + "..."
    
    return answer

# Process the question if asked
if question.strip() and st.session_state.chat_history and st.session_state.chat_history[-1]["content"] == question:
    with st.spinner("Searching textbook and generating answer..."):
        try:
            # Search the database
            docs = st.session_state.db.similarity_search(question, k=1)
            
            if docs:
                doc = docs[0]
                textbook_content = doc.page_content
                page_num = doc.metadata.get('page', 'Unknown')
                
                # Generate answer
                answer = generate_medical_answer(question, textbook_content, page_num)
                
                # Add to chat history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "source": textbook_content,
                    "page": page_num
                })
            else:
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": "I couldn't find information on this topic in the medical textbook.",
                    "source": None,
                    "page": None
                })
                
        except Exception as e:
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"Error: {str(e)}",
                "source": None,
                "page": None
            })

# Display chat history
for chat in st.session_state.chat_history:
    if chat["role"] == "user":
        with st.chat_message("user"):
            st.markdown(f"**You:** {chat['content']}")
    else:
        with st.chat_message("assistant"):
            st.markdown(f"**Assistant:** {chat['content']}")
            
            # Show source if available
            if "source" in chat and chat["source"]:
                with st.expander(f"📖 Source (Page {chat['page']})"):
                    source_text = chat["source"]
                    if len(source_text) > 300:
                        source_text = source_text[:300] + "..."
                    st.write(source_text)

# Example questions
st.markdown("---")
st.subheader("💡 Example Questions")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("What is diabetes?"):
        st.session_state.question_input = "What is diabetes?"

with col2:
    if st.button("Explain heart disease"):
        st.session_state.question_input = "Explain heart disease"

with col3:
    if st.button("Symptoms of asthma"):
        st.session_state.question_input = "What are the symptoms of asthma?"

# Footer
st.markdown("---")
st.caption("Medical Textbook Chatbot • ")