# --- Force Hugging Face Transformers to be offline ---
import os
os.environ['HF_HUB_OFFLINE'] = '1'

# --- API & UTILITY IMPORTS ---
import uvicorn
import tempfile
import hashlib
import shutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List
import torch # Need torch to check for CUDA availability

# --- TTS IMPORT ---
from TTS.api import TTS

# --- LANGCHAIN AND AI LIBRARIES (MODERN IMPORTS) ---
from langchain_community.document_loaders import Docx2txtLoader
from langchain_core.documents import Document # Needed for creating docs from OCR text
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM
from langchain_core.vectorstores import VectorStore

# --- OCR SPECIFIC IMPORTS ---
import pytesseract
from pdf2image import convert_from_path
from PIL import Image

# --- TESSERACT CONFIGURATION ---
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# ===================================================================
# GLOBAL CONFIGURATION
# ===================================================================

CHROMA_DB_PATH = "./chroma_db_folder"
KNOWLEDGE_BASE_PATH = r"C:\Users\vijay\OneDrive\Desktop\RAG-Base" # Use your actual path
AUDIO_DIR = "./static_audio" # Folder to serve generated audio

os.makedirs(AUDIO_DIR, exist_ok=True)

# ===================================================================
# IN-MEMORY CACHE FOR TEMPORARY VECTOR STORES
# ===================================================================
temp_vector_stores_cache = {}
# Optional: Limit cache size to avoid excessive memory use
MAX_CACHE_SIZE = 10 # Store up to 10 temporary vector stores

# ===================================================================
# LOAD MODELS ON STARTUP
# ===================================================================

print("Loading TTS model... (This may take a moment)")
try:
    # Use recommended device placement if possible
    tts_model = TTS("tts_models/en/ljspeech/tacotron2-DDC")
    if torch.cuda.is_available():
        tts_model.to("cuda")
    print("TTS model loaded successfully.")
except Exception as e:
    print(f"Error loading TTS model: {e}. TTS functionality will be disabled.")
    tts_model = None

print("Loading embedding model (all-MiniLM-L6-v2) to GPU...")
try:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    model_kwargs = {'device': device}
    encode_kwargs = {'normalize_embeddings': False} # Usually recommended for Chroma
    embeddings_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    print("Embedding model loaded successfully.")
except Exception as e:
    print(f"CRITICAL: Failed to load embedding model: {e}")
    exit()

print("Loading Ollama LLM (phi3)...")
try:
    llm = OllamaLLM(model="phi3")
    llm.invoke("hello") # Test connection
    print("Ollama LLM (phi3) connected successfully.")
except Exception as e:
    print(f"CRITICAL: Failed to connect to Ollama: {e}")
    print("Please ensure the Ollama server is running and 'phi3' is installed.")
    exit()

# ===================================================================
# REFACTORED CORE FUNCTIONS
# ===================================================================

def generate_audio_file_url(text: str, base_url: str) -> Optional[str]:
    """Generates audio, saves it, returns URL."""
    if tts_model is None:
        print("Skipping TTS: model not loaded.")
        return None
    try:
        text_hash = hashlib.md5(text.encode()).hexdigest()
        filename = f"{text_hash}.wav"
        file_path = os.path.join(AUDIO_DIR, filename)
        if not os.path.exists(file_path):
            print(f"Generating audio: {filename}")
            # Ensure text is not empty
            if text and text.strip():
                 tts_model.tts_to_file(text=text, file_path=file_path)
            else:
                 print("Warning: Attempted TTS on empty text.")
                 return None # Cannot generate audio for empty text
        return f"{base_url}audio/{filename}"
    except Exception as e:
        print(f"Error generating audio: {e}")
        return None

def extract_text_with_ocr(file_path: str, file_name: str) -> List[Document]:
    """Extracts text using OCR for PDF, or Docx2txtLoader for DOCX."""
    extracted_docs = []
    file_lower = file_name.lower()

    if file_lower.endswith('.pdf'):
        print(f"Starting OCR for PDF: {file_name}")
        try:
            # --- Explicit Poppler Path ---
            images = convert_from_path(
                file_path,
                poppler_path=r"C:\Program Files\poppler-25.07.0\Library\bin" # Use your confirmed path
            )
            print(f"Converted {len(images)} PDF pages to images.")

            for i, image in enumerate(images):
                page_num = i + 1
                try:
                    # Perform OCR
                    # Consider adding config options like --psm if needed
                    text = pytesseract.image_to_string(image, lang='eng')
                    print(f"  - OCR extracted text from page {page_num} (length: {len(text)})")
                    page_doc = Document(
                        page_content=text if text else "", # Ensure content is string
                        metadata={'source': file_name, 'page': page_num}
                    )
                    extracted_docs.append(page_doc)
                except pytesseract.TesseractNotFoundError:
                    print("ERROR: Tesseract executable not found or not configured.")
                    print("Ensure Tesseract is installed and pytesseract.pytesseract.tesseract_cmd is set if needed.")
                    raise HTTPException(status_code=500, detail="Tesseract OCR engine not found on server.")
                except Exception as page_e:
                    print(f"  - Error during OCR on page {page_num}: {page_e}")
                    extracted_docs.append(Document(page_content="", metadata={'source': file_name, 'page': page_num, 'error': str(page_e)}))

            if not extracted_docs:
                 print(f"Warning: OCR yielded no documents for {file_name}")

        except Exception as e:
            # Catch errors specifically from pdf2image/poppler if possible
            print(f"Error converting PDF to images or during OCR process: {e}")
            # Check if it's the specific Poppler error message
            if "Unable to get page count" in str(e):
                 print("Poppler error detected. Ensure Poppler's bin directory is correctly specified in poppler_path.")
                 raise HTTPException(status_code=500, detail="Error finding or using Poppler PDF tools.")
            else:
                 raise HTTPException(status_code=500, detail=f"Error processing PDF for OCR: {str(e)}")


    elif file_lower.endswith('.docx'):
        print(f"Loading DOCX: {file_name}")
        try:
            loader = Docx2txtLoader(file_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata['source'] = file_name # Ensure metadata
            extracted_docs.extend(docs)
        except Exception as e:
            print(f"Error loading DOCX {file_name}: {e}")
            raise HTTPException(status_code=500, detail=f"Error processing DOCX: {str(e)}")
    else:
        print(f"Unsupported file type: {file_name}")
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_name}. Only PDF and DOCX supported.")

    print(f"Finished extraction for {file_name}. Got {len(extracted_docs)} document sections.")
    return extracted_docs

def create_single_file_vector_store(file_path: str, file_name: str) -> VectorStore:
    """Processes file (with OCR) and creates in-memory vector store."""
    print(f"Processing vector store for: {file_name}")
    try:
        docs = extract_text_with_ocr(file_path, file_name)
        if not docs:
             print(f"No text extracted from {file_name}, creating empty vector store.")
             return Chroma.from_documents(documents=[], embedding=embeddings_model)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        print(f"Split {file_name} into {len(splits)} chunks.")
        if not splits:
             print(f"Splitting resulted in zero chunks for {file_name}, creating empty vector store.")
             return Chroma.from_documents(documents=[], embedding=embeddings_model)

        print(f"Creating embeddings for {len(splits)} chunks...")
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings_model)
        print(f"Vector store created successfully for '{file_name}'.")
        return vectorstore

    except HTTPException as httpe:
         raise httpe # Pass HTTP exceptions up
    except Exception as e:
        print(f"Unexpected error creating vector store for {file_name}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during vector store creation.")


def load_or_create_folder_vector_store(folder_path: str) -> Chroma:
    """Loads persistent store, indexes new files (with OCR)."""
    print(f"Loading/Updating knowledge base: {folder_path}")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path); print(f"Created directory: {folder_path}")

    vectorstore = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings_model
    )

    try: # Robust check for existing files
        existing_metadatas = vectorstore.get(include=['metadatas']).get('metadatas', [])
        indexed_files = {meta['source'] for meta in existing_metadatas if meta and 'source' in meta}
    except Exception as e:
        print(f"Warning: Could not get metadata from Chroma (may be empty): {e}. Will scan all files.")
        indexed_files = set()
    print(f"Found {len(indexed_files)} previously indexed files in Chroma.")


    current_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.pdf', '.docx'))]
    new_files = [f for f in current_files if f not in indexed_files]

    if new_files:
        print(f"New files to process: {', '.join(new_files)}")
        all_new_splits = []
        for filename in new_files:
            file_path = os.path.join(folder_path, filename)
            try:
                docs = extract_text_with_ocr(file_path, filename)
                if not docs:
                    print(f"Skipping {filename}: No text extracted.")
                    continue
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                splits = text_splitter.split_documents(docs)
                print(f"Processed '{filename}', {len(splits)} chunks.")
                all_new_splits.extend(splits)
            except HTTPException as httpe:
                print(f"HTTP Error processing {filename}: {httpe.detail}. Skipping this file.")
            except Exception as e:
                print(f"General Error processing {filename}: {e}. Skipping this file.")

        if all_new_splits:
            print(f"Adding {len(all_new_splits)} new chunks to vector store...")
            vectorstore.add_documents(all_new_splits)
            print("Knowledge base updated.")
            # Consider persisting changes if needed: vectorstore.persist()
    else:
        print("Knowledge base is up-to-date.")
    return vectorstore

async def get_rag_response(query: str, vectorstore: VectorStore, base_url: str) -> dict:
    """Performs RAG pipeline (retrieve, generate, TTS)."""
    # (This function remains largely the same)
    print(f"Received query: {query}")
    try:
        retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={'k': 5, 'fetch_k': 20})
        doc_count = 0
        if isinstance(vectorstore, Chroma) and hasattr(vectorstore, '_collection'):
             doc_count = vectorstore._collection.count()

        if doc_count == 0:
             print("Vector store is empty, cannot retrieve.")
             retrieved_docs = []
        else:
             retrieved_docs = await retriever.ainvoke(query)

        if not retrieved_docs:
            print("No relevant documents found.")
            return {"response": "Could not find relevant information.", "source": None, "audio_url": None}

        context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs if doc.page_content])
        top_source_filename = retrieved_docs[0].metadata.get('source', 'Unknown') if retrieved_docs else 'Unknown'

        # Handle case where context might be empty
        if not context.strip():
             print("Retrieved documents contained no usable text content.")
             return {"response": "Found relevant document sections, but they contained no text.", "source": top_source_filename, "audio_url": None}

        template = "Use the following context to answer the question concisely... \n\nContext: {context}\nQuestion: {question}\nAnswer:"
        prompt = PromptTemplate.from_template(template)
        rag_chain = prompt | llm | StrOutputParser()

        print("Generating LLM response...")
        response_text = await rag_chain.ainvoke({"context": context, "question": query})

        print("Generating audio...")
        audio_url = generate_audio_file_url(response_text, base_url)

        print("RAG response complete.")
        return {"response": response_text, "source": top_source_filename, "audio_url": audio_url}

    except Exception as e:
        print(f"Error in RAG pipeline: {e}")
        raise e


# ===================================================================
# FASTAPI APP INITIALIZATION & ENDPOINTS
# ===================================================================
app = FastAPI(
    title="Advanced RAG Agent API",
    description="API for the Uni-RAG Agent"
)

# --- Add CORS Middleware ---
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:8081",
    # Add your React app's deployed URL here later
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Mount Static Directory ---
app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")

# --- Load Persistent Knowledge Base ---
try:
    folder_vector_store = load_or_create_folder_vector_store(KNOWLEDGE_BASE_PATH)
    doc_count = 0
    if folder_vector_store and hasattr(folder_vector_store, '_collection'):
         doc_count = folder_vector_store._collection.count()
    print(f"Persistent vector store loaded. {doc_count} documents indexed.")
except Exception as e:
    print(f"CRITICAL: Failed to load persistent vector store: {e}")
    folder_vector_store = None # Handle this in the endpoint

# ===================================================================
# API DATA MODELS (Pydantic)
# ===================================================================
class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    query: str
    response: str
    source: Optional[str] = None
    audio_url: Optional[str] = None

# ===================================================================
# API ENDPOINTS
# ===================================================================

@app.get("/")
def read_root():
    return {"status": "Advanced RAG Agent API is running"}

@app.post("/refresh-knowledge-base")
def refresh_knowledge_base():
    """Triggers a re-scan of the knowledge base folder."""
    global folder_vector_store
    try:
        folder_vector_store = load_or_create_folder_vector_store(KNOWLEDGE_BASE_PATH)
        count = 0
        if folder_vector_store and hasattr(folder_vector_store, '_collection'):
             count = folder_vector_store._collection.count()
        return {"status": "Knowledge base refreshed", "documents_indexed": count}
    except Exception as e:
        print(f"Error refreshing knowledge base: {e}")
        raise HTTPException(status_code=500, detail="Error refreshing knowledge base.")

@app.post("/chat-folder", response_model=ChatResponse)
async def chat_with_folder(request: ChatRequest):
    """Chat with the persistent, pre-loaded folder of documents."""
    if folder_vector_store is None:
        raise HTTPException(status_code=500, detail="Store not loaded.")

    base_url = "http://127.0.0.1:8000/" # Adjust if needed
    try:
        response_data = await get_rag_response(request.query, folder_vector_store, base_url)
        return ChatResponse(query=request.query, **response_data)
    except Exception as e:
         print(f"Error in /chat-folder handling: {e}")
         raise HTTPException(status_code=500, detail="Error processing request.")


@app.post("/chat-file", response_model=ChatResponse)
async def chat_with_single_file(
    query: str = Form(...),
    file: UploadFile = File(...)
):
    """Upload a single file for a temporary chat session. Caches the processed vector store."""
    tmp_file_path = None
    vectorstore = None
    file_hash = None

    try:
        suffix = os.path.splitext(file.filename)[1].lower()
        if suffix not in ['.pdf', '.docx']:
            raise HTTPException(status_code=400, detail="Unsupported file type.")

        # --- Caching Step 1: Calculate File Hash ---
        file_content = await file.read()
        await file.seek(0) # Reset pointer
        file_hash = hashlib.md5(file_content).hexdigest()
        print(f"Calculated hash for {file.filename}: {file_hash}")

        # --- Caching Step 2: Check Cache ---
        if file_hash in temp_vector_stores_cache:
            print(f"Cache hit for file hash: {file_hash}. Reusing vector store.")
            vectorstore = temp_vector_stores_cache[file_hash]
        else:
            print(f"Cache miss for file hash: {file_hash}. Processing file...")
            # --- Process File (Only if not in cache) ---
            # Ensure filename is safe
            safe_filename = "".join(c for c in file.filename if c.isalnum() or c in (' ', '.', '_')).rstrip()
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(file_content) # Write content read earlier
                tmp_file_path = tmp_file.name
            print(f"Temp file saved: {tmp_file_path}")

            vectorstore = create_single_file_vector_store(tmp_file_path, safe_filename)

            # --- Caching Step 3: Store in Cache ---
            if len(temp_vector_stores_cache) >= MAX_CACHE_SIZE:
                 oldest_key = next(iter(temp_vector_stores_cache))
                 print(f"Cache full. Evicting oldest entry: {oldest_key}")
                 del temp_vector_stores_cache[oldest_key]
            temp_vector_stores_cache[file_hash] = vectorstore
            print(f"Stored vector store in cache for hash: {file_hash}")

        # --- Use the vectorstore ---
        base_url = "http://127.0.0.1:8000/" # Adjust if needed
        response_data = await get_rag_response(query, vectorstore, base_url)
        return ChatResponse(query=query, **response_data)

    except HTTPException as httpe:
         raise httpe # Re-raise known HTTP errors
    except pytesseract.TesseractNotFoundError:
         print("TesseractNotFoundError caught in endpoint.")
         raise HTTPException(status_code=500, detail="OCR engine not found/configured.")
    except Exception as e:
        print(f"Unexpected error in /chat-file: {type(e).__name__}: {e}")
        error_detail = f"Error processing file (hash: {file_hash})" if file_hash else "Error processing file."
        raise HTTPException(status_code=500, detail=error_detail)
    finally:
        # Cleanup temp file ONLY if created
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.remove(tmp_file_path); print(f"Cleaned up temp file: {tmp_file_path}")
            except Exception as cleanup_e:
                 print(f"Error cleaning up {tmp_file_path}: {cleanup_e}")
        # Always close uploaded file object
        if file and hasattr(file, 'file') and not file.file.closed:
             file.file.close()


# ===================================================================
# RUN THE API
# ===================================================================

if __name__ == "__main__":
    # Add explicit check for Tesseract command if needed, before starting server
    try:
        # This uses the command set at the top level
        cmd = pytesseract.pytesseract.tesseract_cmd
        if cmd and not os.path.exists(cmd):
            print(f"WARNING: Configured Tesseract command not found at '{cmd}'")
            print("OCR will likely fail. Ensure the path is correct.")
        else:
            print(f"Tesseract command set to: {cmd}")
            # Optional: Verify Tesseract version
            try:
                 tesseract_version = pytesseract.get_tesseract_version()
                 print(f"Successfully found Tesseract version: {tesseract_version}")
            except Exception as tv_e:
                 print(f"Could not get Tesseract version (check installation/PATH): {tv_e}")
    except Exception as tess_check_e:
        print(f"Could not verify Tesseract configuration: {tess_check_e}")
        print("Ensure Tesseract is installed and PATH or tesseract_cmd is set.")

    print("Starting Uvicorn server...")
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )
