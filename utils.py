from langchain_community.vectorstores import FAISS
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup
import requests
import os
import PyPDF2
import pdfplumber

def load_website_content(url):
    """Carga el contenido de un sitio web."""
    try:
        response = requests.get(url)
        response.raise_for_status()  # Lanza una excepción si hay un error de HTTP
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text(separator="\n").strip()
    except requests.exceptions.RequestException as e:
        print(f"Error al cargar el sitio web: {e}")
        return None

def load_pdf_content(pdf_path):
    """Carga el contenido de un archivo PDF con pdfplumber."""
    if not os.path.exists(pdf_path):
        print(f"PDF file not found: {pdf_path}")
        return None

    try:
        pdf_text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                pdf_text += page.extract_text()

        if not pdf_text:
            print(f"El PDF no contiene texto legible o está dañado: {pdf_path}")
            return None

        return pdf_text

    except Exception as e:
        print(f"Error al leer el PDF: {e}")
        return None
    
def create_vectorstore(text, embeddings_class):
    """Crea un vectorstore a partir del texto proporcionado."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(text)

    # Depuración: verifica los fragmentos generados
    print(f"Generated {len(chunks)} text chunks. Sample:")
    for i, chunk in enumerate(chunks[:3]):
        print(f"Chunk {i + 1}: {chunk[:100]}...")

    return FAISS.from_texts(chunks, embeddings_class)

# Función para realizar consultas al chatbot
def query_chatbot(qa_chain, question, chat_history):
    response = qa_chain.invoke({
        'question': question,
        'chat_history': chat_history
    })
    return response['answer']
