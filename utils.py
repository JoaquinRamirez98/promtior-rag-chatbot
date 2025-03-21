from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from bs4 import BeautifulSoup
import requests
import os
from PyPDF2 import PdfReader
from langchain_huggingface import HuggingFaceEmbeddings

def load_website_content(url):
    """Carga el contenido de un sitio web."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text(separator="\n").strip()
    except requests.exceptions.RequestException as e:
        print(f"Error al cargar el sitio web: {e}")
        return None

def load_pdf_content(pdf_path):
    """Carga el contenido de un archivo PDF utilizando PyPDF2."""
    if not os.path.exists(pdf_path):
        print(f"Archivo PDF no encontrado: {pdf_path}")
        return None
    try:
        pdf_text = ""
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pdf_text += text
        if not pdf_text.strip():
            print(f"El PDF no contiene texto legible o está dañado: {pdf_path}")
            return None
        return pdf_text.strip()
    except Exception as e:
        print(f"Error al leer el PDF con PyPDF2: {e}")
        return None

def create_vectorstore(text, embeddings_instance):
    """Crea un almacén vectorial (vectorstore) a partir del texto proporcionado."""
    text_splitter = CharacterTextSplitter(chunk_size=10000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    print(f"Se generaron {len(chunks)} fragmentos de texto.")
    return FAISS.from_texts(chunks, embeddings_instance)

