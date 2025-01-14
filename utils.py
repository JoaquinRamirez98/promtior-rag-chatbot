from langchain_community.vectorstores import FAISS
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup
import requests
import os
from PyPDF2 import PdfReader
import os
from langchain.text_splitter import CharacterTextSplitter

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
    """Carga el contenido de un archivo PDF utilizando PyPDF2."""
    if not os.path.exists(pdf_path):
        print(f"PDF file not found: {pdf_path}")
        return None

    try:
        pdf_text = ""
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            pdf_text += page.extract_text()  # Extrae el texto de cada página

        if not pdf_text.strip():
            print(f"El PDF no contiene texto legible o está dañado: {pdf_path}")
            return None

        return pdf_text.strip()  # Devuelve el texto limpio

    except Exception as e:
        print(f"Error al leer el PDF con PyPDF2: {e}")
        return None

def create_vectorstore(text, embeddings_class):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Ajuste para chunks más grandes
    chunks = text_splitter.split_text(text)

    print(f"Generated {len(chunks)} text chunks.")
    return FAISS.from_texts(chunks, embeddings_class)

# Función para realizar consultas al chatbot
def query_chatbot(qa_chain, question, chat_history):
    response = qa_chain.invoke({
        'question': question,
        'chat_history': chat_history
    })
    return response['answer']
