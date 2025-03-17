# chatbot.py
import os
import google.generativeai as genai
from utils import create_vectorstore, load_website_content, load_pdf_content
from langchain_huggingface import HuggingFaceEmbeddings
from flask import Flask, request, jsonify, render_template

# Define la clave API directamente en el código
api_key = "AIzaSyCpTy_eqpCsrOSyBw6YFjyaVRH2x_CbDH0"  # Reemplaza con tu clave real

# Configurar tu clave de API de Gemini
genai.configure(api_key=api_key)

app = Flask(__name__)

# Inicializar el vectorstore (esto puede tardar un tiempo, hazlo solo una vez)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
website_url = "https://www.ejemplo.com"  # Puedes dejarlo vacío si solo usas el PDF
pdf_path = "AI Engineer.pdf"  # Ruta correcta al archivo PDF

try:
    website_content = load_website_content(website_url) or ""
    #pdf_content = load_pdf_content(pdf_path) or "" #Elimino el PDF

    #all_content = website_content + "\n" + pdf_content #Elimino el PDF
    all_content = website_content #Elimino el PDF

    print(f"Contenido a vectorizar: {all_content}")

    vectorstore = create_vectorstore(all_content, embeddings)

except Exception as e:
    print(f"Error durante la inicialización: {e}")
    vectorstore = None  # Importante: Establecer vectorstore a None si falla la inicialización
    initialization_error = str(e) # Guarda el error

def ask_question(user_question, vectorstore):
    """Generates a response from the Gemini model based on the given prompt."""
    try:
        if vectorstore is None:
            return f"Lo siento, el chatbot no pudo inicializarse correctamente: {initialization_error}"
        # 1. Buscar documentos relevantes en el vectorstore
        relevant_docs = vectorstore.similarity_search(user_question, k=3)  # k=3 para obtener los 3 documentos más relevantes
        context = "\n".join([doc.page_content for doc in relevant_docs])  # Unimos los documentos

        # 2. Construir el prompt con el contexto y la pregunta
        prompt = f"""
        Eres un asistente conversacional amigable e informado.
        Usa la información proporcionada en "Contexto" para responder a la pregunta del usuario.
        Si no encuentras la información en el "Contexto", usa tu conocimiento general para ayudar de manera útil y cortés.

        Contexto:
        {context}

        Pregunta: {user_question}

        Respuesta:
        """

        # 3. Generar la respuesta del modelo Gemini
        model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error en ask_question: Tipo de error: {type(e)}, Mensaje: {str(e)}")
        return f"Lo siento, ocurrió un error: {str(e)}"

@app.route('/ask', methods=['POST'])
def ask():
    """Endpoint para recibir preguntas y devolver respuestas del chatbot."""
    try:
        data = request.get_json()
        user_message = data['question']

        answer = ask_question(user_message, vectorstore)

        return jsonify({'answer': answer})

    except Exception as e:
        print(f"Error en el endpoint /ask: {e}")
        return jsonify({'error': str(e)}), 500  # Devolver un código de error HTTP

@app.route('/')
def index():
    return render_template('index.html')  # Servir el archivo index.html

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))