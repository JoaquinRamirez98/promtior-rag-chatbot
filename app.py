import os
import google.generativeai as genai
from utils import create_vectorstore, load_pdf_content
from langchain_huggingface import HuggingFaceEmbeddings
from flask import Flask, request, jsonify, render_template
import logging  # Importa la biblioteca logging

# Configura el logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define la clave API directamente en el código
api_key = "AIzaSyCpTy_eqpCsrOSyBw6YFjyaVRH2x_CbDH0"  # Reemplaza con tu clave real

# Configurar tu clave de API de Gemini
genai.configure(api_key=api_key)

app = Flask(__name__)

# Variable global para el vectorstore (inicialmente vacio)
vectorstore = None

# Define la función ask_question en app.py
def ask_question(user_question, vectorstore):
    """Generates a response from the Gemini model based on the given prompt."""
    try:
        if vectorstore is None:
            return f"Lo siento, el chatbot no pudo inicializarse correctamente"
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
        model = genai.GenerativeModel(model_name='gemini-1.5-pro-latest')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logging.exception("Error en ask_question")  # Registra la excepción
        return f"Lo siento, ocurrió un error: {str(e)}"

# Endpoint para la página principal
@app.route("/")
def home():
    return render_template("index.html")

# Endpoint para recibir preguntas y devolver respuestas del chatbot
@app.route("/ask", methods=["POST"])
def ask():
    global vectorstore  # Accede a la variable global

    user_question = request.get_json().get("question")

    if vectorstore is None:
        return jsonify({"answer": "El chatbot no se ha inicializado correctamente aun. Por favor, intente nuevamente en unos minutos."})

    # Ahora pasamos el vectorstore a la función ask_question
    answer = ask_question(user_question, vectorstore)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    # Cargar los datos y crear el vectorstore al inicio
    print("Cargando datos y creando vectorstore...")

    try:  # Añade un bloque try...except para capturar excepciones
        embeddings_instance = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        pdf_path = "AI Engineer.pdf"
        pdf_text = load_pdf_content(pdf_path)

        if pdf_text:
            # Crea el vectorstore y lo asigna a la variable global
            vectorstore = create_vectorstore(pdf_text, embeddings_instance)
            print("Vectorstore creado Correctamente")
        else:
            print("Error al cargar el PDF")
    except Exception as e:
        logging.exception("Error durante la inicialización")  # Registra la excepción
        print(f"Error durante la inicialización: {e}")
        vectorstore = None  # Asegúrate de que vectorstore sea None en caso de error

    port = int(os.environ.get('PORT', 8000))
    app.run(debug=True, host="0.0.0.0", port=port)