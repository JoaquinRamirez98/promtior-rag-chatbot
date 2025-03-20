# __init__.py
import logging
import azure.functions as func
import os
import google.generativeai as genai
from utils import create_vectorstore, load_pdf_content  # Importa desde utils
from langchain_huggingface import HuggingFaceEmbeddings  # Import embeddings instance
import json

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    try:
        req_body = req.get_json()
    except ValueError:
        return func.HttpResponse(
             "Please pass a question in the request body",
             status_code=400
        )

    user_question = req_body.get('question')

    # Inicializar el vectorstore (esto se hace CADA vez que se recibe una pregunta)
    try:
        embeddings_instance = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        pdf_path = "AI Engineer.pdf"  # Asegúrate de que este archivo esté accesible
        pdf_text = load_pdf_content(pdf_path)

        if pdf_text:
            vectorstore = create_vectorstore(pdf_text, embeddings_instance)
            print("Vectorstore creado Correctamente")
        else:
            logging.error("Error al cargar el PDF.  Revisa la ruta y el archivo.")
            return func.HttpResponse(
                "Error al cargar el PDF.  Revisa los logs.",
                status_code=500
            )
    except Exception as e:
        logging.exception("Error durante la inicialización del vectorstore")  # Log the full exception
        return func.HttpResponse(
             f"Error durante la inicialización: {str(e)}",
             status_code=500
        )

    if user_question:
        try:
            answer = ask_question(user_question, vectorstore)  # Llama a la función ask_question
            return func.HttpResponse(
                json.dumps({
                    "answer": answer
                }),
                mimetype="application/json",
                status_code=200
            )
        except Exception as e:
            logging.exception("Error al responder a la pregunta")
            return func.HttpResponse(
                 f"Error al responder a la pregunta: {str(e)}",
                 status_code=500
            )
    else:
        return func.HttpResponse(
             "Please pass a question in the request body",
             status_code=400
        )

def ask_question(user_question, vectorstore):
    """Generates a response from the Gemini model based on the given prompt."""
    try:
        import google.generativeai as genai
        # Define la clave API directamente en el código
        api_key = os.environ["GOOGLE_API_KEY"]  # Tomar clave de la configuracion

        # Configurar tu clave de API de Gemini
        genai.configure(api_key=api_key)
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
        print(f"Error en ask_question: Tipo de error: {type(e)}, Mensaje: {str(e)}")
        return f"Lo siento, ocurrió un error: {str(e)}"