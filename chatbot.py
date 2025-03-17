# chatbot.py
import os
import google.generativeai as genai
from utils import create_vectorstore, load_website_content, load_pdf_content
from langchain_huggingface import HuggingFaceEmbeddings

# Define la clave API directamente en el código
api_key = "AIzaSyCpTy_eqpCsrOSyBw6YFjyaVRH2x_CbDH0"  # Reemplaza con tu clave real

# Configurar tu clave de API de Gemini
genai.configure(api_key=api_key)


def ask_question(user_question, vectorstore):
    """Generates a response from the Gemini model based on the given prompt."""
    try:
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


def start_interactive_chat():
    # Inicializa los embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Carga el contenido (reemplaza con tu URL y ruta de PDF)
    website_url = "https://www.ejemplo.com"  # Puedes dejarlo vacío si solo usas el PDF
    pdf_path = "AI Engineer.pdf"  # Ruta correcta al archivo PDF

    try:
        website_content = load_website_content(website_url) or ""
        pdf_content = load_pdf_content(pdf_path) or ""
        all_content = website_content + "\n" + pdf_content

        print(f"Contenido a vectorizar: {all_content}")  # Añade esta línea

        # Crea el vectorstore
        vectorstore = create_vectorstore(all_content, embeddings)

        print("Bienvenido al chatbot de Promtior. Escribe 'exit' para salir.")
        while True:
            user_message = input("Tú: ")
            if user_message.lower() == "exit":
                print("¡Gracias por usar el chatbot! Hasta luego.")
                break

            answer = ask_question(user_message, vectorstore)  # Pasa el vectorstore a la funcion
            print(f"Bot: {answer}")

    except Exception as e:
        print(f"Error durante la inicialización: {e}")
        print("El chatbot no pudo inicializarse correctamente.")


if __name__ == "__main__":
    start_interactive_chat()