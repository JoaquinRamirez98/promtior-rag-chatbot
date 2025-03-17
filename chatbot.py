import time
import os
import google.generativeai as genai

# Define la clave API directamente en el código
api_key = "AIzaSyCpTy_eqpCsrOSyBw6YFjyaVRH2x_CbDH0"  # Reemplaza con tu clave real

# Configurar tu clave de API de Gemini
genai.configure(api_key=api_key)

def ask_question(user_question, vectorstore):
    """Generates a response from the Gemini model based on the given prompt."""
    try:
        # 1. Buscar documentos relevantes en el vectorstore
        relevant_docs = vectorstore.similarity_search(user_question, k=3)  # k=3 para obtener los 3 documentos más relevantes
        context = "\n".join([doc.page_content for doc in relevant_docs]) #Unimos los documentos

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

def start_interactive_chat():
    #Aqui va el vectorstore
    vectorstore = create_vectorstore()
    print("Bienvenido al chatbot de Promtior. Escribe 'exit' para salir.")
    while True:
        user_message = input("Tú: ")
        if user_message.lower() == "exit":
            print("¡Gracias por usar el chatbot! Hasta luego.")
            break

        answer = ask_question(user_message, vectorstore) #Pasa el vectorstore a la funcion
        print(f"Bot: {answer}")
        time.sleep(1)  # Esperar 1 segundo entre cada llamada a la API <--- Añadido

if __name__ == "__main__":
    start_interactive_chat()