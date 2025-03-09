from langchain_ollama.llms import OllamaLLM
from langchain.chains import ConversationalRetrievalChain
from langchain_ollama.embeddings import OllamaEmbeddings
from utils import load_website_content, create_vectorstore, load_pdf_content
from langchain.prompts import PromptTemplate
from utils import load_pdf_content
import re

# Define un prompt base
base_prompt = """
Eres un asistente conversacional amigable e informado. 
Usa la información proporcionada en "context" para responder las preguntas del usuario.
Si no encuentras información relevante en el "context", usa tu conocimiento general para ayudar de manera útil y cortés.

Contexto: {context}

Pregunta: {question}

Respuesta:
"""



# Cargar y procesar el contenido del sitio web
text = load_website_content('https://www.promtior.ai') 
embeddings = OllamaEmbeddings(model="llama3")
vectorstore = create_vectorstore(text, embeddings_class=embeddings)
llama_model = OllamaLLM(model="llama3", temperature=0.7)
prompt_template = PromptTemplate(template=base_prompt, input_variables=["context", "question"])

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llama_model,
    retriever=vectorstore.as_retriever(),
    combine_docs_chain_kwargs={
        "prompt": prompt_template,
        "document_variable_name": "context",  # Define el nombre de la variable para los documentos
    }
)

# Variable global para mantener el historial de chat
chat_history = []

def formatear_texto(texto):
    # Este regex inserta dos saltos de línea después de un punto seguido de un espacio y una letra mayúscula.
    texto_formateado = re.sub(r'\.\s+([A-Z])', r'.\n\n\1', texto)
    return texto_formateado

def add_pdf_to_vectorstore(pdf_text):
    """Añade el contenido de un PDF al vectorstore existente."""
    global vectorstore
    new_vectorstore = create_vectorstore(pdf_text, embeddings)
    vectorstore.merge_from(new_vectorstore)

def ask_question(user_question):
    """
    Recibe una pregunta del usuario, consulta el modelo generativo utilizando el historial actual,
    y actualiza el historial de la conversación.
    
    Args:
        user_question (str): La pregunta que ingresa el usuario.
    
    Returns:
        str: La respuesta generada por el modelo.
    """
    global chat_history
    try:
        result = qa_chain.invoke({"question": user_question, "chat_history": chat_history})
        answer = result.get("answer", "No estoy seguro de cómo responder eso.")
        chat_history = result.get("chat_history", [])

        # Llamamos a la función para formatear el texto antes de devolver la respuesta
        answer = formatear_texto(answer)
        
        # Si se detectan marcadores de respuesta incompleta, se proporciona un fallback
        if "[insert date here]" in answer or "[fecha de inicio]" in answer or not answer.strip():
            answer = "Lo siento, no tengo esa información específica."
        
        return answer
    except Exception as e:
        return f"Lo siento, ocurrió un error: {str(e)}"

def start_interactive_chat():
    """Función para la interacción en la terminal."""
    print("Bienvenido al chatbot de Promtior. Escribe 'exit' para salir.")
    while True:
        user_message = input("Tú: ")  # El usuario escribe su mensaje
        if user_message.lower() == "exit":
            print("¡Gracias por usar el chatbot! Hasta luego.")
            break

        answer = ask_question(user_message)  # Obtiene la respuesta
        print(f"Bot: {answer}") 

# Evitar que el chat se ejecute automáticamente si se importa este archivo
if __name__ == "__main__":
    # Carga inicial del PDF (si es necesario)
    pdf_path = "AI Engineer.pdf"  # Ruta del archivo PDF
    pdf_text = load_pdf_content(pdf_path)  # Cargar el contenido del PDF

    if pdf_text:
        print("Contenido cargado del PDF:")
        print(pdf_text[:1000])  # Muestra los primeros 1000 caracteres para verificar
        add_pdf_to_vectorstore(pdf_text)  # Agrega el contenido al vectorstore
    else:
        print("No se pudo cargar el contenido del PDF.")
    
    start_interactive_chat()  # Ejecutar el chatbot en modo interactivo si se ejecuta el archivo

# iniciar: venv\Scripts\activate
# ejecuta: python chatbot.py
# preguntas:  question = "What services does Promtior offer?"  question = "When was the company founded?"
