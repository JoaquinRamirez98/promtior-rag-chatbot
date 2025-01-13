from langchain_ollama.llms import OllamaLLM
from langchain.chains import ConversationalRetrievalChain
from langchain_ollama.embeddings import OllamaEmbeddings
from utils import load_website_content, create_vectorstore, load_pdf_content

# Cargar y procesar el contenido del sitio web
text = load_website_content('https://www.promtior.ai')  # Cambia la URL si es necesario
embeddings = OllamaEmbeddings(model="llama3")
vectorstore = create_vectorstore(text, embeddings_class=embeddings)
llama_model = OllamaLLM(model="llama3")
qa_chain = ConversationalRetrievalChain.from_llm(llama_model, vectorstore.as_retriever())

# Variable global para mantener el historial de chat
chat_history = []

def add_pdf_to_vectorstore(pdf_text):
    """Añade el contenido de un PDF al vectorstore existente."""
    global vectorstore
    new_vectorstore = create_vectorstore(pdf_text, embeddings)
    vectorstore.merge_from(new_vectorstore)

def ask_question(user_question):
    """
    Esta función recibe una pregunta del usuario y devuelve la respuesta generada por el modelo.
    """
    global chat_history
    result = qa_chain.invoke({"question": user_question, "chat_history": chat_history})
    answer = result['answer']
    chat_history = result['chat_history']  # Actualiza el historial de chat
    
    return answer

def start_interactive_chat():
    """Función para la interacción en la terminal."""
    print("Bienvenido al chatbot de Promtior. Escribe 'exit' para salir.")
    while True:
        user_message = input("Tú: ")  # El usuario escribe su mensaje
        if user_message.lower() == "exit":
            print("¡Gracias por usar el chatbot! Hasta luego.")
            break

        answer = ask_question(user_message)  # Obtiene la respuesta
        print(f"Bot: {answer}")  # Muestra la respuesta

# Evitar que el chat se ejecute automáticamente si se importa este archivo
if __name__ == "__main__":
    # Carga inicial del PDF (si es necesario)
    pdf_path = "AI Engineer.pdf"  # Ruta del archivo PDF
    pdf_text = load_pdf_content(pdf_path)  # Cargar el contenido del PDF

    if pdf_text:
        add_pdf_to_vectorstore(pdf_text)  # Agregar el contenido del PDF al vectorstore
    else:
        print("No se cargó el contenido del PDF o el archivo no se encontró.")
    
    start_interactive_chat()  # Ejecutar el chatbot en modo interactivo si se ejecuta el archivo

# iniciar: venv\Scripts\activate
# ejecuta: python chatbot.py
# ejecutar: gui.py
# preguntas:  question = "What services does Promtior offer?"  question = "When was the company founded?"
