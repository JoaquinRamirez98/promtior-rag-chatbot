from langchain_ollama.llms import OllamaLLM
from langchain.chains import ConversationalRetrievalChain
from langchain_ollama.embeddings import OllamaEmbeddings
from utils import load_website_content, create_vectorstore, load_pdf_content  # Asegúrate de tener la función load_pdf_content

# Cargar y procesar el contenido del sitio web
text = load_website_content('https://www.promtior.ai')  # Cambia la URL si es necesario
embeddings = OllamaEmbeddings(model="llama3")
vectorstore = create_vectorstore(text, embeddings_class=embeddings)
llama_model = OllamaLLM(model="llama3")
qa_chain = ConversationalRetrievalChain.from_llm(llama_model, vectorstore.as_retriever())

def add_pdf_to_vectorstore(pdf_text):
    """Añade el contenido de un PDF al vectorstore existente."""
    global vectorstore
    new_vectorstore = create_vectorstore(pdf_text, embeddings)
    vectorstore.merge_from(new_vectorstore)

def ask_question(user_question, chat_history):
    """
    Esta función recibe una pregunta del usuario y devuelve la respuesta generada por el modelo.
    """
    # Enviar la pregunta al modelo y obtener la respuesta usando 'invoke'
    result = qa_chain.invoke({"question": user_question, "chat_history": chat_history})
    answer = result['answer']
    updated_chat_history = result['chat_history']
    
    return answer, updated_chat_history

def chat():
    """
    Esta función maneja la interacción con el chatbot desde la terminal.
    """
    print("Bienvenido al chatbot de Promtior. Escribe 'exit' para salir.")
    chat_history = []  # Inicializamos el historial de la conversación
    
    while True:
        user_message = input("Tú: ")  # El usuario escribe su mensaje
        if user_message.lower() == "exit":
            print("¡Gracias por usar el chatbot! Hasta luego.")
            break

        # Obtener la respuesta del chatbot
        answer, chat_history = ask_question(user_message, chat_history)
        print(f"Bot: {answer}")  # Mostrar la respuesta del chatbot

if __name__ == "__main__":
    # Carga inicial del PDF
    pdf_path = "AI Engineer.pdf"  # Ruta del archivo PDF
    pdf_text = load_pdf_content(pdf_path)  # Cargar el contenido del PDF

    if pdf_text:
        add_pdf_to_vectorstore(pdf_text)  # Agregar el contenido del PDF al vectorstore
    else:
        print("No se cargó el contenido del PDF o el archivo no se encontró.")

    chat()  # Ejecutar el chatbot en modo CLI

# iniciar: venv\Scripts\activate
# ejecuta: python chatbot.py
# ejecutar: gui.py
# preguntas:  question = "What services does Promtior offer?"  question = "When was the company founded?"
