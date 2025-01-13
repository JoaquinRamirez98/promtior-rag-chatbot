from langchain_ollama.llms import OllamaLLM
from langchain.chains import ConversationalRetrievalChain
from langchain_ollama.embeddings import OllamaEmbeddings
from utils import load_website_content, create_vectorstore, load_pdf_content
from langchain.prompts import PromptTemplate



text = load_website_content('https://www.promtior.ai')


embeddings = OllamaEmbeddings(model="llama3")
vectorstore = create_vectorstore(text, embeddings)

# Configurar el modelo generativo
llama_model = OllamaLLM(model="llama3")

# Plantilla de prompt personalizada
prompt_template = PromptTemplate(
    input_variables=["context", "question", "chat_history"],
    template=(
        "You are an assistant answering questions about Promtior. Based on the following context:\n"
        "{context}\n"
        "Answer the following question: {question}\n"
        "Consider the chat history: {chat_history}"
    )
)

# Crear la cadena de recuperación y generación
qa_chain = ConversationalRetrievalChain.from_llm(
    llama_model, 
    vectorstore.as_retriever(),
    combine_docs_chain_kwargs={"prompt": prompt_template}
)

def ask_question(question, chat_history):
    """Responde a una pregunta utilizando la cadena de QA."""
    retrieved_docs = vectorstore.similarity_search(question, k=5)
    context = "\n".join([doc.page_content for doc in retrieved_docs])

    # Depuración: muestra el contexto usado
    print(f"Context passed to model:\n{context}")

    response = qa_chain({"question": question, "chat_history": chat_history, "context": context})
    return response['answer'], chat_history + [(question, response['answer'])]

def add_pdf_to_vectorstore(pdf_text):
    """Añade el contenido de un PDF al vectorstore existente."""
    global vectorstore
    new_vectorstore = create_vectorstore(pdf_text, embeddings)
    vectorstore.merge_from(new_vectorstore)

# Carga inicial del PDF
pdf_path = "AI Engineer.pdf"
pdf_text = load_pdf_content(pdf_path)

if pdf_text:
    add_pdf_to_vectorstore(pdf_text)
else:
    print("No PDF content loaded or file not found.")

if __name__ == "__main__":
    chat_history = []

    # Preguntar por los servicios
    question = "What services does Promtior offer?"
    answer, chat_history = ask_question(question, chat_history)
    print(f"Answer: {answer}")

    # Preguntar por la fundación
    question = "When was the company founded?"
    answer, chat_history = ask_question(question, chat_history)
    print(f"Answer: {answer}")


# iniciar: venv\Scripts\activate
# ejecutra: python chatbot.py
# ejecutar: gui.py