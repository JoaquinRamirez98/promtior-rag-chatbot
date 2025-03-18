from flask import Flask, request, jsonify, render_template
from chatbot import ask_question  # Importa solo ask_question
from utils import load_pdf_content, create_vectorstore  # Importa desde utils
from langchain_huggingface import HuggingFaceEmbeddings #Import embeddings instance
import os

app = Flask(__name__)

#Crear instancia de embeddings

#Variable global para el vectorstore (inicialmente vacio)
vectorstore = None

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    global vectorstore #Accede a la variable global

    user_question = request.json.get("question")

    if vectorstore is None:
      return jsonify({"answer": "El chatbot no se ha inicializado correctamente aun. Por favor, intente nuevamente en unos minutos."})

    #Ahora pasamos el vectorstore a la funcion ask_question
    answer = ask_question(user_question, vectorstore)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    #Cargar los datos y crear el vectorstore al inicio
    print("Cargando datos y creando vectorstore...")

    embeddings_instance = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    pdf_path = "AI Engineer.pdf"
    pdf_text = load_pdf_content(pdf_path)

    if pdf_text:
        #Crea el vectorstore y lo asigna a la variable global
        vectorstore = create_vectorstore(pdf_text, embeddings_instance)
        print(f"Vectorstore creado Correctamente")
    else:
        print(f"Error al cargar el PDF")