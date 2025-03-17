from flask import Flask, request, jsonify, render_template
from chatbot import ask_question  # Importa solo ask_question
from utils import load_pdf_content, create_vectorstore  # Importa desde utils
from langchain_huggingface import HuggingFaceEmbeddings #Import embeddings instance
import os

app = Flask(__name__)

#Crear instancia de embeddings
embeddings_instance = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

#Variable global para el vectorstore (inicialmente vacio)
vectorstore = None

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload-pdf", methods=["POST"])
def upload_pdf():
    global vectorstore #Accede a la variable global
    if "pdf" not in request.files:
        return jsonify({"message": "No se encontró ningún archivo PDF."}), 400

    pdf_file = request.files["pdf"]
    if pdf_file.filename == "":
        return jsonify({"message": "El archivo PDF está vacío."}), 400

    if not pdf_file.filename.endswith(".pdf"):
        return jsonify({"message": "Por favor, sube un archivo PDF válido."}), 400

    uploads_dir = "uploads"
    os.makedirs(uploads_dir, exist_ok=True)
    file_path = os.path.join(uploads_dir, pdf_file.filename)
    pdf_file.save(file_path)

    pdf_text = load_pdf_content(file_path)
    if pdf_text:
        #Crea el vectorstore y lo asigna a la variable global
        vectorstore = create_vectorstore(pdf_text, embeddings_instance)
        return jsonify({"message": "El archivo PDF se cargó y procesó correctamente."})
    else:
        return jsonify({"message": "No se pudo leer el contenido del archivo PDF."}), 500

@app.route("/ask", methods=["POST"])
def ask():
    global vectorstore #Accede a la variable global

    user_question = request.json.get("question")

    if vectorstore is None:
      return jsonify({"answer": "No se ha cargado ningun PDF aun. Por favor, carga un PDF primero."})

    #Ahora pasamos el vectorstore a la funcion ask_question
    answer = ask_question(user_question, vectorstore)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)