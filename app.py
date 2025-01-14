from flask import Flask, request, jsonify, render_template
from chatbot import ask_question, add_pdf_to_vectorstore
import os

app = Flask(__name__)

# Variable global para iniciar la conversación
chat_history = []

# Ruta para cargar la interfaz principal
@app.route("/")
def home():
    return render_template("index.html")

# Ruta para manejar la carga de archivos PDF
@app.route("/upload-pdf", methods=["POST"])
def upload_pdf():
    if "pdf" not in request.files:
        return jsonify({"message": "No se encontró ningún archivo PDF."}), 400

    pdf_file = request.files["pdf"]
    if pdf_file.filename == "":
        return jsonify({"message": "El archivo PDF está vacío."}), 400

    if not pdf_file.filename.endswith(".pdf"):
        return jsonify({"message": "Por favor, sube un archivo PDF válido."}), 400

    # Guardar el archivo temporalmente
    file_path = os.path.join("uploads", pdf_file.filename)
    os.makedirs("uploads", exist_ok=True)  # Crear la carpeta si no existe
    pdf_file.save(file_path)

    # Cargar el contenido del PDF y añadirlo al vectorstore
    from utils import load_pdf_content  # Importar la función de carga
    pdf_text = load_pdf_content(file_path)
    if pdf_text:
        add_pdf_to_vectorstore(pdf_text)
        return jsonify({"message": "El archivo PDF se cargó y procesó correctamente."})
    else:
        return jsonify({"message": "No se pudo leer el contenido del archivo PDF."}), 500

# Ruta para manejar preguntas al chatbot
@app.route("/ask", methods=["POST"])
def ask():
    user_question = request.json.get("question")
    answer = ask_question(user_question)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
