from flask import Flask, request, jsonify, render_template
from chatbot import ask_question  # Importa la función ask_question

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")  # Sirve el archivo HTML

@app.route("/favicon.ico")
def favicon():
    return "", 204  # Responde con un código 204 (sin contenido) para evitar el error

@app.route("/ask", methods=["POST"])
def ask():
    # Recibe la pregunta del usuario desde la solicitud POST
    user_question = request.json.get("question")
    answer = ask_question(user_question)  # Llama a ask_question sin chat_history
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
