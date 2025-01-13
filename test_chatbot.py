import unittest
from chatbot import ask_question

class TestChatbot(unittest.TestCase):
    def test_ask_question(self):
        chat_history = []
        question = "What services does Promtior offer?"
        
        response, _ = ask_question(question, chat_history)
        self.assertIn("Promtior offers", response)  # Verifica que la respuesta contenga algo relevante
        
    def test_error_handling(self):
        chat_history = []
        question = "When was the company founded?"
        
        response, _ = ask_question(question, chat_history)
        self.assertIn("Lo siento", response)  # Verifica que el manejo de errores sea correcto

if __name__ == '__main__':
    unittest.main()
