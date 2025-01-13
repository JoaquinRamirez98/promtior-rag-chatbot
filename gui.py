import customtkinter as ctk
from chatbot import ask_question

class ChatbotGUI:
    def __init__(self):
        # Configuración inicial de la ventana
        ctk.set_appearance_mode("System")  # "Dark" o "Light"
        ctk.set_default_color_theme("blue")  # Tema por defecto

        self.window = ctk.CTk()
        self.window.title("Promtior Chatbot")
        self.window.geometry("700x500")

        self.chat_history = []

        # Título de la aplicación
        self.title_label = ctk.CTkLabel(
            self.window, 
            text="Promtior Chatbot", 
            font=ctk.CTkFont(size=20, weight="bold")
        )
        self.title_label.pack(pady=10)

        # Área de chat (deshabilitada para evitar edición)
        self.chat_display = ctk.CTkTextbox(self.window, width=650, height=300, wrap="word")
        self.chat_display.configure(state="disabled")  # Deshabilitar edición
        self.chat_display.pack(pady=10)

        # Campo de entrada
        self.input_frame = ctk.CTkFrame(self.window)
        self.input_frame.pack(fill="x", pady=10)

        self.input_field = ctk.CTkEntry(self.input_frame, width=520, placeholder_text="Type your message here...")
        self.input_field.pack(side="left", padx=10, pady=5)
        self.input_field.bind("<Return>", self.send_message)

        # Botón de enviar
        self.send_button = ctk.CTkButton(self.input_frame, text="Send", command=self.send_message)
        self.send_button.pack(side="right", padx=10)

    def send_message(self, event=None):
        """Envía el mensaje del usuario y muestra la respuesta del bot."""
        user_message = self.input_field.get().strip()
        if user_message:
            self.display_message("🔵 You", user_message, sender_type="user")
            self.input_field.delete(0, ctk.END)

            bot_message, self.chat_history = ask_question(user_message, self.chat_history)
            self.display_message("🤖 Bot", bot_message, sender_type="bot")

    def display_message(self, sender, message, sender_type="user"):
        """Muestra un mensaje en el área de chat con formato y color según el remitente."""
        self.chat_display.configure(state="normal")

        if sender_type == "user":
            # Mensaje del usuario
            formatted_message = f"{sender}: {message}\n\n"
            self.chat_display.insert("end", formatted_message, ("user",))
            self.chat_display.tag_config("user", foreground="#1E90FF")  # Solo color
        elif sender_type == "bot":
            # Mensaje del bot
            formatted_message = f"{sender}: {message}\n\n"
            self.chat_display.insert("end", formatted_message, ("bot",))
            self.chat_display.tag_config("bot", foreground="#32CD32")  # Solo color

        self.chat_display.configure(state="disabled")
        self.chat_display.see("end")  # Desplazamiento automático

    def run(self):
        """Ejecuta la aplicación."""
        self.window.mainloop()


if __name__ == "__main__":
    gui = ChatbotGUI()
    gui.run()
