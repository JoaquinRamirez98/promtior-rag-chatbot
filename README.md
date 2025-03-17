# Promtior Chatbot

### Descripción del Proyecto
Este proyecto implementa un chatbot inteligente utilizando la arquitectura de **Retrieval-Augmented Generation (RAG)** y la biblioteca **LangChain**. Su objetivo es responder preguntas sobre el contenido del sitio web de Promtior y documentos relevantes, ofreciendo una experiencia interactiva y eficiente para los usuarios.

### Características Principales
- **RAG (Retrieval-Augmented Generation):** Combina búsqueda de información y generación de texto para proporcionar respuestas precisas y contextuales.
- **Interfaz de Usuario (GUI):** Una interfaz gráfica moderna y amigable creada con **CustomTkinter**.
- **Vectorstore:** Implementado con **FAISS** para gestionar embeddings generados con **Ollama Embeddings**.
- **Soporte para Documentos:** Permite cargar documentos PDF adicionales para enriquecer la base de conocimiento.

### Tecnologías Utilizadas
- **Lenguaje de Programación:** Python
- **Framework de IA:** LangChain
- **Embeddings:** Ollama (LLaMA3)
- **Almacenamiento Vectorial:** FAISS
- **Interfaz de Usuario:** CustomTkinter
- **Despliegue:** Railway

---

## Estructura del Proyecto

```plaintext
promtior/
├── chatbot.py          # Lógica principal del chatbot
├── gui.py              # Interfaz gráfica para interactuar con el chatbot
├── utils.py            # Funciones de utilidad (procesamiento de texto y PDF, carga de datos)
├── requirements.txt    # Dependencias del proyecto
├── README.md           # Descripción del proyecto
└── resources/          # Archivos adicionales (como documentos PDF)

Configuración del Proyecto
Requisitos Previos
Python 3.10 o superior.

Git instalado en tu máquina.
Cuenta en Railway para el despliegue.
Instalación
Clona este repositorio:
bash
Copiar código
git clone
cd promtior
Crea un entorno virtual e instala las dependencias:
bash
Copiar código
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt

Crea un archivo .env y agrega tus claves de API necesarias (si aplica).
Ejecución Local
Activa el entorno virtual:
bash
Copiar código
source venv/bin/activate  # En Windows: venv\Scripts\activate

Ejecuta la interfaz gráfica:
bash
Copiar código
python gui.py
