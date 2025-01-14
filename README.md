

Promtior Chatbot
Descripción del Proyecto
Este proyecto implementa un chatbot inteligente utilizando la arquitectura Retrieval-Augmented Generation (RAG) y la biblioteca LangChain. El chatbot está diseñado para responder preguntas sobre el contenido del sitio web de Promtior y documentos PDF relevantes, proporcionando respuestas precisas y contextuales mediante la combinación de búsqueda y generación de texto.

Además, el chatbot está integrado con un sistema de almacenamiento de embeddings eficiente, usando FAISS, y aprovecha el modelo Ollama para generar embeddings de los textos extraídos.

Características Principales
RAG (Retrieval-Augmented Generation): Combina la búsqueda de información en documentos con la generación de texto para ofrecer respuestas precisas basadas en el contenido relevante.
Almacenamiento de Embeddings: Utiliza FAISS para almacenar y gestionar eficientemente los embeddings generados.
Soporte para Documentos PDF: Permite cargar documentos PDF adicionales y extraer texto para enriquecer la base de conocimiento del chatbot.
Data Pipeline (ETL): Incluye un pipeline de extracción, transformación y carga (ETL) de datos desde diversas fuentes (sitio web, PDFs, etc.), procesados antes de ser almacenados en FAISS.
Modelo Generativo: Utiliza LangChain RAG y un modelo generativo (como Llama o GPT-3) para la generación de respuestas a partir de los datos recuperados.
Interfaz de Usuario (GUI): Ofrece una interfaz gráfica creada con CustomTkinter para interactuar con el chatbot de manera sencilla.
Despliegue en la Nube: El proyecto está desplegado en Railway, asegurando la accesibilidad del chatbot de manera remota.

Tecnologías Utilizadas
Lenguaje de Programación: Python
Framework de IA: LangChain
Embeddings: Ollama (LLaMA3)
Almacenamiento Vectorial: FAISS
Interfaz de Usuario: CustomTkinter
Despliegue: Railway
Procesamiento de PDFs: PyPDF2
Extracción Web: BeautifulSoup (requests)
API de Ollama: Conexión para la generación de embeddings

---

## Estructura del Proyecto

promtior/
├── chatbot.py          # Lógica principal del chatbot
├── gui.py              # Interfaz gráfica para interactuar con el chatbot
├── utils.py            # Funciones de utilidad (procesamiento de texto y PDF, carga de datos)
├── requirements.txt    # Dependencias del proyecto
├── README.md           # Descripción del proyecto
├── resources/          # Archivos adicionales (como documentos PDF)
└── .env                # Variables de entorno necesarias (API Keys, configuraciones)


Requisitos Previos
Python 3.10 o superior.
Git instalado en tu máquina.
Cuenta en Railway para el despliegue del proyecto.
Cuenta en Ollama para la generación de embeddings (si es necesario).
Instalación
Clona este repositorio:

git clone https://github.com/JoaquinRamirez98/promtior-rag-chatbot
cd promtior
Crea un entorno virtual e instala las dependencias:

python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt
Crea un archivo .env en la raíz del proyecto y agrega tus claves de API necesarias (si aplica), como la URL de Ollama.
Ejecución Local
Activa el entorno virtual:

source venv/bin/activate  # En Windows: venv\Scripts\activate
Ejecuta el chatbot con la interfaz gráfica:

python gui.py
Carga y Procesa los Datos:

El chatbot soporta dos fuentes de datos principales:

Sitio Web de Promtior: Usando requests y BeautifulSoup para extraer el contenido relevante.
Documentos PDF: Utilizando PyPDF2 para extraer texto de los documentos cargados.
Estos datos son procesados a través de un pipeline ETL (Extract, Transform, Load) antes de ser almacenados en FAISS para su consulta rápida.

Desafíos Encontrados
Problemas con la instalación de dependencias: Al principio, hubo dificultades con la instalación de versiones específicas de bibliotecas. Esto fue solucionado utilizando configuraciones flexibles en pip.
Conexión con Ollama: La integración de Ollama para generar embeddings en la nube (Railway) presentó algunos problemas debido a las restricciones de tamaño, lo cual fue resuelto ajustando la configuración.
Despliegue en Railway: Se realizaron ajustes en las configuraciones de las variables de entorno y las dependencias necesarias para asegurar que el chatbot estuviera disponible en la nube. Aunque no pude acerlo, opte por hacerlo de manera local
Para eso configure el archivo app.py, que con ngrok genero el enlace para acceder: https://bf57-181-94-150-70.ngrok-free.app 
