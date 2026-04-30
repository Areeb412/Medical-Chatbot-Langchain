# Medical Chatbot LangChain

A Flask-based medical question-answering chatbot that uses LangChain, OpenAI embeddings, and Pinecone vector search to answer questions from indexed medical PDF documents.

## Features

- Clean web chat interface built with Flask templates, HTML, CSS, and JavaScript
- Retrieval augmented generation using LangChain LCEL
- Pinecone vector database for semantic document search
- OpenAI embeddings for document indexing
- PDF ingestion and chunking pipeline
- Simple `/get` endpoint for chat responses

## Tech Stack

- Python 3.12+
- Flask
- LangChain
- OpenAI
- Pinecone
- PyPDF
- python-dotenv

## Project Structure

```text
.
├── app.py                 # Flask app and chat chain
├── store_index.py         # PDF ingestion and Pinecone indexing script
├── src/
│   ├── helper.py          # PDF loading, text splitting, embeddings
│   └── prompt.py          # Prompt template
├── data/                  # Add medical PDF files here
├── templates/
│   └── chat.html          # Chat UI
├── static/
│   └── style.css          # UI styling
├── requirements.txt
└── README.md
```

## Setup

1. Clone the repository and enter the project folder.

```bash
git clone <your-repository-url>
cd Medical-Chatbot-Langchain
```

2. Create and activate a virtual environment.

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

3. Install dependencies.

```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root.

```env
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

## Index Your PDFs

Add your medical PDF files to the `data/` directory, then run:

```bash
python store_index.py
```

This script loads PDFs, splits them into text chunks, creates OpenAI embeddings, creates the `medical-bot` Pinecone index if needed, and uploads the vectors.

## Run the App

```bash
python app.py
```

Open the app in your browser:

```text
http://localhost:8080
```

## How It Works

1. PDFs in `data/` are loaded with `PyPDFLoader`.
2. Documents are split into small overlapping chunks.
3. Chunks are embedded with OpenAI embeddings.
4. Embeddings are stored in a Pinecone index named `medical-bot`.
5. The Flask app retrieves the most relevant chunks for each user question.
6. LangChain sends the retrieved context and question to the language model.
7. The answer is returned to the chat UI.

## Important Note

This project is for learning and educational support. It should not be used as a replacement for professional medical advice, diagnosis, or treatment.

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Build or refresh the Pinecone index
python store_index.py

# Start the Flask server
python app.py
```

## License

This project includes a `LICENSE` file. Review it before using or distributing the project.
