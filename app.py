from flask import Flask, render_template, jsonify, request
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
from src.prompt import prompt_template
from pinecone import Pinecone
import os

app = Flask(__name__)
load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

# ── Load embeddings and connect to Pinecone ───────────────────
embeddings = download_embeddings()

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = 'medical-bot'

docsearch = PineconeVectorStore.from_existing_index(index_name, embeddings)
retriever = docsearch.as_retriever(search_kwargs={'k': 2})

# ── Build the QA chain (modern LCEL style) ────────────────────
PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=['context', 'question']
)

llm = OpenAI(
    model_name='gpt-3.5-turbo-instruct',
    temperature=0.8,
    max_tokens=512
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

qa_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | PROMPT
    | llm
    | StrOutputParser()
)


@app.route('/')
def index():
    return render_template('chat.html')


@app.route('/get', methods=['GET', 'POST'])
def chat():
    msg = request.form.get('msg', '')
    if not msg:
        return jsonify({'response': 'Please enter a message.'})
    print(f'User input: {msg}')
    answer = qa_chain.invoke(msg)
    print(f'Response: {answer}')
    return str(answer)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)