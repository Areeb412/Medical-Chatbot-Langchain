from src.helper import load_pdf, text_split, download_embeddings
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

# ── 1. Load and chunk the PDF ──────────────────────────────────
print('Loading PDF...')
extracted_data = load_pdf('data/')
print(f'Loaded {len(extracted_data)} pages')

print('Splitting into chunks...')
text_chunks = text_split(extracted_data)
print(f'Created {len(text_chunks)} chunks')

# ── 2. Create OpenAI embeddings client ────────────────────────
print('Loading embeddings model...')
embeddings = download_embeddings()

# ── 3. Initialise Pinecone (new SDK style) ────────────────────
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = 'medical-bot'

# ── 4. Create index if it does not exist ─────────────────────
existing_indexes = [i.name for i in pc.list_indexes()]
if index_name not in existing_indexes:
    print(f'Creating Pinecone index: {index_name}')
    pc.create_index(
        name=index_name,
        dimension=1536,   # OpenAI ada-002 always produces 1536-dim vectors
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )

# ── 5. Embed and upsert chunks ────────────────────────────────
print('Embedding chunks and uploading to Pinecone...')
print('This may take 3-10 minutes depending on PDF size...')

from langchain_pinecone import PineconeVectorStore

docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=index_name
)

print('Done! All chunks stored in Pinecone.')