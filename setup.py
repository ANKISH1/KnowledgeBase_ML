from groq import Groq
import chromadb
import os
from dotenv import load_dotenv

load_dotenv()

chroma_client = chromadb.PersistentClient(path="./chroma_storage")
collection = chroma_client.get_or_create_collection(name = "knowledge_base")
groq_client= Groq(api_key=(os.getenv("GROQ_API_KEY")))