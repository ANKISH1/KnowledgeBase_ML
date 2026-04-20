from pypdf import PdfReader
from setup import collection
import uuid
import os


def chunk_text(text, chunk_size = 500):
    chunks= []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size]
        chunks.append(chunk)
    return chunks

filename = input("Enter path to file:")

if not os.path.exists(filename):
    print("File not found. Check path.")
    exit()

if filename.endswith('.pdf'):
    reader = PdfReader(filename)
    text = ""
    for page in reader.pages:
        text += page.extract_text()

if filename.endswith('.txt'):
    with open (filename, "r") as f:
        text = f.read()            

chunks =chunk_text(text)
collection.add(
    documents=chunks,
    ids = [str(uuid.uuid4()) for _ in range(len(chunks))]
    )

print(f"{len(chunks)}chunks added to knowledge base")