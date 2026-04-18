import chromadb
from groq import Groq
import os
from dotenv import load_dotenv
load_dotenv()

#Initialize
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="problems")
groq_client = Groq(api_key=(os.getenv("GROQ_API_KEY")))

#Knowledge Base
problems = [
    "Reverse a linked list: Given a linked list, reverse it in place",
    "Two Sum: Find two numbers in array that add up to target",
    "Binary Tree Height: Find the maximum height of a binary tree",
    "Graph BFS: Traverse a graph using breadth first search",
    "Merge Sort: Sort an array using divide and conquer approach",
    "Detect cycle in linked list using Floyd's algorithm",
    "Find shortest path in graph using Dijkstra's algorithm",
    "Check if binary tree is balanced using recursion",
]

#Store in vectorDB

collection.add(
    documents=problems,
    ids = [str(i) for i in range(len(problems))]
)

print("Knowledge Base ready")

def rag_search(question):
    #retrieve relevant documents
    results = collection.query(
        query_texts=[question],
        n_results=3
    )
    
    retrieved_docs = results['documents'][0]

    #Augment the prompt
    context = "\n".join(retrieved_docs)

    #Generate answer using LLM
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages = [
            {
                "role":"system",
                "content":f"""You are a DSA assistant for an online Judge. 
                Answer questions based on these relevant problems: 
                {context} 
                Answer based on this context only. Be concise."""    
            },
            {
                "role":"user",
                "content": question
            }
        ]
    )
    return response.choices[0].message.content

#Test

questions = [
    "Which problems involve linked lists?",
    "What graph problems do you have",
    "How do I sort an array"
]
for question in questions:
    print(f"Q:{question}")
    print(f"A:{rag_search(question)}")
    print()