from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLm-L6-v2')

problems = [
    "Reverse a linked list",
    "Find the shortest path in a graph",
    "Check if a binary tree is balanced",
    "Sort an array using merge sort",
    "Find two numbers that add up to a target",
]

problem_embeddings = model.encode(problems)

print(f"Number of problems: {len(problem_embeddings)}")
print(f"Each embedding Size: {len(problem_embeddings[0])}")

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search(query):
    query_embedding = model.encode(query)

    similarities = []
    for i, prob_embedding in enumerate(problem_embeddings):
        score = cosine_similarity(query_embedding, prob_embedding)
        similarities.append((score, problems[i]))

    similarities.sort(reverse=True)

    return similarities    

query = "how do I reverse things in DSA"
results = search(query)

print(f"Query: {query}\n")
for score, problem in results:
    print(f"{score:.3f} → {problem}")