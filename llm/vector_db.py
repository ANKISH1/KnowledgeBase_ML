import chromadb

# Create a local Chroma database
client = chromadb.Client()

# Create a collection (like a table in PostgreSQL)
collection = client.create_collection(name="problems")

print("Database created successfully")

# Our problems
problems = [
    "Reverse a linked list",
    "Find the shortest path in a graph",
    "Check if a binary tree is balanced",
    "Sort an array using merge sort",
    "Find two numbers that add up to a target",
]

# Add to collection
collection.add(
    documents=problems,
    ids=["1", "2", "3", "4", "5"]
)

print("Problems added successfully")

# Search
query = "how do I reverse things in DSA"

results = collection.query(
    query_texts=[query],
    n_results=3
)

print(f"\nQuery: {query}\n")
for doc, distance in zip(results['documents'][0], results['distances'][0]):
    print(f"{distance:.3f} → {doc}")