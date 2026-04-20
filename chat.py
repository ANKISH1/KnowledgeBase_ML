from setup import collection, groq_client

def get_context(question):
    results = collection.query(
        query_texts= [question],
        n_results=3
    )

    retrieved_docs = results['documents'][0]

    return "\n".join(retrieved_docs)


messages = [
    {
        "role":"system",
        "content":"base instructions"
    }
]

print("Knowledge Base Chat (type 'quit' to exit)")
print("─" * 40)


while True:
    question = input("Your question: ")
    if question.lower() == "quit":
        break

    context = get_context(question)  

    messages[0]= {
        "role":"system",
        "content":f"""You are a strict knowledge base assistant.
                    Answer ONLY from this context:
                    {context}
                    If answer is not in context, say: 'This information is not in your knowledge base.'
                    Never use outside knowledge."""
    }

    messages.append({
        "role":"user",
        "content":question
    })

    response = groq_client.chat.completions.create(
        model = "llama-3.3-70b-versatile",
        messages=messages
    )
    reply = response.choices[0].message.content

    messages.append(
        {
            "role":"assistant",
            "content":reply
        }
    )
    print(f"AI: {reply}")
    print()