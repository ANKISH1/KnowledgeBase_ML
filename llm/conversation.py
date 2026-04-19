import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Conversation history
messages = [
    {
        "role": "system",
        "content": "You are a helpful ML teacher. Answer concisely."
    }
]

print("Chat with AI (type 'quit' to exit)")
print("─" * 40)

while True:
    user_input = input("You: ")
    
    if user_input.lower() == 'quit':
        break
    
    messages.append({
        "role": "user",
        "content": user_input
    })
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages
    )
    
    reply = response.choices[0].message.content
    
    messages.append({
        "role": "assistant",
        "content": reply
    })
    
    print(f"AI: {reply}")
    print()