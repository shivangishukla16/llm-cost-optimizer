import ollama

response = ollama.chat(
    model="llama3.2:3b",
    messages=[{"role": "user", "content": "Ek line mein bolo: API kaam kar raha hai?"}],
)

print("Jawab:", response["message"]["content"])
print("Input tokens:", response.get("prompt_eval_count"))
print("Output tokens:", response.get("eval_count"))