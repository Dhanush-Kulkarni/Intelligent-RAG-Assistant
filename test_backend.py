from rag_backend import chat_with_hr

question = "How many casual leaves are employees entitled to?"

print("\nTesting HR Chatbot...\n")

answer = chat_with_hr(question)

print("Question:", question)
print("\nAnswer:", answer)