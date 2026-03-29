from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("Who is the founder of Meta?")

result_new = model.invoke("Give a short description about Llama model")

print(result.content)

print(result_new.content)