from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

# this model has a large size so if your PC configuration is not sufficient don't run it on your device it might crash the environment....

llm = HuggingFacePipeline.from_model_id(
    model_id='meta-llama/Llama-3.1-8B-Instruct',
    task='text-generation',
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=100
    )
)
model = ChatHuggingFace(llm=llm)

result = model.invoke("what is the capital of india?")
print(result.content)