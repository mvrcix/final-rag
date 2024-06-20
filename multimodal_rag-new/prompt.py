from retriever import create_retriever
from decodeencode import split_image_text_types
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_community.chat_models import ChatOllama


main_folder_path = '/home/vqa/RAG/20_manuals_extracted'
retriever, vectorstore = create_retriever(main_folder_path)
vectorstore.get()


def prompt_func(data_dict):
    formatted_texts = "\n".join(data_dict["context"]["texts"])
    messages = []
    if data_dict["context"]["images"]:
        image_message = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{data_dict['context']['images'][0]}"},}
        messages.append(image_message)
    text_message = {
        "type": "text",
        "text": (
            "Provide a precise answer to the user question based on the provided context."
            f"User question: {data_dict['question']}\n\n"
            "Context:\n"
            f"{formatted_texts}"),}

    messages.append(text_message)
    return [HumanMessage(content=messages)]


model = ChatOllama(model="llava")


# RAG pipeline
chain = (
    {
        "context": retriever | RunnableLambda(split_image_text_types),
        "question": RunnablePassthrough(),
    }
    | RunnableLambda(prompt_func)
    | model
    | StrOutputParser()
)

print('answer:')
print(chain.invoke("What should I pay attention to when I use headphones with this CD Player?"))