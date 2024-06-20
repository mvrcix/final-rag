import sys 
from retriever import create_retriever
from decodeencode import split_image_text_types
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_community.chat_models import ChatOllama
import pandas as pd
import json
from tabulate import tabulate

__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

main_folder_path = '/home/vqa/RAG/10_manu_2048'
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




data = []
with open('10_manuals.jsonl', 'r') as f:
    for line in f:

        json_data = json.loads(line)
        id = json_data['id'][:-6]
        qa_data = json_data['qa_data']
        for i in qa_data:
            question = i['question']['text']
            answer = i['answer']['text']
        data.append({'id': id, 'question' : question, 'ground_truth': answer})

df = pd.DataFrame(data)
df = pd.DataFrame(data, columns = ['id', 'question', 'ground_truth'])
print(tabulate(df, headers='keys', tablefmt='psql'))

n = len(pd.unique(df['id']))
print("No.of.unique values :", n)


# MAKE THE DATEFRAME INTO A LIST OF TUPLES

qa_list = [(row['question'], row['ground_truth']) for index, row in df.iterrows()]
print(qa_list)


import os

# Update with your API URL if using a hosted instance of Langsmith.
# os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "ls__9478a4fed44b477bbd2a4040c50cc935"  
# os.environ["LANGCHAIN_API_KEY"] = "ls__dfec5f97d1de407f93106c572de7ca06"

project_name = "rag"  # Update with your project name

from langsmith import Client

client = Client()


import uuid

dataset_name = f"Retrieval QA Questions {str(uuid.uuid4())}"
dataset = client.create_dataset(dataset_name=dataset_name)
for q, a in qa_list:
    client.create_example(
        inputs={"question": q}, outputs={"answer": a}, dataset_id=dataset.id
    )


from langchain.smith import RunEvalConfig
from langchain.evaluation import EvaluatorType

eval_config = RunEvalConfig(
    evaluators=[
        RunEvalConfig.LabeledCriteria("relevance"),
        RunEvalConfig.LabeledCriteria("coherence"), 
        "cot_qa"
        ],
    eval_llm = ChatOllama(model="llama2"),
)


_ = await client.arun_on_dataset(dataset_name=dataset_name, llm_or_chain_factory=lambda: chain, evaluation=eval_config,)
