import uuid
import os
import json
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema.document import Document
from datasets import Dataset
from langchain_community.llms import Ollama
from langchain.llms import Ollama


def read_txtelements(output_folder, filename, delimiter):
    with open(os.path.join(output_folder, filename), "r") as f:
        return [s.strip() for s in f.read().split(delimiter) if s.strip()]

def read_jsonelements(output_folder, filename):
    with open(os.path.join(output_folder, filename), "r") as json_file:
        return json.load(json_file)


def add_data(image_text_summaries, image_text_elements, image_summaries, retriever):
    # Add texts
    # if text_summaries:
    #     doc_ids = [str(uuid.uuid4()) for _ in text_elements]
    #     summary_texts = [
    #         Document(page_content=s, metadata={id_key: doc_ids[i]})
    #         for i, s in enumerate(text_summaries)]
    #     retriever.vectorstore.add_documents(summary_texts)
    #     retriever.docstore.mset(list(zip(doc_ids, text_elements)))

    # Add image texts
    if image_text_summaries:
        doc_ids = [str(uuid.uuid4()) for _ in image_text_elements]
        summary_texts = [
            Document(page_content=s, metadata={id_key: doc_ids[i]})
            for i, s in enumerate(image_text_summaries)]
        retriever.vectorstore.add_documents(summary_texts)
        retriever.docstore.mset(list(zip(doc_ids, image_text_elements)))

    # Add tables
    # if table_summaries:
    #     table_ids = [str(uuid.uuid4()) for _ in table_elements]
    #     summary_tables = [
    #         Document(page_content=s, metadata={id_key: table_ids[i]})
    #         for i, s in enumerate(table_summaries)]
    #     retriever.vectorstore.add_documents(summary_tables)
    #     retriever.docstore.mset(list(zip(table_ids, table_elements)))

    # Add images
    if image_summaries:
        img_ids = [str(uuid.uuid4()) for _ in image_summaries]
        summary_img = [
            Document(page_content=s, metadata={id_key: img_ids[i]})
            for i, s in enumerate(image_summaries)]
        retriever.vectorstore.add_documents(summary_img)
        retriever.docstore.mset(list(zip(img_ids, image_summaries)))  
    return retriever



def process_subfolder(subfolder_path, retriever):
    print(f"Processing subfolder: {subfolder_path}")
    delimiter = "~~~"
    # text_elements = read_elements(output_folder, "text_elements.txt")
    # table_elements = read_elements(output_folder, "table_elements.txt")
    # text_summaries = read_elements(output_folder, "text_summaries.txt", delimiter)
    # table_summaries = read_elements(output_folder, "table_summaries.txt", delimiter)
    image_text_elements = read_jsonelements(subfolder_path, "imagestexts.json")
    image_text_elements = [i['text'] for i in image_text_elements]
    image_text_summaries = read_txtelements(subfolder_path, "imagestexts_summary.txt", delimiter)
    image_summaries = read_txtelements(subfolder_path, "image_summaries.txt", delimiter)

    retriever = add_data(image_text_summaries, image_text_elements, image_summaries, retriever)
    return retriever 



vectorstore = Chroma(collection_name="summaries", embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))

store = InMemoryStore()  
id_key = "doc_id"
our_retriever = MultiVectorRetriever(vectorstore=vectorstore, docstore=store, id_key=id_key,)
our_retriever.search_kwargs['k'] = 2
#our_retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

output_folder = "/home/vqa/RAG/test_manuals"



for item in os.listdir(output_folder):
    item_path = os.path.join(output_folder, item)
    if os.path.isdir(item_path):
        # Now item_path is a subdirectory within output_folder
        # For each subdirectory, perform the operations you need
        process_subfolder(item_path, our_retriever)



from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


# Prompt template
template = """Answer the question based only on the following context, which can include text and tables:
{context}
Question: {question}
Answer:
"""

# prompt = PromptTemplate.from_template(template)
prompt = ChatPromptTemplate.from_template(template)

model = Ollama(model="llama2:7b-chat")
# model = Ollama(model="llama2", verbose=True)


# RAG pipeline
chain = (
    {"context": our_retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser())


question = "When will the speaker go into standby mode?"
answer = chain.invoke("When will the speaker go into standby mode?")
print(answer)
# print('lol', retriever.get_relevant_documents(question))
