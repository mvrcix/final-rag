import os
from langchain_community.vectorstores import Chroma
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
import glob
import json



def process_subfolder(subfolder_path, vectorstore):
    image_uris = [os.path.join(subfolder_path, img_name) for img_name in os.listdir(subfolder_path) if img_name.endswith(".jpg")]
    text_list = []
    for json_path in glob.glob(os.path.join(subfolder_path, '*.json')):
        with open(json_path, 'r') as json_file:
            data = json.load(json_file)  # Load the JSON content, which is a list of dictionaries
            
            # Iterate through each dictionary in the list and extract the text using the "text" key
            for item in data:
                if "text" in item:
                    text_list.append(item["text"])  # Append the text snippet to the text_list

    # Add images and texts to the vectorstore
    vectorstore.add_images(uris=sorted(image_uris))
    vectorstore.add_texts(texts=text_list)

def create_retriever(base_directory):
    # Initialize components for the vectorstore outside the loop
    vectorstore = Chroma(collection_name="multimodaldata", embedding_function=OpenCLIPEmbeddings())
    store = InMemoryStore()
    id_key = "doc_id"
    retriever = MultiVectorRetriever(vectorstore=vectorstore, docstore=store, id_key=id_key,)

    for item in os.listdir(base_directory):
        item_path = os.path.join(base_directory, item)
        if os.path.isdir(item_path):
            # Pass the shared vectorstore to process each subfolder
            process_subfolder(item_path, vectorstore)

    
#    retriever = vectorstore.as_retriever()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    return retriever, vectorstore