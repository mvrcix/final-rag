from langchain_community.chat_models import ChatOllama
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
import os
import subprocess
import json
import re
from natsort import natsorted 
import glob


# TEXT 
def generate_summary(elements):
    summaries = []
    model = ChatOllama(temperature=0, model="llama2:7b-chat")
    for element in elements:
        text_splitter = CharacterTextSplitter()
        texts = text_splitter.split_text(element['text'])
        chunks = [Document(page_content=e) for e in texts if e != ""]
        chain = load_summarize_chain(model, chain_type='map_reduce')
        summaries.append(chain.run(chunks))
    return summaries


def modify_json(file_path):
    with open(file_path, 'r') as file:
        elements = json.load(file)  # Load the entire JSON content
    summaries = generate_summary(elements)
    new_file_path = os.path.splitext(file_path)[0] + "_summary.txt"
    summary_content = "\n".join(summaries)
    with open(new_file_path, 'w') as new_file:
        new_file.write(summary_content)


# def summarize_and_replace_image(img_path, prompt):
#     summaries = []  # Initialize a list to hold all summaries
#     output_filename = "image_summaries.txt"  # Name of the output file to store all summaries
#     output_filepath = os.path.join(img_path, output_filename)  # Full path to the output file
    
#     for img in natsorted(os.listdir(img_path)):
#         if img.endswith(".jpg"):
#             print(f"Processing image: {img}")
#             base_name = os.path.splitext(img)[0]
#             # Temporarily store each image's summary in a unique file
#             temp_output_file = os.path.join(img_path, f"{base_name}_temp_summary.txt")
#             command = [
#                 "python3",
#                 "/home/vqa/masterthesis/ourproject/runllava.py",
#                 "--i", os.path.join(img_path, img),
#                 "--p", prompt,
#                 "--o", temp_output_file,
#             ]
#             subprocess.run(command, check=True)  # Run the command to generate summary
            
#             # Read the temporary summary file and add its content to the summaries list
#             with open(temp_output_file, 'r') as file:
#                 summary = file.read()
#                 summaries.append(summary)
            
#             # Optionally, remove the temporary summary file
#             os.remove(temp_output_file)
#             os.remove(os.path.join(img_path, img))
#     # After processing all images, write all summaries to a single file with the delimiter
#     with open(output_filepath, 'w') as outfile:
#         outfile.write(f"~~~\n".join(summaries))  # Use ~~~ as delimiter between summaries
    

# def clean_summary(summary_file):
#     # Read the summary from the file
#     with open(summary_file, "r") as file:
#         summary = file.read()
#     # Clean the summary text
#     cleaned_summary = re.sub(r'\n+', '\n', summary)
#     # Write the cleaned summary back to the file
#     with open(summary_file, "w") as file:
#         file.write(cleaned_summary)


folder_path = '/home/vqa/RAG/10_manuals_64_summaries'
prompt = """You are an assistant whose task is to describe images for developing a Visual Question Answering tool.
    Provide a comprehensive description of the image, including all relevant details and elements like graphs, charts, diagrams, or textual information.
    Describe any notable features or patterns observed. Ensure that the description is clear, detailed, and covers all aspects of the image to facilitate understanding it."""


for item in os.listdir(folder_path):
    item_path = os.path.join(folder_path, item)
    if os.path.isdir(item_path):
        # Process each .json file in the directory
        for file in glob.glob(os.path.join(item_path, '*.json')):
            modify_json(file)  # Note: Using the updated function name here
        # Continue handling images as before
        # summarize_and_replace_image(item_path, prompt)
        

