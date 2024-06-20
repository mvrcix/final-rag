import os
import json
from langchain.text_splitter import TokenTextSplitter

def determine_chunk_size(text, max_chunk_size, overlap_ratio=0.1):
    """
    Function to determine the chunk size based on the length of the text.
    """
    overlap_size = int(max_chunk_size * overlap_ratio)
    num_chunks = len(text) // (max_chunk_size - overlap_size) + 1
    actual_chunk_size = len(text) // num_chunks
    return actual_chunk_size



def split_text_into_chunks(text, chunk_size):
    """
    Function to split text into chunks of specified size.
    """
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks



def process_json_files_in_directory(directory, max_chunk_size):
    text_splitter = TokenTextSplitter(
        model_name="gpt-3.5-turbo",
        chunk_size=max_chunk_size,
        chunk_overlap = 50
    ) 
    
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith('.json'):
                json_file_path = os.path.join(root, filename)
                with open(json_file_path, 'r') as file:
                    data_list = json.load(file)
                    
                # Process each dictionary in the list
                for data in data_list:
                    long_text = data['text']  # Assuming 'text' is the key containing the long string of text

                    # Determine the chunk size based on the length of the text
                    # chunk_size = determine_chunk_size(long_text, max_chunk_size)
                    
                    # Split the text into chunks
                    # chunks = split_text_into_chunks(long_text, chunk_size)
                    chunks = text_splitter.split_text(long_text)


                    # Create a list of dictionaries where each dictionary contains a single chunk of text
                    chunk_dicts = [{'type': 'image_text', 'text': chunk} for chunk in chunks]
                    
                    # Save the modified JSON back to the original file
                    with open(json_file_path, 'w') as file:
                        json.dump(chunk_dicts, file, indent=4)
                        print(f"Processed: {json_file_path}")

# Example usage
directory = '/home/vqa/RAG/10_64'  
max_chunk_size = 64
process_json_files_in_directory(directory, max_chunk_size)
