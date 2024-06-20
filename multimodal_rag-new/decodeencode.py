import os
import json
import base64
import io
from PIL import Image
import glob


def resize_image(image_path, size=(128, 128)):
    """
    Resize an image from a file path.
    Args:
    image_path (str): Path to the original image.
    size (tuple): Desired size of the image as (width, height).
    Returns:
    str: Base64 string of the resized image.
    """
    img = Image.open(image_path)
    resized_img = img.resize(size, Image.LANCZOS)
    buffered = io.BytesIO()
    img_format = 'JPEG' if img.format == 'JPEG' else 'PNG'  # Adjust according to your needs
    resized_img.save(buffered, format=img_format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def is_base64(s):
    """Check if a string is Base64 encoded"""
    try:
        return base64.b64encode(base64.b64decode(s)) == s.encode()
    except Exception:
        return False


def split_image_text_types(docs):
    """Split numpy array images and texts"""
    images = []
    text = []
    for doc in docs:
        doc = doc.page_content  # Extract Document contents
        if is_base64(doc):
            # print('hmm', doc)
            # Resize image to avoid OAI server error
            images.append(
                resize_image(doc, size=(250, 250))
            )  # base64 encoded str
        else:
            # print('meh', doc)
            text.append(doc)
    # print('lenimg', len(images), 'lentxt', len(text))
    return {"images": images, "texts": text}


def process_subfolder(subfolder_path):
    images = []
    texts = []

    # Process JPG files
    for image_path in glob.glob(os.path.join(subfolder_path, '*.jpg')):
        base64_image = resize_image(image_path, size=(250, 250))
        images.append(base64_image)

    # Process JSON files
    for json_path in glob.glob(os.path.join(subfolder_path, '*.json')):
        with open(json_path, 'r') as file:
            # Assuming the JSON file structure is a list of texts
            content = json.load(file)
            if isinstance(content, list):
                texts.extend(content)
            elif isinstance(content, str):
                texts.append(content)
    print(f'Processed {len(images)} images and {len(texts)} texts in {subfolder_path}')


def process_folders(main_folder_path):
    for item in os.listdir(main_folder_path):
        item_path = os.path.join(main_folder_path, item)
        if os.path.isdir(item_path):
            process_subfolder(item_path)



# main_folder_path = '/home/vqa/RAG/20_manuals_extracted'
# process_folders(main_folder_path)