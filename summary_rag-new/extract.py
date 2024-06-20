import fitz
from PIL import Image
import pytesseract
import io
from unstructured.partition.pdf import partition_pdf
import tabula
import cv2
import os 
import numpy as np
import os
import json


def extract_and_save_images(pdf_path, output_dir):
    images = partition_pdf(
        filename=pdf_path,
        extract_images_in_pdf=True,
        infer_table_structure=True,
        chunking_strategy="by_title",
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
        image_output_dir_path=output_dir,)
    return images


def extract_text_from_image(image):
    # Convert the image to grayscale
    grayscale_image = image.convert('L')
    # Use Tesseract OCR to extract text from the image
    text = pytesseract.image_to_string(grayscale_image)
    return text


def extract_text_elements(pdf_path):
    text_elements = []
    image_elements = []
    image_text_combined = ""
    table_elements = []
    # Open the PDF file using PyMuPDF (fitz)
    pdf_document = fitz.open(pdf_path)
    for page_number in range(pdf_document.page_count):
        page = pdf_document.load_page(page_number)
        # Extract images from the page
        images = page.get_images(full=True)
        for image_index, image_info in enumerate(images, start=1):
            # Extract image data
            xref = image_info[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            # Convert the image bytes to a PIL Image
            pil_image = Image.open(io.BytesIO(image_bytes))
            # Extract text from the image using OCR
            image_text = extract_text_from_image(pil_image)
            image_text_combined += str(image_text) + " "
            #image_elements.append({"type": "image", "text": str(image_text)})

        # Extract text from the page that's not in the image
        if not page.get_images(full=True):
            page_text = page.get_text("text")
            if page_text.strip():  # Skip empty text pages
                text_elements.append({"type": "text", "text": str(page_text)})

    # Extract tables:
    if image_text_combined.strip():
        text_elements.append({"type": "image_text", "text": image_text_combined.strip()})

        tables = tabula.read_pdf(pdf_path, pages=page_number + 1, multiple_tables=True)
        if tables:
            table_elements.extend(tables)

    elements = {
    # "tables": table_elements,
    # "texts": text_elements,
    "imagestexts": text_elements}
    return elements
    # return table_elements, text_elements, image_elements


def process_pdfs(base_directory):
    for file in os.listdir(base_directory):
        if file.lower().endswith('.pdf'):
            pdf_path = os.path.join(base_directory, file)
            # Create a folder for the PDF's extracted content
            output_dir = os.path.join(base_directory, file[:-4] + "_extracted")
            os.makedirs(output_dir, exist_ok=True)
            # Extract and save the content
            extract_and_save_images(pdf_path, output_dir)
            
            elements = extract_text_elements(pdf_path)
            for filename, lst in elements.items():
                # Constructing the file path to include the output directory
                file_path = os.path.join(output_dir, f"{filename}.json")
                with open(file_path, 'w') as file:
                    # Serialize the list of dictionaries to JSON and write it to the file
                    json.dump(lst, file, indent=4)
                print(f"Saved: {file_path}")


base_directory = "/home/vqa/RAG/20_manuals_extracted"
process_pdfs(base_directory)