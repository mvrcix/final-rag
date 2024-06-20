from PIL import Image
import img2pdf
import os


def jpgs_to_pdf(jpg_files, output_pdf):
    """
    Convert a list of JPG files to a single PDF file using img2pdf.
    """
    if not jpg_files:  # Check if the list is empty
        return  # Exit the function if there are no JPG files to process
    with open(output_pdf, "wb") as f:
        f.write(img2pdf.convert(jpg_files))
        print(f"PDF created: {output_pdf}")


if __name__ == '__main__':
    base_directory = "/home/vqa/RAG/PM209/images"
    output_directory = "/home/vqa/RAG/all_merged_manuals"
    os.makedirs(output_directory, exist_ok=True)  # Ensure the output directory exists
    for root, dirs, files in os.walk(base_directory):
        for subdir in dirs:
            subdirectory_path = os.path.join(root, subdir)
            jpg_files = [os.path.join(subdirectory_path, file) for file in sorted(os.listdir(subdirectory_path)) if file.lower().endswith('.jpg')]
            if jpg_files:  # Proceed if there are JPG files
                # Define the output PDF path using the subdirectory name
                manual_name = subdirectory_path.split('/')[-2]
                output_pdf_path = os.path.join(output_directory, f"{manual_name}.pdf")
                jpgs_to_pdf(jpg_files, output_pdf_path)

