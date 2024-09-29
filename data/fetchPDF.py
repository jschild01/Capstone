#%%
import os
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from io import BytesIO

'''
terresact-ocr installation instructions:
https://github.com/UB-Mannheim/tesseract/wiki
'''

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Specify the folder containing PDF files
pdf_folder = r'C:\Users\schil\OneDrive\Desktop\School\Capstone\LOC\pdf'
output_folder = r'C:\Users\schil\OneDrive\Desktop\School\Capstone\LOC\pdf\textConversion'

# Function to save extracted text to a file
def save_text_to_file(pdf_file, text):
    filename = os.path.basename(pdf_file).replace('.pdf', '.txt')
    output_path = os.path.join(output_folder, filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)

# Function to extract text using PyMuPDF for digital PDFs
def extract_text_from_pdf_pymupdf(pdf_file):
    doc = fitz.open(pdf_file)
    full_text = ''
    for page in doc:
        full_text += page.get_text("text") or ''
    return full_text.strip()

# Function to extract text using Tesseract OCR for scanned PDFs
def extract_text_using_ocr_pymupdf(pdf_file):
    doc = fitz.open(pdf_file)
    text = ''
    for page_num in range(len(doc)):
        # Extract image from the page
        pix = doc.load_page(page_num).get_pixmap()
        img = Image.open(BytesIO(pix.tobytes("png")))
        # Perform OCR
        text += pytesseract.image_to_string(img)
    return text

# Function to check if a PDF has embedded text
def is_pdf_scanned_pymupdf(pdf_file):
    doc = fitz.open(pdf_file)
    for page in doc:
        page_text = page.get_text("text")
        if page_text.strip():  # If there's text, it's not scanned
            return False
    return True

# Function to process all PDFs in the folder
def process_pdfs(pdf_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(pdf_folder):
        if filename.lower().endswith('.pdf'):
            pdf_file = os.path.join(pdf_folder, filename)
            print(f'Processing {filename}...')

            # Check if the PDF is scanned
            if is_pdf_scanned_pymupdf(pdf_file):
                print(f'  {filename} seems to be a scanned document. Using OCR...')
                extracted_text = extract_text_using_ocr_pymupdf(pdf_file)
            else:
                print(f'  {filename} contains embedded text. Using PDF parser...')
                extracted_text = extract_text_from_pdf_pymupdf(pdf_file)

            # Save the extracted text to a file
            save_text_to_file(pdf_file, extracted_text)

    print('Processing complete.')

# Run the script
process_pdfs(pdf_folder)
















