
# PDF to Text Conversion Script

This script extracts text from PDF files and saves the extracted text in `.txt` format. The script can handle both digitally embedded PDFs and scanned PDFs using two different methods: PyMuPDF for digital PDFs and Tesseract OCR for scanned PDFs. After processing all PDFs in a specified folder, the script combines the extracted text from all the processed PDFs into a single CSV file.

## Requirements

To run this script, you will need to install the following Python libraries:

- `PyMuPDF` (also known as `fitz`)
- `Pillow` (for image processing)
- `pytesseract` (for Optical Character Recognition)
- `pandas` (for working with DataFrames)
- `glob` (for file pattern matching)

### Install Required Libraries

Use the following pip commands to install the required libraries:

```bash
pip install pymupdf Pillow pytesseract pandas
```

### Tesseract OCR Setup

If you are working with scanned PDFs, you will need to install Tesseract OCR on your machine.

**Tesseract-OCR installation instructions:**
Follow the installation guide on the official [Tesseract GitHub](https://github.com/UB-Mannheim/tesseract/wiki).

Make sure to update the `tesseract_cmd` in the script to point to your Tesseract installation path:
```python
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

## Folder Structure

- `pdf_folder`: This is the folder where all the PDFs you want to process are stored.
- `output_folder`: This is the folder where the extracted text files will be saved.

## How the Script Works

1. **PDF Processing:**
   - The script checks each PDF to determine whether it has embedded text or if it's a scanned document.
   - If the PDF has embedded text, it uses the PyMuPDF library to extract the text.
   - If the PDF is scanned (i.e., contains no embedded text), it uses Tesseract OCR to perform Optical Character Recognition (OCR) on the images within the PDF.

2. **Text Extraction and Saving:**
   - The extracted text from each PDF is saved as a `.txt` file in the output folder.
   
3. **Combining Text Files into a DataFrame:**
   - After all PDFs are processed, the script reads all the `.txt` files and combines their content into a pandas DataFrame.
   - The DataFrame is then saved as a CSV file called `afc_pdfFiles.csv` in the output folder.

## How to Use

1. Clone this repository or download the script.
2. Update the paths in the script to match your local environment:
   - `pdf_folder`: Path to your folder containing PDF files.
   - `output_folder`: Path to your folder where text files and CSV output will be saved.
3. Run the script:
   ```bash
   python script_name.py
   ```

   The script will process all PDFs in the specified folder, extract text, save the text files, and create a CSV file of all extracted text.

## Output

- `.txt` files for each PDF are saved in the `output_folder`.
- A combined CSV file (`afc_pdfFiles.csv`) is generated in the `output_folder` containing the filename and extracted text of each PDF.

## Notes

- **Scanned PDFs:** If your PDFs are scanned, Tesseract OCR will be used, which may result in varying accuracy depending on the quality of the scanned images.
- **Digital PDFs:** For PDFs with embedded text, the extraction is more reliable and faster.

