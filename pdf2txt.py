import pdfplumber
import os

def convert_pdf_to_txt(pdf_folder, txt_folder):
    if not os.path.exists(txt_folder):
        os.makedirs(txt_folder)

    for root, _, files in os.walk(pdf_folder):
        for file in files:
            if file.endswith(".pdf"):
                pdf_path = os.path.join(root, file)
                txt_filename = os.path.splitext(file)[0] + ".txt"
                txt_path = os.path.join(txt_folder, txt_filename)

                try:
                    with pdfplumber.open(pdf_path) as pdf:
                        with open(txt_path, "w", encoding="utf-8") as txt_file:
                            for page in pdf.pages:
                                txt_file.write(page.extract_text() + "\n")
                    print(f"Converted: {pdf_path} -> {txt_path}")
                except Exception as e:
                    print(f"Failed to convert {pdf_path}: {e}")

pdf_folder = "./data_pdf"
txt_folder = "./data_txt"

convert_pdf_to_txt(pdf_folder, txt_folder)