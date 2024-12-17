import PyPDF2

def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Example usage
pdf_text = extract_text_from_pdf("example.pdf")
print(pdf_text[:500])  # Print the first 500 characters
