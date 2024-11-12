import os
import PyPDF2
import json

# Directory containing PDF files
pdf_dir = 'data'

# List to store text and filenames
pdf_data = []

# Process each PDF file
for filename in os.listdir(pdf_dir):
    if filename.endswith('.pdf'):
        # Open the PDF file
        with open(os.path.join(pdf_dir, filename), 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''
            for page in reader.pages:
                text += page.extract_text()

            # Append text and filename to the list
            pdf_data.append([text, filename])

# Save the list to a JSON file
with open('data/pdf_data.json', 'w') as json_file:
    json.dump(pdf_data, json_file, indent=4)
