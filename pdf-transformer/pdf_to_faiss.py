import os
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Initialize the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Directory containing PDF files
pdf_dir = 'data'

# List to store all document vectors
all_vectors = []

# Process each PDF file
for filename in os.listdir(pdf_dir):
    if filename.endswith('.pdf'):
        # Open the PDF file
        with open(os.path.join(pdf_dir, filename), 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''
            for page in reader.pages:
                text += page.extract_text()

            # Convert text to vector
            vector = model.encode(text)
            all_vectors.append(vector)

# Convert list of vectors to a numpy array
all_vectors = np.array(all_vectors)

# Create a FAISS index
dimension = all_vectors.shape[1]
index = faiss.IndexFlatL2(dimension)

# Add vectors to the index
index.add(all_vectors)

# Save the index
faiss.write_index(index, 'faiss_index.idx')
