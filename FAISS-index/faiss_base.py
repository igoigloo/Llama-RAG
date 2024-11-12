import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

data = [
['Throw my cellphone in the water', 'random'],
['Network Access Control?', 'networking'],
['Address', 'location'],
['Where are your headquarters located?', 'location']]

df = pd.DataFrame(data, columns=['text', 'category'])

text = df['text']
encoder = SentenceTransformer("paraphrase-mpnet-base-v2")
vectors = encoder.encode(text)
# print(vectors)

#Build a FAISS index from the vectors
vector_dimension = vectors.shape[1]
index = faiss.IndexFlatL2(vector_dimension)
faiss.normalize_L2(vectors)
index.add(vectors)

# creating search vectors
search_text = 'where is your office?'
search_vector = encoder.encode(search_text)
_vector = np.array([search_vector])
faiss.normalize_L2(_vector)

#search
k = index.ntotal
distances, ann = index.search(_vector, k=k)
# print(k)

results = pd.DataFrame({'distance': distances[0], 'ann': ann[0]})
# print(results)

merge = pd.merge(results, df, left_on='ann', right_index=True)
# print(merge)

labels = df['category']
category = labels[ann[0][0]]
print(category)
