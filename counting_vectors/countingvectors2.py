import numpy as np

dosya_yollari = [
    "data/d1.txt",
    "data/d2.txt",
    "data/d3.txt"
]

tokens_index_dict = {}
docs = []
tokens_index = 0

for dosya in dosya_yollari:
    with open(dosya,'r') as f:
        icerik = f.readline()
        tokens = icerik.split(" ")
        docs.append(tokens)
        for token in tokens:
            if(token not in tokens_index_dict):
                tokens_index_dict[token] = tokens_index
                tokens_index+=1



matrix = np.zeros((len(dosya_yollari),len(tokens_index_dict)))


for i, doc in enumerate(docs):
    for token in doc:
        matrix[i][tokens_index_dict[token]] +=1



#benzerlik matrisi oluşturma
def cos_similarity(v1,v2):
    return np.dot(v1,v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

num_docs = len(dosya_yollari)
similarities = np.zeros((num_docs,num_docs))


for i in range(num_docs):
    for j in range(num_docs):
        similarities[i, j] = cos_similarity(matrix[i], matrix[j])

print(similarities)

def en_benzeri_hangisi(doc_index,dosya_yollari):
    satir = similarities[doc_index]
    en_benzer_index = np.argmax(satir)
    return dosya_yollari[en_benzer_index]


print(en_benzeri_hangisi(0,dosya_yollari))






