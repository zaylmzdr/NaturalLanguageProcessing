import numpy as np
sentences = [ "Kediler çok sevimli hayvanlardır.",
               "Kediler evde beslenen en popüler evcil hayvanlardandır." ,
               "Kuşlar ve balıklar, insanlar tarafından sıkça evcil hayvan olarak tercih edilir.",
              "Kediler sevimli hayvalardır",
              "Kuşlar ve balıklar, insanlar tarafından evcil hayvan olarak tercih edilir."]


token_index_dict ={}
token_index = 0;
docs = []
for sentence in sentences:
    tokens = sentence.split(" ")
    docs.append(tokens)
    for token in tokens:
        if(token not in token_index_dict):
            token_index_dict[token] = token_index
            token_index +=1


print("Token index dictionary: ",token_index_dict)

counting_vectors = np.zeros((len(sentences),len(token_index_dict)))

for sentence_index , tokens in enumerate(docs):
    for token in tokens:
        counting_vectors[sentence_index][token_index_dict[token]] +=1

#cos similarity function
def cos_similarity(v1,v2):
    return np.dot(v1,v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

similarities = np.zeros((len(sentences),len(sentences)))


#print(similarities)

for i in range(len(sentences)):
    for j in range(len(sentences)):
        similarities[i,j] = cos_similarity(counting_vectors[i],counting_vectors[j])

print("Similarity matrix:\n",similarities)

def en_benzeri(sentence_index,sentences):
    satir = similarities[sentence_index]
    satir[sentence_index] = 0
    en_benzer_index = np.argmax(satir)
    return sentences[en_benzer_index]

print("\nEn benzer cümle:")
print(en_benzeri(2,sentences))

