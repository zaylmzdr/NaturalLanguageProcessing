#fonksiyon ile vektör çarpımlarını bulmak
def dot_product(vektor1,vektor2):
    vektor_zip =zip(vektor1,vektor2)
    total = 0
    for i,j in vektor_zip:
        total += i*j
    return total


print(f"İki vektörün çarpımları sonucu:" ,dot_product([1,2,3],[5,9,2]))
#numpy ile vektör çarpımlarını bulmak

import numpy as np
import math
a = np.array([1,2,3])
b = np.array([5,9,2])

print(f"Numpy ile bulunan vektör çarpımı",np.dot(a,b))


#fonksiyon ile vektör boyunu bulmak
def vektor_size(vektor):
    total2 = 0
    for el in vektor:
        total2 += el**2
    return int(math.sqrt(total2))  #boy tam sayı


#kosinüs benzerliği için fonksiyon
def cos_similarity(v1,v2):
    vektor_size_multiply =  vektor_size(v1) * vektor_size(v2)
    similarity = dot_product(v1,v2) / vektor_size_multiply
    print(f"{v1} ve {v2}'nin kosinüs benzerliği:",similarity)

#kosinüs benzerliği
cos_similarity([2,3,5],[6,9,15])



