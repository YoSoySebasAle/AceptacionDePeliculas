import math

from nltk import sent_tokenize, word_tokenize, PorterStemmer
from nltk.corpus import stopwords

f = open ('C:/Users/Alejandro Sebas/Desktop/Analisis de texto/Proyecto/AceptacionDePeliculas/PeliculasRegularesResult.txt','r')
sub = f.read()
f.close()

def FrecuenciaPalabras(archivo) -> dict:

    stopWords = set(stopwords.words("english"))
    palabras = word_tokenize(archivo)
    ps = PorterStemmer()

    tabla = dict()
    for x in palabras:
        x = ps.stem(x)
        if x in stopWords:
            continue
        if x in tabla:
            tabla[x] += 1
        else:
            tabla[x] = 1

    return tabla


table = FrecuenciaPalabras(sub)
