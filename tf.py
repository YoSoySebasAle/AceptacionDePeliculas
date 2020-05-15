import math

from nltk import sent_tokenize, word_tokenize, PorterStemmer
from nltk.corpus import stopwords

f = open ('C:/Users/Alejandro Sebas/Desktop/Analisis de texto/Proyecto/AceptacionDePeliculas/PeliculasRegularesResult.txt','r')
sub = f.read()
f.close()


def matrizDeFrecuencias(sentencia):
    MatrizFreq = {}
    stopWords = set(stopwords.words("english"))
    ps = PorterStemmer()

    for x in sentencia:
        frecuencias = {}
        words = word_tokenize(x)
        for word in words:
            word = word.lower()
            word = ps.stem(word)
            if word in stopWords:
                continue

            if word in frecuencias:
                frecuencias[word] += 1
            else:
                frecuencias[word] = 1

        MatrizFreq[x[:15]] = frecuencias

    return MatrizFreq

def matrizTF(frecuenciasPal):
    tf_matrix = {}

    for x, f_table in frecuenciasPal.items():
        freqTerm = {}

        palabras = len(f_table)
        for word, count in f_table.items():
            freqTerm[word] = count / palabras

        tf_matrix[x] = freqTerm

    return tf_matrix

sentencia = sent_tokenize(sub)
table = matrizDeFrecuencias(sentencia)
# print (sentencia)
# table = FrecuenciaPalabras(sentencia)
table2 = matrizTF(table)
print(table2)
