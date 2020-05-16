"""
	Asignatura: Análisis y procesamiento digital de textos
	Grupo: 1
	Semestre: 2020-2

	Descripción:
		Esta clase realiza la tokenizacion de un conjunto de archivos dados

    Recibe: El nombre de los archivos a tokenizar
        Ejemplo
            C:/Users/Ivan/Anaconda3/python.exe c:/Users/Ivan/Documents/Textos/Tokenization.py PeliculasBuenasResult.txt,PeliculasExcelentesResult.txt,PeliculasMalasResult.txt,PeliculasRegularesResult.txt

"""

import sys
import glob, os
import math

from nltk import sent_tokenize, word_tokenize, PorterStemmer
from nltk.corpus import stopwords

class Tokenization:

	def __init__(self, files):
		self.files = files.split(',')
		self.dirname, self.filename = os.path.split(os.path.abspath(__file__))
	#
    # '''
    #     CreateTokens
    #     Devuelve: Diccionario
    #         Clave: Nombre del archivo a tokenizar
    #         Valor: Lista con todos los tokens del archivo
    # '''

	def CreateTokens(self):
		allTokens = {}
		for file in self.files:
			tokens = []
			actualFile = open(self.dirname+'/'+file, encoding="utf8", errors="ignore")
			linesList = actualFile.readlines()
			allSubtitlesInMovie = ""
			i  = 0
			for line in linesList:
				i+=1
				line = line.replace("\n","")
				if((line == "-----NewMovie-----" and len(allSubtitlesInMovie)>0) or (i == len(linesList))):
					if (i == len(linesList)):
						allSubtitlesInMovie = allSubtitlesInMovie + line + " "
					tokens.append(allSubtitlesInMovie)
					allSubtitlesInMovie = ""
				elif(line != "-----NewMovie-----"):
					allSubtitlesInMovie = allSubtitlesInMovie + line + " "
			allTokens[file] = tokens
		return allTokens

	def verTabla(self,diccionario):
		print("\n\nTOKENS")
		for key in diccionario.keys():
			# print(str(key),"\n")
			a = diccionario[key]
			# print("Cantidad de peliculas de la categoria: ", len(diccionario[key]),"\n\n\n\n\n\n")
		return a

	def matrizDeFrecuencias(self,docencia):
	    MatrizFreq = {}
	    stopWords = set(stopwords.words("english"))
	    ps = PorterStemmer()

	    for x in docencia:
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

	def matrizTF(self, frecuenciasPal):
	    tf_matriz = {}

	    for x, f_table in frecuenciasPal.items():
	        freqTerm = {}

	        palabras = len(f_table)
	        for word, count in f_table.items():
	            freqTerm[word] = count / palabras

	        tf_matriz[x] = freqTerm

	    return tf_matriz

	def PalabrasPorDocumentos(self, frecuenciasPal):
		palPorDoc = {}
		for doc, f_table in frecuenciasPal.items():
			for word, count in f_table.items():
				if word in palPorDoc:
					palPorDoc[word] += 1
				else:
					palPorDoc[word] = 1

		return palPorDoc

	def matrizIDF(self, frecuenciasPal, palPorDoc, total):
		idf_matriz = {}
		for doc, f_table in frecuenciasPal.items():
			idf_table = {}
			for word in f_table.keys():
				idf_table[word] = math.log10(total / float(palPorDoc[word]))

			idf_matriz[doc] = idf_table
		return idf_matriz

	def matrizTFIDF(self, tf_matriz, idf_matriz):
		tf_idf_matriz = {}
		for (doc2, f_table1), (doc2, f_table2) in zip(tf_matriz.items(), idf_matriz.items()):
			tf_idf_table = {}
			for (word1, valor1), (word2, valor2) in zip(f_table1.items(), f_table2.items()):
				tf_idf_table[word1] = float(valor1 * valor2)
			tf_idf_matriz[doc2] = tf_idf_table

		return tf_idf_matriz



tk = Tokenization("PeliculasExcelentesResult.txt")
tokens = tk.CreateTokens()
table = tk.matrizDeFrecuencias(tk.verTabla(tokens))
tfTabla = tk.matrizTF(table)
tabla3 = tk.PalabrasPorDocumentos(table)
idfTabla = tk.matrizIDF(table, tabla3, len(tk.verTabla(tokens)))
tfidfTabla = tk.matrizTFIDF(tfTabla, idfTabla)
print(tfidfTabla)
