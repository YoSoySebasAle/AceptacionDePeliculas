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
import pandas as pd
from nltk import sent_tokenize, word_tokenize, PorterStemmer
from nltk.corpus import stopwords

class Texto:

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
		for (doc1, f_table1), (doc2, f_table2) in zip(tf_matriz.items(), idf_matriz.items()):
			tf_idf_table = {}
			for (word1, valor1), (word2, valor2) in zip(f_table1.items(), f_table2.items()):
				tf_idf_table[word1] = float(valor1 * valor2)
			tf_idf_matriz[doc1] = tf_idf_table

		nombreDic = list(tf_idf_matriz)
		nombreDic = list(set(nombreDic))
		#print("Nombre dic: ", nombreDic)

		#print("\nMatriz: ",tf_idf_matriz)

		tf_idf_matriz2 ={}
		for x in  nombreDic:
			tf_idf_matriz2 = dict(tf_idf_matriz2, **tf_idf_matriz[x])
		#	tf_idf_matriz2 = tf_idf_matriz2 + dict(tf_idf_matriz[nombreDic[0]])

		print("\n\n\nMATRIX 2 FINAL: ", tf_idf_matriz2)
		print("LEN MATRIX 2: ", len(tf_idf_matriz2))
		return tf_idf_matriz2

	def matrizTFIDF2(self, tf_matriz, idf_matriz):
		nombreDic = list(tf_matriz) + list(idf_matriz)
		nombreDic = list(set(nombreDic))
		#print("FILAS: ",nombreDic)
		tf_matriz = tf_matriz[nombreDic[0]]
		idf_matriz = idf_matriz[nombreDic[0]]

		print("TF: ",tf_matriz)
		print("IDF: ",idf_matriz)

		filas = list(tf_matriz.keys()) + list(idf_matriz.keys())
		filas = list(set(filas))

		print("FILAS: ",filas)


		tf_idf_matriz = {}
		for fila in filas:
			tf_idf_matriz[fila] = float(float(tf_matriz[fila])*float(idf_matriz[fila]))
		return tf_idf_matriz




class TFIDF_global:
	def tablaDataFrame(self,bolsaA,bolsaB, bolsaC, bolsaD):
		#Se obtienen valores únicos en las filas
		filas = list(bolsaA.keys()) + list(bolsaB.keys()) +  list(bolsaC.keys()) +  list(bolsaD.keys())
		filas = list(set(filas))
		filas.sort()

		#listas para cada idioma de coincidencias
		filasMalas = []
		filasRegulares = []
		filasBuenas = []
		filasExcelentes = []

		#Se itera sobre cada fila para saber si hay coincidencias
		for fila in filas:

			if fila in bolsaA.keys():
				filasMalas.append(bolsaA[fila])
			else:
				filasMalas.append(0)


			if fila in bolsaB.keys():
				filasRegulares.append(bolsaB[fila])
			else:
				filasRegulares.append(0)


			if fila in bolsaC.keys():
				filasBuenas.append(bolsaC[fila])
			else:
				filasBuenas.append(0)



			if fila in bolsaD.keys():
				filasExcelentes.append(bolsaD[fila])
			else:
				filasExcelentes.append(0)

		tabla = {'Malas': filasMalas, 'Regulares':filasRegulares, 'Buenas':filasBuenas, 'Excelentes':filasExcelentes }
		tabla = pd.DataFrame(tabla)
		tabla.index = filas
		return tabla

	def guardarTablaTFIDF(self, tabla,nombre):
		guardaArc = tabla.to_csv(str(nombre + ".csv"), index = True, header = True)






class main:
	archivos = ["PeliculasMalasResult.txt","PeliculasRegularesResult.txt", "PeliculasBuenasResult.txt","PeliculasExcelentesResult.txt"]
	#archivos = ["PeliculasExcelentesResult.txt"]
	columnasDFIDF = []
	for archivo in archivos:
		tk = Texto(archivo)
		tokens = tk.CreateTokens()
		table = tk.matrizDeFrecuencias(tk.verTabla(tokens))
		tfTabla = tk.matrizTF(table)
		tabla3 = tk.PalabrasPorDocumentos(table)
		#print("FRECUENCIAS: ",tabla3)
		idfTabla = tk.matrizIDF(table, tabla3, len(tk.verTabla(tokens)))

		tfidfTabla= tk.matrizTFIDF(tfTabla, idfTabla)

		columnasDFIDF.append(tfidfTabla)



	print("\n\n\n\n")
	print(columnasDFIDF[0])
	print("\n\n\n\n")
	print(columnasDFIDF[1])
	print("\n\n\n\n")
	print(columnasDFIDF[2])
	print("\n\n\n\n")
	print(columnasDFIDF[3])


	tabla = TFIDF_global()
	columnas = tabla.tablaDataFrame(columnasDFIDF[0],columnasDFIDF[1],columnasDFIDF[2],columnasDFIDF[3])
	tabla.guardarTablaTFIDF(columnas, "tablaTFIDF")


main()



"""tk = Tokenization("PeliculasMalasResult.txt")
tokens = tk.CreateTokens()
table = tk.matrizDeFrecuencias(tk.verTabla(tokens))
tfTabla = tk.matrizTF(table)
tabla3 = tk.PalabrasPorDocumentos(table)
idfTabla = tk.matrizIDF(table, tabla3, len(tk.verTabla(tokens)))
tfidfTabla = tk.matrizTFIDF(tfTabla, idfTabla)
print(tfidfTabla)
print("LEN: ", len(tfidfTabla))
"""
