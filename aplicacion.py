"""
	Asignatura: Análisis y procesamiento digital de textos
	Grupo: 1
	Semestre: 2020-2
	
	Descripción:
		El programa realiza  la tokenizacion,  stemming y matriz de frecuencias a partir de un conjunto de documentos
    
    Recibe: El nombre de los archivos 
        Ejemplo
            C:/Users/{user}/Anaconda3/python.exe c:/Users/{user}/Documents/Textos/aplicacion.py PeliculasBuenasResult.txt,PeliculasExcelentesResult.txt,PeliculasMalasResult.txt,PeliculasRegularesResult.txt
		
"""


import sys
import glob, os
from  nltk.stem import PorterStemmer


class Tokenization:
	def __init__(self, files):
		self.files = files.split(',')
		self.dirname, self.filename = os.path.split(os.path.abspath(__file__))

	"""
        CreateTokens
        Devuelve: Diccionario 
            Clave: Nombre del archivo a tokenizar
            Valor: Lista con todos los tokens del archivo
	"""
	def CreateTokens(self):
		allTokens = {}
		for file in self.files:
			tokens = []
			actualFile = open(self.dirname+'/'+file, encoding="utf8", errors="ignore")
			linesList = actualFile.readlines()
			for line in linesList:
				for word in line.split():
					tokens.append(word)
			allTokens[file] = tokens
		return allTokens


class  Stemming:
	def __init__(self, dicTokens):
		self.dicTokens = dicTokens

	def doStemming(self):
		stemmer = PorterStemmer()
		for  key in self.dicTokens.keys():
			for  i in range(0,len(self.dicTokens[key])):
				self.dicTokens[key][i]= str(stemmer.stem(str(self.dicTokens[key][i])))
		return self.dicTokens


class FrecuenciaPalabras:
	def __init__(self,dicStemming):
		self.dicStemming = dicStemming

	def tablaFrecuencias(self):
		for key in self.dicStemming.keys():
			tabla = dict()
			for x in self.dicStemming[key]:
				if x in tabla:
					tabla[x] += 1
				else:
					tabla[x] = 1
			self.dicStemming[key] = tabla
		return self.dicStemming

	def verTabla(self):
		print("\n\nDiccionario de frecuencias")
		for key in self.dicStemming.keys():
			print(str(key),"\n",self.dicStemming[key],"\n\n\n\n")



class main:
	def prueba(self):
		tk = Tokenization("PeliculasBuenasResult.txt,PeliculasExcelentesResult.txt,PeliculasMalasResult.txt,PeliculasRegularesResult.txt")
		tokens = tk.CreateTokens()

		st = Stemming(tokens)
		datosStemming = st.doStemming()

		tb = FrecuenciaPalabras(datosStemming)
		tabla = tb.tablaFrecuencias()
		tb.verTabla()



principal = main()
principal.prueba()
