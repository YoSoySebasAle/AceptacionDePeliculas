import pandas as pd
import math
import glob, os
import sys
from nltk import sent_tokenize, word_tokenize, PorterStemmer
from nltk.corpus import stopwords
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.neighbors import KNeighborsClassifier




class  limpiadorTexto:
	def __init__(self):
		"""
			Constructor para la clase Limpiador de texto
		"""
		self.dirname = ''


	def FindFiles(self,extension,carpetas):
		"""
		El método FindFiles encuentra todos los archivos con una extensión en específica en una carpeta.

		Los parámetros que recibe son:
				-Extension [string]: La extensión de los archivos que se desean buscar y limpiar
				-carpetas: [string]: Es el nombre de las carpetas donde se van a buscar los archivos. Van separadas por coma
							y sin espacios
		Parametros de saldia:
				-diccionario con la siguiente estructura:
						filesAllCategories = {
												carpeta1: [nombreArchivo1, nombreArchivo2, nombreArchivo3...]
												carpeta2: [nombreArchivo1, nombreArchivo2, nombreArchivo3...]
												...
						}
		"""

		filesAllCategories = {}
		folders = carpetas.split(",")
		dirname, filename = os.path.split(os.path.abspath(__file__))
		self.dirname = dirname
		for folder in folders:
			files = []
			os.chdir(dirname+"/"+folder)
			for file in glob.glob("*."+extension):
				files.append(file)
			filesAllCategories[folder] = files
		return filesAllCategories


	def ProcessFiles(self,filesAllCategories):
		"""
			Este método hace la limpieza de todos los archivos de una carpeta (solo deja letras, elimita todo lo demás)
			y crea UN SOLO documento donde las guarda a todas. Para diferenciar entre una película y otra se añade la
			leyenda "-----NewMovie-----".

			Recibe:
				filesAllCategories [diccionario]:  es un diccionario con la siguiente estructura
								filesAllCategories = {
																carpeta1: [nombreArchivo1, nombreArchivo2, nombreArchivo3...]
																carpeta2: [nombreArchivo1, nombreArchivo2, nombreArchivo3...]
																...
										}
			Salida
				Archivo .txt: Archivo de texto plano con todas las películas. Se guarda en la misma carpeta donde está almacenado
								este programa ejecutable. El nombre se construye con el nombre de la carpeta + "result.txt"
		"""

		dirname, filename = os.path.split(os.path.abspath(__file__))
		dirname = self.dirname
		for key in filesAllCategories:
			os.chdir(dirname+"/"+key)

			for fileName in filesAllCategories[key]:
				#print(dirname+"/"+key+"/"+fileName)
				actualFile = open(dirname+"/"+key+"/"+fileName, encoding="utf8", errors="ignore")
				linesList = actualFile.readlines()
				linesForMovie = []
				linesForMovie.append("-----NewMovie-----")
				for line in linesList:
					line = line.lower()
					line = line.rstrip()
					line = line.replace('<i>','')
					line = line.replace('</i>','')
					line = line.replace('<b>','')
					line = line.replace('</b>','')
					line = line.replace('<font>','')
					line = line.replace('</font>','')
					line = line.replace('♪','')
					line = line.replace('- ','')
					line = line.replace('"','')
					line = line.replace('..','')
					line = line.replace('.','')
					line = line.replace('\'s','')
					line = line.replace(',','')
					line = line.replace('?','')
					line = line.replace('(','')
					line = line.replace(')','')
					line = line.replace('!','')
					line = line.replace(':','')
					line = line.replace('1','')
					line = line.replace('[','')
					line = line.replace(']','')
					line = line.replace('\'','')
					for caracter in line: #quita caracteres que no sean A-Z o a-z
						if ord(caracter) in range(34,39) or ord(caracter) in range(40,63) or ord(caracter) in range(64,64) or ord(caracter) in range(91,96) or ord(caracter) in range(123,161) or ord(caracter) in range(162,255):
							line = line.replace(caracter, '')
					line = line.strip()
					if not line.isdigit() and not '-->' in line:
						linesForMovie.append(line)

				path=dirname + '/' + key + "Result.txt"
				with open(path, 'a+', encoding="utf8", errors="ignore") as f:
					for item in linesForMovie:
						if item is not '':
							f.write("%s\n" % item)
		dirname = self.dirname
		os.chdir(dirname)



	def limpiarPeliculaIndividual(self, ruta, nombrePelicula):
		"""
			El método hace la limpieza de un solo archivo a la vez (elimina todo caracter que no sea una letra) y lo guarda
			en la misma carpeta donde está  almacenado esta aplicación.

			Entrada:
				ruta [string]: Es la ruta donde está guardado el archivo .srt que se desea procesar
				nombrePelicula [string]: Nombre del archivo .srt

			Salida:
				Archivo [.txt]: Archivo de texto plano que contiene todos los subtitulos limpios

		"""
		dirname = ruta
		actualFile = open(dirname+"/"+nombrePelicula, encoding="utf8", errors="ignore")
		linesList = actualFile.readlines()
		linesForMovie = []
		for line in linesList:
			line = line.lower()
			line = line.rstrip()
			line = line.replace('<i>','')
			line = line.replace('</i>','')
			line = line.replace('<b>','')
			line = line.replace('</b>','')
			line = line.replace('<font>','')
			line = line.replace('</font>','')
			line = line.replace('♪','')
			line = line.replace('- ','')
			line = line.replace('"','')
			line = line.replace('..','')
			line = line.replace('.','')
			line = line.replace('\'s','')
			line = line.replace(',','')
			line = line.replace('?','')
			line = line.replace('(','')
			line = line.replace(')','')
			line = line.replace('!','')
			line = line.replace(':','')
			line = line.replace('1','')
			line = line.replace('[','')
			line = line.replace(']','')
			line = line.replace('\'','')
			for caracter in line: #quita caracteres que no sean A-Z o a-z
				if ord(caracter) in range(34,39) or ord(caracter) in range(40,63) or ord(caracter) in range(64,64) or ord(caracter) in range(91,96) or ord(caracter) in range(123,161) or ord(caracter) in range(162,255):
					line = line.replace(caracter, '')
			line = line.strip()
			if not line.isdigit() and not '-->' in line:
				linesForMovie.append(line)

			path=dirname+"/"+nombrePelicula + ".txt"
			with open(path, 'w', encoding="utf8", errors="ignore") as f:
				for item in linesForMovie:
					if item is not '':
						f.write("%s\n" % item)



class Texto:
	"""
		La clase texto contiene los métodos necesarios para realizar el tokenizado, stemming, matriz TF,
			matriz IDF y la matriz TF-IDF de un conjunto de datos provenientes de un archivo .txt.
			Cabe destacar que para utilizar  un objeto de esta clase de forma adecuada, es obligatorio
			haber preprocesado el archivo para eliminar cualquier símbolo que no sea una letra, para ello utilice
			la clase limpiarTexto
	"""

	def __init__(self, files):
		"""
			Constructor de la clase Texto
			Entrada:
				files: Es el nombre del archivo o archivos a tokenizar. No debe haber espacios y se separan por comas.
						Por ejemplo: archivo1,archivo2,archivo3...
		"""
		self.files = files.split(',')
		self.dirname, self.filename = os.path.split(os.path.abspath(__file__))


	def CreateTokens(self):
		"""
			Aunque el método lleva por título "Crear tokens" lo que en realidad hace es  generar un diccionario
			en donde las llaves son los nombres de los archivos  y los valores es una lista con el texto completo de la película
			Recibe: -
			Salida: Diccionario de diccionarios, que siguen la siguiente estructura:
					Salida = {
						archivo1 = [subtitulo completo de la pelicula...]
						archivo2 = [subtitulo completo de la pelicula...]
						archivo3 = [subtitulo completo de la pelicula...]
						.
						.
						.
					}
		"""
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
				#print(line,i)
				if (i==1):
					line = line.replace("-----NewMovie-----","")
				if((line == "-----NewMovie-----" and len(allSubtitlesInMovie)>0) or (i == len(linesList))):
					if (i == len(linesList)):
						allSubtitlesInMovie = allSubtitlesInMovie + line + " "
					tokens.append(allSubtitlesInMovie)

					allSubtitlesInMovie = ""
				elif(line != "-----NewMovie-----"):
					allSubtitlesInMovie = allSubtitlesInMovie + line + " "
			allTokens[file] = tokens
		return allTokens

	def desempaquetaDiccionario(self,diccionario):
		"""
			Esta función obtiene  la lista donde se encuentra todo el texto de las películas, es un auxiliar para
			el método CreateTokens().
			Gráficamente hace esto:
			ResultadoDeTokens = {
						archivo1 = [texto completo de la pelicula...]
					}

			Salida = [texto completo de la pelicula...]

		"""
		for key in diccionario.keys():
			a = diccionario[key]
		return a

	def matrizDeFrecuencias(self,docencia):
		"""
			El método hace todo el proceso para obtener la matriz de Frecuencias de un documento. En el proceso
			obtiene las stopWords,  tokeniza el documento  y  a través del algoritmo de Porter, aplica el stemming
			a cada una de las palabras para que posteriormente  se guarde un diccionario con todas las palabras y
			frecuencias con la estructura: llave = palabra y valor = frecuencia.

			ENTRADA:
				docencia: Son los subtitulos de la película en un solo string

			Salida:
				Diccionario con la matriz de frecuencias, en donde las llaves son las palabras de cada uno de los textos
				y los valores son las frecuencias
		"""
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
		"""
			El método obtiene la matriz TF de un documento. Para ello utiliza un diccionario con las frecuencias
			de las palabras y divide  ese valor por la cantidad total de palabras en el documento.
			Entrada:
				frecuenciasPal [Diccionario]: Diccionario con las palabras y su frecuencia
			Salida:
				Diccionario con las frecuencias de término de cada una de las palabras del  documento

		"""
		tf_matriz = {}

		for x, f_table in frecuenciasPal.items():
			freqTerm = {}

			palabras = len(f_table)
			for word, count in f_table.items():
				freqTerm[word] = count / palabras

			tf_matriz[x] = freqTerm

		return tf_matriz

	def PalabrasPorDocumentos(self, frecuenciasPal):
		"""
			Este método cuenta la cantidad de veces que se repite una palabra en todos los documentos de la colección.
			Entrada:
				frecuenciasPal [Diccionario]: Diccionario con todas las palabras del documento y sus frecuencias
			Salida:
				Diccionario con cada de una de las palabras como llave y como valor el número de veces que aparece en todos
					los documentos
		"""
		palPorDoc = {}
		for doc, f_table in frecuenciasPal.items():
			for word, count in f_table.items():
				if word in palPorDoc:
					palPorDoc[word] += 1
				else:
					palPorDoc[word] = 1

		return palPorDoc

	def matrizIDF(self, frecuenciasPal, palPorDoc, total):
		"""
			Calcula la matriz IDF, para ello aplica la formula:
											idf_palabra = (TotalDeDocumentos/NumVecesQueAparece)+1
			El "+1" es un factor de corrección.

			ENTRADA:
				frecuenciasPal: [Diccionario] con todas las palabras del documento y sus frecuencias
				palPorDoc  [Diccionario] con todas las palabras del documento y el número de veces que aparece en todos lso documentos
				total= Número de documentos  de esa categoría

			SALIDA:
				Diccionario con la matriz IDF. Las llaves son las palabras y los valores son los resultados de IDF
		"""
		idf_matriz = {}
		for doc, f_table in frecuenciasPal.items():
			idf_table = {}
			for word in f_table.keys():
				#se añadioo el factor correctivo
				idf_table[word] = math.log(total / float(palPorDoc[word]))+1

			idf_matriz[doc] = idf_table
		return idf_matriz

	def matrizTFIDF(self, tf_matriz, idf_matriz):
		"""
			El método calcula la matriz de TF-IDF a través de la multiplicación  de cada una de las palabras.
			La fórmula es TFIDF_palabra = TF_palabra*IDF_palabra
			ENTRADA:
				tf_matriz [Diccionario] con la matriz TF
				idf_matriz [Diccionario] con la matriz TF
			SALIDA:
				Diccionario con la matriz TF_IDF. Las llaves son las palabras y los valores son los TFIDF calculados
				para esa palabra
		"""
		tf_idf_matriz = {}
		for (doc1, f_table1), (doc2, f_table2) in zip(tf_matriz.items(), idf_matriz.items()):
			tf_idf_table = {}
			for (word1, valor1), (word2, valor2) in zip(f_table1.items(), f_table2.items()):
				tf_idf_table[word1] = float(valor1 * valor2)
			tf_idf_matriz[doc1] = tf_idf_table
		nombreDic = list(tf_idf_matriz)
		nombreDic = list(set(nombreDic))
		tf_idf_matriz2 ={}
		for x in  nombreDic:
			tf_idf_matriz2 = dict(tf_idf_matriz2, **tf_idf_matriz[x])

		return tf_idf_matriz2




class TFIDF_global:
	"""
		Esta clase junta las 4 tablas TF-IDF generadas para cada una de las categorías de la clasificación (malas,
		regulares, buenas, excelentes) y genera un archivo .csv  que posteriormente guarda para poder utilizarlo en
		los algoritmos
	"""


	def tablaDataFrame(self,bolsaA,bolsaB, bolsaC, bolsaD,transpuesta, cargaFilas):
		"""
			El método genera el dataframe con las 4 tablas TF-IDF  creadas para cada una de las categorías de la
			clasificación.

			RECIBE
				bolsaA: [Diccionario]  con la tabla TF-IDF de la categoría Malas
				bolsaB: [Diccionario]  con la tabla TF-IDF de la categoría Regulares
				bolsaC: [Diccionario]  con la tabla TF-IDF de la categoría Buenas
				bolsaD: [Diccionario]  con la tabla TF-IDF de la categoría Excelentes
				transpuesta [BOOLEANO]: utilizar True si se desea transponer la matriz resultante, False si no.
				cargaFilas [DICCIONARIO]: Si son conocidas las filas de la matriz TF-IDF usada para entrenar el sistema,
											se pueden cargar directamente con este parámetro. Esto sirve para que cuando
											se evalúe una película, tenga todas las filas con las que se entrenó el sistema
											y únicamente se llenen los valores que se usen, de lo contrario la tabla TF-IDF
											de una película no tendrá la misma cantidad de filas que la del sistema completo
											y provocaría un error al ejecutar los algoritmos.
			SALIDA
				[Dataframe de Pandas]

		"""
		if cargaFilas == None:
			filas = list(bolsaA.keys()) + list(bolsaB.keys()) +  list(bolsaC.keys()) +  list(bolsaD.keys())
			filas = list(set(filas))
		else:
			filas = cargaFilas
		filas.sort()
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
		if transpuesta == True:
			tabla = tabla.transpose()
		return tabla

	def guardarTablaTFIDF(self, tabla,nombre):
		"""
			Guarda el dataframe de pandas generado con el método  tablaDataFrame().
			ENTRADA:
				tabla [DATAFRAME de Pandas]
				nombre [STRING]: Nombre con el que se guardará la tabla. Debe colocar la extensión .csv también
		"""
		guardaArc = tabla.to_csv(str(nombre), index = True, header = True)




class Tokenization:
	def __init__(self, ruta,archivo):
		"""
			Constructor de la clase
				ENTRADA:
					ruta: Ruta del archivo a tokenizar
					archivo: Nombre del archivo a tokenizar
		"""
		self.dirname, self.filename = os.path.split(os.path.abspath(__file__))
		self.ruta = ruta
		self.archivo = archivo


	def CreateTokens(self):
		"""
			CreateTokens
			Devuelve: Diccionario
			Clave: Nombre del archivo a tokenizar
			Valor: Lista con todos los tokens del archivo
		"""
		allTokens = {}
		tokens = []
		actualFile = open(self.ruta+'/'+self.archivo, encoding="utf8", errors="ignore")
		linesList = actualFile.readlines()
		for line in linesList:
			for word in line.split():
				tokens.append(word)
		allTokens[self.archivo] = tokens
		return allTokens


class  Stemming:
	def __init__(self, dicTokens):
		"""
			Constructor de la clase
				Entrada:
					dicTokens [Diccionario] con los tokens de la película
		"""
		self.dicTokens = dicTokens

	def doStemming(self):
		"""
			Hace el stemming de cada palabra del  un diccionario a través del algoritmo PoterStemmer

			Salida: [Diccionario] de los tokens ya con stemming
		"""
		stemmer = PorterStemmer()
		for  key in self.dicTokens.keys():
			for  i in range(0,len(self.dicTokens[key])):
				self.dicTokens[key][i]= str(stemmer.stem(str(self.dicTokens[key][i])))
		return self.dicTokens













""""
	La siguiente sección tiene distintos clasificadores  que fueron construidos por nosotros o
	a través de bibliotecas para encontar el que mejor se adapta a nuestras necesidades

"""


class NaiveBayesClassifierWithTfIdf:
	"""
	CLASIFICADOR #1. Bayes
		Este clasificador es de nuestra autoría. Hace uso de la matriz TF-IDF para calcular la probabilidad
		de que una película será mala, regular,  buena o excelente.
	"""
	def __init__(self):
		"""
		Clasificador Bayesiano usando la Matriz TF-IDF
		Atributos:
			tf_idf_matrix: Atributo dónde se guarda el dataframe
			con una matriz TF-IDF, se inicializa este atributo con
			el método read_CSV_toObtainTfIdfMatrix.
			priors: Lista con las probabilidades Prior de cada clase.
			Se inicializa con el método buildPriors.
			VocabularyCount: Tamaño del Vocabulario, este se calcula
			cuando se iniciliza el tf_idf_matrix dentro del método
			read_CSV_toObtainTfIdfMatrix.
			probabiliesWordGivenClass: DataFrame con las probabilidades
			condicionales del clasificador
			tfIdfSumOfClasses: suma del TF-IDF de todas las palabras de la
			clase.
		"""
		self.tf_idf_matrix = None
		self.priors = None
		self.VocabularyCount = None
		self.probabiliesWordGivenClass = None
		self.tfIdfSumOfClasses = {}

	def read_CSV_toObtainTfIdfMatrix(self, filePath, indexBy):
		"""
		Función para crear un DataFrame que contiene la Matriz TF-IDF
		y a partir de ese data frame saber el tamaño del Vocabulario.
		Args:
			filePath: Dirección del archivo a leer
			indexBy: Columna por la que se va indexar el DataFrame, debe
			ser el nombre de la columna dónde se encuentran las palabras.
		"""
		# creación del dataframe
		self.tf_idf_matrix = pd.read_csv(filePath, delimiter=",", index_col=indexBy)

		for label in self.tf_idf_matrix.columns:
			self.tfIdfSumOfClasses[label] = self.tf_idf_matrix[label].sum()
		# Vocabulary count
		self.VocabularyCount = self.tf_idf_matrix.shape[0]

	def buildPriors(self, OcurrencesOfClasses, documentsCount):
		"""
		Función para la construcción de las Probabilidades prior para
		el clasificador.
		Recordemos que esta probabilidad está definida como:
			P(clase) = Número de documentos en la que aparece la clase /
			Número de documentos Totales
		Args:
			OcurrencesOfClasses: Lista con el número de documentos en la que aparece
			las clases.
			documentsCount: Número total de Documentos.
		"""
		self.priors = []
		for classOcurrence in OcurrencesOfClasses:
			priorProbability = classOcurrence / documentsCount
			self.priors.append(priorProbability)

	def get_TF_IDF_of_a_Word_By_label(self, word, label):
		"""
		Función que permite obtener el TF-IDF de una palabra deacuerdo a su clase
		Args:
			word: palabra de la que se desea saber su TF-IDF
			label: clase de la palabra
		Returns:
			Valor TF-IDF  de la palabra
		"""
		tfIdfValue = self.tf_idf_matrix.loc[[word], [label]]
		return (tfIdfValue.values[0])[0]

	def calculateProbabilityOfWordForEachClass(self, word, label, TfIdfSumOfclass):
		"""
		Función para obtener la probabilidad condicional
		de una palabra dada la clase.
		Recordando, la probabilidad está definida cómo:

			P(w | c) = (TF-IDF(w, c) + 1) / (TF-IDF(c) + |V|)

			dónde:
				TF-IDF(w,c) : TF-IDF de la palabra en la clase
				TF-IDF(c) : suma de los TF-IDF de las palabras de la clase
				|V| tamaño de Vocabulario
		Args:
			word: palabra de la que se desea obtener la probabilidad condicional.
			label: clase de la cual se desea saber la probabilidad condicional.
			TfIdfSumOfclass: suma de los TF-IDF de las palabras de la clase
		Returns:
			Probabilidaad de la palabra dada la clase.
		"""

		if word not in self.tf_idf_matrix.index:
			numerator = 1
		else:
			numerator = (self.get_TF_IDF_of_a_Word_By_label(word, label) + 1)

		# print(numerator)
		denominator = (TfIdfSumOfclass + self.VocabularyCount)
		return math.log(numerator) - math.log(denominator)

	def buildClassifier(self):
		"""
		Construcción de una Matriz con las probabilidades condicionales
		de las Palabras, se exporta a un CSV para que no tengamos
		que calcular la matriz cada vez que la quisieramos usar, además
		de inicializar el atributo probabiliesWordGivenClass si se desea
		usar la matriz en Memoria.

		La ruta a la que se exporta el archivo es "./Probabilidades.csv"
		"""
		data = []

		tfIdfSumOfClasses = {}
		for label in self.tf_idf_matrix.columns:
			tfIdfSumOfClasses[label] = self.tf_idf_matrix[label].sum()

		classes = self.tf_idf_matrix.columns

		for word in self.tf_idf_matrix.index:
			wordProbabilitiesForEachClass = dict()
			for label in classes:
				wordProbabilitiesForEachClass[label] = \
				self.calculateProbabilityOfWordForEachClass(word, label, tfIdfSumOfClasses[label])
			data.append(wordProbabilitiesForEachClass)

		self.probabiliesWordGivenClass = \
		pd.DataFrame(data, columns=['Malas', 'Regulares', 'Buenas', 'Excelentes'],
		 index= self.tf_idf_matrix.index)

		self.probabiliesWordGivenClass.to_csv("./Probabilidades.csv", header = True)


	def read_CSV_toObtainClassifier(self, filePath, indexBy):
		"""
		Función para crear un DataFrame que contiene la Matriz de probabilidades
		condicionales.
		Args:
			filePath: Dirección del archivo a leer
			indexBy: Columna por la que se va indexar el DataFrame, debe
			ser el nombre de la columna dónde se encuentran las palabras.
		"""
		# creación del dataframe
		self.probabiliesWordGivenClass = pd.read_csv(filePath, delimiter=",", index_col=indexBy)

	def getProbabilityOfWordForGivenClass(self, word, label):
		"""
			Función para la obtención de la probabilidad condicional de
			una palabra dada
			Returns:
			Probabilidad de la Palabra dada la clase
		"""
		if word not in self.probabiliesWordGivenClass.index:
			return self.calculateProbabilityOfWordForEachClass(word, label,
				self.tfIdfSumOfClasses[label])

		probabilityValue = self.probabiliesWordGivenClass.loc[[word], [label]]
		return (probabilityValue.values[0])[0]

	def getClassOfaMovie(self, filePath):
		"""
			Método de para obtener la clase de un Documento.
			Las clases son ['Malas', 'Regulares', 'Buenas', 'Excelentes'].

			Args: ruta del archivo a evaluar
		"""

		print(self.tfIdfSumOfClasses)

		tk = Tokenization(filePath)
		tokens = tk.CreateTokens()

		st = Stemming(tokens)
		datosStemming = st.doStemming()

		words = list(datosStemming.values())[0]

		classes = ['Malas', 'Regulares', 'Buenas', 'Excelentes']

		probabilities = []

		for probabilityForClass, label in zip(self.priors, classes):
			probability = probabilityForClass
			for word in words:
				probability = probability + \
				self.getProbabilityOfWordForGivenClass(word, label)

			probabilities.append(probability)

		# print(probabilities)
		# print(max(probabilities))
		print(classes[(probabilities.index(max(probabilities)))])




class Laplace:

	def crearTablaEntrenamiento (self, nombresArchivosPeliculas, nombreArchivoTablaTFIDF):
		"""
			El  método  crearTablaEntrenamiento  automatiza el proceso de creación de la tabla TF-IDF con 4 clasificaciones
			(mala, regular, buena y excelente) y guarda la tabla TFDIF resultante en un archivo con extensión .csv.
			El documento de salida tiene la siguiente estructura:
									|	Malas 		 |		Regulares 		|		Buenas  	|		Excelentes
							palabra1|    TF-IDF					TF-IDF				TF-IDF				TF-IDF
							palabra2|	 TF-IDF					TF-IDF				TF-IDF				TF-IDF
							palabra3|		.						.					.					.
							palabra4|		.						.					.					.
							...

			Parámetros de entrada:
				nombresArchivosPeliculas: Es una palabra separada por comas donde  se mencionan los nombres de archivos
										donde se encuentran todas las películas ya limpias. Debe colocar la extensión
										Ejemplo:
											"PeliculasMalasResult.txt,PeliculasRegularesResult.txt,PeliculasBuenasResult.txt,PeliculasExcelentesResult.txt"
											Donde:
												PeliculasMalasResult.txt contiene las 100 películas LIMPIAS listas para usarse para entrenar
												PeliculasRegularesResult.txt contiene las 100 películas LIMPIAS listas para usarse para entrenar, etc

			Parametros de salida:
				nombreArchivoTablaTFIDF: Es el nombre .csv donde se guarda toda la tabla TF-IDF generada. NO colocar extensión

		"""
		archivosPeliculas = nombresArchivosPeliculas.split(',')
		columnasDFIDF = []
		for archivo in archivosPeliculas:
			tk = Texto(archivo)
			tokens = tk.CreateTokens()
			table = tk.matrizDeFrecuencias(tk.desempaquetaDiccionario(tokens))
			tfTabla = tk.matrizTF(table)
			tabla3 = tk.PalabrasPorDocumentos(table)
			idfTabla = tk.matrizIDF(table, tabla3, len(tk.desempaquetaDiccionario(tokens)))
			tfidfTabla= tk.matrizTFIDF(tfTabla, idfTabla)
			columnasDFIDF.append(tfidfTabla)
		tabla = TFIDF_global()
		columnas = tabla.tablaDataFrame(columnasDFIDF[0],columnasDFIDF[1],columnasDFIDF[2],columnasDFIDF[3],False,None)

		tabla.guardarTablaTFIDF(columnas, nombreArchivoTablaTFIDF)


	def tokenizarPeliculaAAnalizar(self,ruta,nombrePelicula):
		"""
			Se obtiene  una lista con los tokens de la pelicula que se desea analizar para que posteriormente se pueda
			utilizar en el método  "determinarPuntuación". Cabe destacar que el archivo debe estar libre de cualquier caracter
			que no sea  una letra, por lo que PREVIAMENTE  A USAR ESTE MÉTODO se necesita usar  el método
			limpiarPeliculaIndividual(ruta, nombrePelicula) perteneciente a la clase limpiadorTexto

			Entrada:
					nombrePelicula: Es el archivo .txt de la pelicula que se desea analizar, ya preprocesada para que solo tenga texto
										Debe colocar la extensión. Ejemplo: Batman.txt
			salida:
					Lista con los tokens
		"""

		tk = Tokenization(ruta,nombrePelicula)
		tokens = tk.CreateTokens()

		st = Stemming(tokens)
		datosStemming = st.doStemming()

		words = list(datosStemming.values())[0]
		return words


	def determinarPuntuacion (self,tokens,k,x,tabla,sumas):
		"""
			Calcula la probabilidad  de que una película X sea Mala, regular, buena o excelente.
			Para ello se emplea  Laplace y una escala logarítmica porque muchas veces las probabilidades
			serán tan pequeñas que en otra escala podrían provocar pérdida de información, de esta forma se elimita ese problema.

			Datos de entrada:
				tokens: es una lista con todos los n-gramas pertenecientes  a un texto
				k: constante para aplanar la curva. Generalmente se utiliza 1
				x: clases
				tabla: Es el dataframe donde aparecen todos los n-gramas como filas y los idiomas como columnas, y en cada celda está la frecuencia con la que apareció en el documento
				sumas: lista con la cantidad total de n-gramas en cada idioma

		"""

		prob_malas = math.log(tabla['Malas'].gt(0).sum()+k) - math.log(len(list(tabla.index))+k*x)
		prob_regulares = math.log(tabla['Regulares'].gt(0).sum()+k) - math.log(len(list(tabla.index))+k*x)
		prob_buenas = math.log(tabla['Buenas'].gt(0).sum()+k) - math.log(len(list(tabla.index))+k*x)
		prob_excelentes = math.log(tabla['Excelentes'].gt(0).sum()+k)-math.log(len(list(tabla.index))+k*x)


		prob_malas = math.log(tabla['Malas'].gt(0).sum()+k)-math.log(len(list(tabla.index))+k*x)
		prob_regulares = math.log(tabla['Regulares'].gt(0).sum()+k)-math.log(len(list(tabla.index))+k*x)
		prob_buenas = math.log(tabla['Buenas'].gt(0).sum()+k)-math.log(len(list(tabla.index))+k*x)
		prob_excelentes = math.log(tabla['Excelentes'].gt(0).sum()+k)-math.log(len(list(tabla.index))+k*x)


		for grama in tokens:
			if grama in list(tabla.index):
				prob_malas += math.log(list(tabla.loc[grama])[0]+k)-math.log(sumas[0]+k*len(list(tabla.index)))
				prob_regulares += math.log(list(tabla.loc[grama])[1]+k)-math.log(sumas[1]+k*len(list(tabla.index)))
				prob_buenas += math.log(list(tabla.loc[grama])[2]+k)-math.log(sumas[2]+k*len(list(tabla.index)))
				prob_excelentes  += math.log(list(tabla.loc[grama])[3]+k)-math.log(sumas[3]+k*len(list(tabla.index)))

		#print([prob_malas, prob_regulares, prob_buenas, prob_excelentes])

		probaMax = max([prob_malas,prob_regulares,prob_buenas, prob_excelentes])
		if probaMax ==  prob_malas:
			return 'Malas'
		elif probaMax == prob_regulares:
			return 'Regulares'
		elif probaMax == prob_buenas:
			return 'Buenas'
		else:
			return 'Excelentes'



class algoritmoSVM:

	def construirTablaEntrenamiento(self, nombresArchivosPeliculas, nombreArchivoTablaTFIDF):
		#
		#	El  método  construirTablaEntrenamiento  automatiza el proceso de creación de la tabla TF-IDF con 4 clasificaciones
		#	(mala, regular, buena y excelente) y guarda la tabla TFDIF resultante en un archivo con extensión .csv.
		#	El documento de salida tiene la siguiente estructura:
		#							|	palabra1 		 |		palabra2 		|		palabra3  	|		palabra4
		#					  Malas |    TF-IDF					TF-IDF				TF-IDF				TF-IDF
		#					Regulres|	 TF-IDF					TF-IDF				TF-IDF				TF-IDF
		#					Buenas  |		.						.					.					.
		#				  Excelentes|		.						.					.					.
		#					...
		#
		#	Parámetros de entrada:
		#		nombresArchivosPeliculas: Es una palabra separada por comas donde  se mencionan los nombres de archivos
		#								donde se encuentran todas las películas ya limpias. Debe colocar la extensión
		#								Ejemplo:
		#									"PeliculasMalasResult.txt,PeliculasRegularesResult.txt,PeliculasBuenasResult.txt,PeliculasExcelentesResult.txt"
		#									Donde:
		#										PeliculasMalasResult.txt contiene las 100 películas LIMPIAS listas para usarse para entrenar
		#										PeliculasRegularesResult.txt contiene las 100 películas LIMPIAS listas para usarse para entrenar, etc
		#
		#	Parametros de salida:
		#		nombreArchivoTablaTFIDF: Es el nombre .csv donde se guarda toda la tabla TF-IDF generada.
		columnasDFIDF = []
		archivos = nombresArchivosPeliculas.split(",")
		for archivo in archivos:
			tk = Texto(archivo)
			tokens = tk.CreateTokens()
			table = tk.matrizDeFrecuencias(tk.desempaquetaDiccionario(tokens))
			tfTabla = tk.matrizTF(table)
			tabla3 = tk.PalabrasPorDocumentos(table)
			idfTabla = tk.matrizIDF(table, tabla3, len(tk.desempaquetaDiccionario(tokens)))
			tfidfTabla= tk.matrizTFIDF(tfTabla, idfTabla)
			columnasDFIDF.append(tfidfTabla)
			tabla = TFIDF_global()

		columnas = tabla.tablaDataFrame(columnasDFIDF[0],columnasDFIDF[1],columnasDFIDF[2],columnasDFIDF[3],True,None)
		tabla.guardarTablaTFIDF(columnas, nombreArchivoTablaTFIDF)
		tablaEntrenamiento = pd.read_csv(nombreArchivoTablaTFIDF, index_col=0)
		tablaEntrenamiento["Calificacion"] = ["Mala","Regular", "Buena","Excelente"]
		guardaArc = tablaEntrenamiento.to_csv(str(nombreArchivoTablaTFIDF), index = True, header = True)


	def entrenarSistema(self, nombreTablaTFIDF, tipoKernel):
		"""
			El método entrenarSistema está diseñado para inicializar el algoritmo de Máquina de Vectores de Soporte (SVM).
			Recibe los siguientes parámetros:
				ENTRADA:
					nombreTablaTFIDF: Nombre del archivo .csv creado con el método construirTablaEntrenamiento(nombresArchivosPeliculas, nombreArchivoTablaTFIDF)
										de este mismo objeto
					tipoKernel:  El algoritmo cuenta con varios tipos de kernel. Puede ser 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'

				SALIDA:
					Instancia del algoritmo SVM ya entrenado
		"""
		datos = pd.read_csv(nombreTablaTFIDF, index_col=0)
		x = datos.drop("Calificacion",axis =1)
		y = datos["Calificacion"]
		x_train, x_test, y_train, y_test = train_test_split(x,y,test_size= 0.1)

		#algoritmo
		svc = SVC(kernel=tipoKernel)
		return svc.fit(x_train,y_train)


	def analizarPelicula(self,nombreTablaTFIDF, archivoPelicula, instanciaEntrenada):
		"""
			El método analizarPelicula, hace la predicción sobre los subtítulos de una película con el algoritmo SVM.

			ENTRADA:
				nombreTablaTFIDF: Se coloca el mismo archivo con el que se entrenó el sistema
				archivoPelicula: Nombre del archivo donde está la película LIMPIA (solo texto) que se va analizar.
				instanciaEntrenada Instancia del algoritmo que ya fue entrenada previamente
			SALIDA:
				Texto con la predicción
		"""

		datos = pd.read_csv(nombreTablaTFIDF, index_col=0)

		columnasDFIDF = []
		tk = Texto(archivoPelicula)
		tokens = tk.CreateTokens()
		table = tk.matrizDeFrecuencias(tk.desempaquetaDiccionario(tokens))
		tfTabla = tk.matrizTF(table)
		tabla3 = tk.PalabrasPorDocumentos(table)
		idfTabla = tk.matrizIDF(table, tabla3, len(tk.desempaquetaDiccionario(tokens)))
		tfidfTabla= tk.matrizTFIDF(tfTabla, idfTabla)
		columnasDFIDF.append(tfidfTabla)
		tabla = TFIDF_global()

		columnas = tabla.tablaDataFrame(columnasDFIDF[0],{},{},{},True,list(datos.keys()))

		tabla.guardarTablaTFIDF(columnas, archivoPelicula +".csv")
		svc = instanciaEntrenada
		peliculasAnalizar = pd.read_csv(archivoPelicula +".csv", index_col=0)
		x_prediccion = peliculasAnalizar.drop("Calificacion",axis =1)
		prediccion = svc.predict(x_prediccion)
		print("SVC: ",prediccion)




class algoritmoKNeighbors:

	def construirTablaEntrenamiento(self, nombresArchivosPeliculas, nombreArchivoTablaTFIDF):
		#
		#	El  método  construirTablaEntrenamiento  automatiza el proceso de creación de la tabla TF-IDF con 4 clasificaciones
		#	(mala, regular, buena y excelente) y guarda la tabla TFDIF resultante en un archivo con extensión .csv.
		#	El documento de salida tiene la siguiente estructura:
		#							|	palabra1 		 |		palabra2 		|		palabra3  	|		palabra4
		#					  Malas |    TF-IDF					TF-IDF				TF-IDF				TF-IDF
		#					Regulres|	 TF-IDF					TF-IDF				TF-IDF				TF-IDF
		#					Buenas  |		.						.					.					.
		#				  Excelentes|		.						.					.					.
		#					...
		#
		#	Parámetros de entrada:
		#		nombresArchivosPeliculas: Es una palabra separada por comas donde  se mencionan los nombres de archivos
		#								donde se encuentran todas las películas ya limpias. Debe colocar la extensión
		#								Ejemplo:
		#									"PeliculasMalasResult.txt,PeliculasRegularesResult.txt,PeliculasBuenasResult.txt,PeliculasExcelentesResult.txt"
		#									Donde:
		#										PeliculasMalasResult.txt contiene las 100 películas LIMPIAS listas para usarse para entrenar
		#										PeliculasRegularesResult.txt contiene las 100 películas LIMPIAS listas para usarse para entrenar, etc
		#
		#	Parametros de salida:
		#		nombreArchivoTablaTFIDF: Es el nombre .csv donde se guarda toda la tabla TF-IDF generada.

		archivos = nombresArchivosPeliculas.split(",")
		columnasDFIDF = []
		for archivo in archivos:
			tk = Texto(archivo)
			tokens = tk.CreateTokens()
			table = tk.matrizDeFrecuencias(tk.desempaquetaDiccionario(tokens))
			tfTabla = tk.matrizTF(table)
			tabla3 = tk.PalabrasPorDocumentos(table)
			idfTabla = tk.matrizIDF(table, tabla3, len(tk.desempaquetaDiccionario(tokens)))
			tfidfTabla= tk.matrizTFIDF(tfTabla, idfTabla)
			columnasDFIDF.append(tfidfTabla)
			tabla = TFIDF_global()

		columnas = tabla.tablaDataFrame(columnasDFIDF[0],columnasDFIDF[1],columnasDFIDF[2],columnasDFIDF[3],True,None)
		tabla.guardarTablaTFIDF(columnas, nombreArchivoTablaTFIDF)
		tablaEntrenamiento = pd.read_csv(nombreArchivoTablaTFIDF, index_col=0)
		tablaEntrenamiento["Calificacion"] = ["Mala","Regular", "Buena","Excelente"]
		guardaArc = tablaEntrenamiento.to_csv(str(nombreArchivoTablaTFIDF), index = True, header = True)


	def entrenarSistema(self, nombreTablaTFIDF, n):
		"""
			El método entrenarSistema está diseñado para inicializar el algoritmo de Máquina de Vectores de Soporte (SVM).
			Recibe los siguientes parámetros:
				ENTRADA:
					nombreTablaTFIDF: Nombre del archivo .csv creado con el método construirTablaEntrenamiento(nombresArchivosPeliculas, nombreArchivoTablaTFIDF)
										de este mismo objeto
					tipoKernel:  El algoritmo cuenta con varios tipos de kernel. Puede ser 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
					n: Numero de vecinos del algoritmo. Debe estar entre 1 y 3
				SALIDA:
					Instancia del algoritmo SVM ya entrenado
		"""
		datos = pd.read_csv(nombreTablaTFIDF, index_col=0)
		x = datos.drop("Calificacion",axis =1)
		y = datos["Calificacion"]
		x_train, x_test, y_train, y_test = train_test_split(x,y,test_size= 0.1)

		#algoritmo
		knn = KNeighborsClassifier(n_neighbors = 3)
		return knn.fit(x_train,y_train)


	def analizarPelicula(self,nombreTablaTFIDF, archivoPelicula, instanciaEntrenada):
		"""
			El método analizarPelicula, hace la predicción sobre los subtítulos de una película con el algoritmo SVM.

			ENTRADA:
				nombreTablaTFIDF: Se coloca el mismo archivo con el que se entrenó el sistema
				archivoPelicula: Nombre del archivo donde está la película LIMPIA (solo texto) que se va analizar.
				instanciaEntrenada: Instancia del algoritmo que ya fue entrenada previamente

			SALIDA:
				Texto con la predicción
		"""

		datos = pd.read_csv(nombreTablaTFIDF, index_col=0)

		columnasDFIDF = []
		tk = Texto(archivoPelicula)
		tokens = tk.CreateTokens()
		table = tk.matrizDeFrecuencias(tk.desempaquetaDiccionario(tokens))
		tfTabla = tk.matrizTF(table)
		tabla3 = tk.PalabrasPorDocumentos(table)
		idfTabla = tk.matrizIDF(table, tabla3, len(tk.desempaquetaDiccionario(tokens)))
		tfidfTabla= tk.matrizTFIDF(tfTabla, idfTabla)
		columnasDFIDF.append(tfidfTabla)
		tabla = TFIDF_global()

		columnas = tabla.tablaDataFrame(columnasDFIDF[0],{},{},{},True,list(datos.keys()))

		tabla.guardarTablaTFIDF(columnas, archivoPelicula +".csv")
		knn = instanciaEntrenada
		peliculasAnalizar = pd.read_csv(archivoPelicula +".csv", index_col=0)
		x_prediccion = peliculasAnalizar.drop("Calificacion",axis =1)
		print("Prediccion por Knn: ",knn.predict(x_prediccion))


def main():

	#LimpiarArchivos
	limpiador = limpiadorTexto()
	#limpiador.ProcessFiles(limpiador.FindFiles("srt","PeliculasBuenas,PeliculasExcelentes,PeliculasMalas,PeliculasRegulares"))

	peliculas = [["./EjemplosExternos/buenaResult","Dunkirk.2017.1080p.BluRay.H264.AAC-RARBG.srt"],["./EjemplosExternos/buenaResult","Jim Carrey - The Un-Natural Act (1991).srt"], \
	["./EjemplosExternos/excelentResult","Banking.On.Africa.The.Bitcoin.Revolution.2020.WEBRip.x264-ION10.srt"],["./EjemplosExternos/excelentResult","Marvels.The.Punisher.S02E13.WEB.x264-STRiFE.srt"],\
	["./EjemplosExternos/malaResult","2_English.srt"],["./EjemplosExternos/malaResult","Skinned.2020.NORDiC.1080p.WEB-DL.H.264.DD5.1-TWA.sv.srt"],\
	["./EjemplosExternos/regularResult","Splice.en.srt"],["./EjemplosExternos/regularResult","Stranded (2001).srt"]]


	"""for ruta,pelicula in peliculas:
		print("Limpiando actualmente: ", pelicula)
		limpiador.limpiarPeliculaIndividual(ruta,pelicula)"""


	laplace = Laplace()
	#laplace.crearTablaEntrenamiento("PeliculasMalasResult.txt,PeliculasRegularesResult.txt,PeliculasBuenasResult.txt,PeliculasExcelentesResult.txt","TFIDF-Laplace.csv")
	tablaEntrenamientoLaplace = pd.read_csv("TFIDF-Laplace.csv", index_col=0)


	# print("Algoritmo de SVM")
	svm_instancia = algoritmoSVM()
	#svm_instancia.construirTablaEntrenamiento("PeliculasMalasResult.txt,PeliculasRegularesResult.txt,PeliculasBuenasResult.txt,PeliculasExcelentesResult.txt","TFIDF-SVM.csv")
	svm_entrenado = svm_instancia.entrenarSistema("TFIDF-SVM.csv","linear")
	
	# print("Algoritmo de N-Neighbors")
	n_neighbors = algoritmoKNeighbors()
	#n_neighbors.construirTablaEntrenamiento("PeliculasMalasResult.txt,PeliculasRegularesResult.txt,PeliculasBuenasResult.txt,PeliculasExcelentesResult.txt","TFIDF-Nneig.csv")
	neighbors_entrenado = n_neighbors.entrenarSistema("TFIDF-Nneig.csv",3)
	

	for ruta,pelicula in peliculas:
		print("------------------------------------------------------")
		ruta = ruta.replace("./","")
		#pelicula = pelicula.replace(".srt",".txt")
		print ("ANALIZANDO PELICULA: ", pelicula[:len(pelicula)-3])
		print("Laplace")
		laplace.dirname =''
		tokensPelicula = laplace.tokenizarPeliculaAAnalizar(ruta, pelicula)
		print(laplace.determinarPuntuacion(tokensPelicula,1,4,tablaEntrenamientoLaplace,list(tablaEntrenamientoLaplace.sum())))

		print("SVM")
		svm_instancia.dirname = ''
		svm_instancia.analizarPelicula("TFIDF-SVM.csv",ruta+'/'+pelicula,svm_entrenado)

		print("N-Neighbors")
		n_neighbors.dirname = ''
		n_neighbors.analizarPelicula("TFIDF-Nneig.csv",ruta+'/'+pelicula,neighbors_entrenado)






if __name__ == "__main__":
	main()
