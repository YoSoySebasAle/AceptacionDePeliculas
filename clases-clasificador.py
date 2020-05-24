import pandas as pd
import math
import glob, os
import sys
from nltk import sent_tokenize, word_tokenize, PorterStemmer
from nltk.corpus import stopwords





class  limpiadorTexto:
	def __init__(self):
		"""
			Constructor para la clase Limpiador de texto. Los parámetros que recibe son:
				-Extension: La extensión de los archivos que se desean buscar y limpiar
				-Ruta: Ruta de la carpeta que contiene los archivos a limpiar
				-Modo: Puede ser True o  False. Si se requiere limpiar todas las películas de la carpeta y meterlo todo 
						en un solo archivo de texto con el separador "----NEWMOVIE----", entonces seleccione TRUE (ESTO ES 
						PARA ENTRENAR EL SISTEMA), de lo contrario coloque FALSE y así se limpiarán cada uno de los 
						archivos de dicha carpeta conservando su nombre original (ESTO ES PARA LA OPERACIÓN DEL SISTEMA)
		"""
		self.dirname = ''


	def FindFiles(self,extension,carpetas):
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

	"""
		Process Files

		Recibe: Diccionario
			Clave: Nombre de la carpeta de los subtitulos
			Valor: Lista con todos los nombres de los subtítulos en dicha carpeta
		Devuleve: Crea un archivo para cada clave del diccionario. Este archivo contiene todos los subtitulos limpios de la categoria.
	"""
	def ProcessFiles(self,filesAllCategories):
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
		#dirname, filename = os.path.split(os.path.abspath(__file__))
		#dirname = self.dirname
		dirname = ruta
		#os.chdir(dirname+"/"+nombrePelicula)
		print(dirname+"/"+nombrePelicula)
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
			haber preprocesado el archivo para eliminar cualquier símbolo que no sea una letra.
	"""

	def __init__(self, files):
		"""
			Constructor de la clase Texto
			Recibe: el nombre del archivo
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
						archivo1 = [texto completo de la pelicula...]
						archivo2 = [texto completo de la pelicula...]
						archivo3 = [texto completo de la pelicula...]
						.
						.
						.
					}
		"""
		allTokens = {}
		for file in self.files:
			tokens = []
			print("SSSS: ",self.dirname)
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
			#doc tiene el nombre de la pelicula
			#f_table tiene la tabla de frecuencias de cada documento
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
				#se añadioo el factor correctivo
				idf_table[word] = math.log(total / float(palPorDoc[word]))+1

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

		return tf_idf_matriz2






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


#Para procesar archivos a analizar

class Tokenization:
	def __init__(self, ruta,archivo):
		#self.files = files.split(',')
		self.dirname, self.filename = os.path.split(os.path.abspath(__file__))
		self.ruta = ruta
		self.archivo = archivo
	"""
        CreateTokens
        Devuelve: Diccionario
            Clave: Nombre del archivo a tokenizar
            Valor: Lista con todos los tokens del archivo
	"""
	def CreateTokens(self):
		allTokens = {}
		#for file in self.files:
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
		self.dicTokens = dicTokens

	def doStemming(self):
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
	def __init__(self, tokens,k,numclases,tabla,sumas):
		self.tokens = tokens
		self.k = k
		self.x = numclases
		self.tabla = tabla
		self.sumas = sumas


	def determinarPuntuacion (self):
		"""
			Calcula las probabilidades de que pertenezca a ese idioma el texto hecho n-gramas y devuelve cuál es
			el idioma más probable del que se trate dicho texto.
			Para ello se emplea  Laplace y una escala logarítmica porque muchas veces las probabilidades
			serán tan pequeñas que en otra escala podrían provocar pérdida de información, de esta forma se elimita ese problema.

			Datos de entrada:
				tokens: es una lista con todos los n-gramas pertenecientes  a un texto
				k: constante para aplanar la curva. Generalmente se utiliza 1
				x: clases
				tabla: Es el dataframe donde aparecen todos los n-gramas como filas y los idiomas como columnas, y en cada celda está la frecuencia con la que apareció en el documento
				sumas: lista con la cantidad total de n-gramas en cada idioma

		"""

		prob_malas = math.log(self.tabla['Malas'].gt(0).sum()+self.k) - math.log(len(list(self.tabla.index))+self.k*self.x)
		prob_regulares = math.log(self.tabla['Regulares'].gt(0).sum()+self.k) - math.log(len(list(self.tabla.index))+self.k*self.x)
		prob_buenas = math.log(self.tabla['Buenas'].gt(0).sum()+self.k) - math.log(len(list(self.tabla.index))+self.k*self.x)
		prob_excelentes = math.log(self.tabla['Excelentes'].gt(0).sum()+self.k)-math.log(len(list(self.tabla.index))+self.k*self.x)

			
		prob_malas = math.log(self.tabla['Malas'].gt(0).sum()+self.k)-math.log(len(list(self.tabla.index))+self.k*self.x)
		prob_regulares = math.log(self.tabla['Regulares'].gt(0).sum()+self.k)-math.log(len(list(self.tabla.index))+self.k*self.x)
		prob_buenas = math.log(self.tabla['Buenas'].gt(0).sum()+self.k)-math.log(len(list(self.tabla.index))+self.k*self.x)
		prob_excelentes = math.log(self.tabla['Excelentes'].gt(0).sum()+self.k)-math.log(len(list(self.tabla.index))+self.k*self.x)
		
		
		for grama in self.tokens:
			if grama in list(self.tabla.index):
				prob_malas += math.log(list(self.tabla.loc[grama])[0]+self.k)-math.log(self.sumas[0]+self.k*len(list(self.tabla.index)))
				prob_regulares += math.log(list(self.tabla.loc[grama])[1]+self.k)-math.log(self.sumas[1]+self.k*len(list(self.tabla.index)))
				prob_buenas += math.log(list(self.tabla.loc[grama])[2]+self.k)-math.log(self.sumas[2]+self.k*len(list(self.tabla.index)))
				prob_excelentes  += math.log(list(self.tabla.loc[grama])[3]+self.k)-math.log(self.sumas[3]+self.k*len(list(self.tabla.index)))

		print([prob_malas, prob_regulares, prob_buenas, prob_excelentes])

		probaMax = max([prob_malas,prob_regulares,prob_buenas, prob_excelentes])
		if probaMax ==  prob_malas:
			return 'Malas'
		elif probaMax == prob_regulares:
			return 'Regulares'
		elif probaMax == prob_buenas:
			return 'Buenas'
		else:
			return 'Excelentes'




class main:

	#LimpiarArchivos
	#limpiador = limpiadorTexto()
	#limpiador.ProcessFiles(limpiador.FindFiles("srt","PeliculasBuenas,PeliculasExcelentes,PeliculasMalas,PeliculasRegulares"))


	#Prueba para un solo archivo
	"""archivos = []
	tk = Texto("PeliculasExcelentesResult.txt")
	tokens = tk.CreateTokens()

	print(len(tokens))
	"""
	"""print("Se crearon los nuevos arhcivos de peliiculas")
	#Creación de la tabla
	archivos = ["PeliculasMalasResult.txt","PeliculasRegularesResult.txt", "PeliculasBuenasResult.txt","PeliculasExcelentesResult.txt"]
	#archivos = ["PeliculasExcelentesResult.txt"]
	columnasDFIDF = []
	for archivo in archivos:
		#Crea objeto de la clase texto
		tk = Texto(archivo)
		#Obtienen los tokens
		tokens = tk.CreateTokens()
		#print(tokens)
		#print(tk.desempaquetaDiccionario(tokens))
		#Se obtiene la matriz de Frecuencias
		table = tk.matrizDeFrecuencias(tk.desempaquetaDiccionario(tokens))
		#print(table)
		#print("LEN FRECUENCIAS ", len(table))

		#Obtiene la matriz TF
		tfTabla = tk.matrizTF(table)
		#print(tfTabla)
		#print(len(tfTabla))
	
		#Se obtiene el número de documentos en los que aparece cada palabra
		tabla3 = tk.PalabrasPorDocumentos(table)
		#print("FRECUENCIAS POR DOCUMENTO: ",tabla3)
		#print("FRECUENCIAS POR DOCUMENTO: ",len(tabla3))
		#Crea matriz IDF
		idfTabla = tk.matrizIDF(table, tabla3, len(tk.desempaquetaDiccionario(tokens)))

		#Crea matriz TF-IDF
		tfidfTabla= tk.matrizTFIDF(tfTabla, idfTabla)

		#Se guarda esa columna para poder generar la tabla completa
		columnasDFIDF.append(tfidfTabla)
	
	tabla = TFIDF_global()
		
	columnas = tabla.tablaDataFrame(columnasDFIDF[0],columnasDFIDF[1],columnasDFIDF[2],columnasDFIDF[3])
		
	tabla.guardarTablaTFIDF(columnas, "tablaTFIDF")
	


	



	filesPath = ["EjemplosExternos/buenaResult/Jim Carrey - The Un-Natural Act (1991).txt",
			"EjemplosExternos/excelentResult/Marvels.The.Punisher.S02E13.WEB.x264-STRiFE.txt",
			"EjemplosExternos/malaResult/2_English.txt",
			"EjemplosExternos/regularResult/Splice.en.txt"]

	"""

	tf_idf_matrix = pd.read_csv("tablaTFIDF.csv", index_col=0)

	#Limpiar pelicula Individual
	limpiar = limpiadorTexto()
	limpiar.limpiarPeliculaIndividual("C:/Users/Miguel/Desktop/proyTextos/EjemplosExternos/excelentResult", "Community S01E23 1080p WEB-DL DD+ 5.1 x264-TrollHD.srt")
	#Medir pelicula
	
	
	filesPath = [["C:/Users/Miguel/Desktop/proyTextos/EjemplosExternos/excelentResult","Community S01E23 1080p WEB-DL DD+ 5.1 x264-TrollHD.srt.txt"]]

	for filePath in filesPath:	
		print(filePath)
		tk = Tokenization(filePath[0],filePath[1])
		tokens = tk.CreateTokens()

		st = Stemming(tokens)
		datosStemming = st.doStemming()

		words = list(datosStemming.values())[0]
		#print("DATOS: ", words,"\n\n\n")
		pelicula = Laplace(words, 1, 4, tf_idf_matrix, list(tf_idf_matrix.sum()))
		print(pelicula.determinarPuntuacion())
		

main()





















