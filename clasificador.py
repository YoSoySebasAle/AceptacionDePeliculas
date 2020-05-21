"""
	Alumno: Ramírez Castillo Miguel Ángel
	Profesor: Octavio
	Asignatura: Análisis y procesamiento digital de textos
	Grupo: 1:
	Semestre: 2020-2

	Descripción:
		El presente programa permite clasificar textos de tres idiomas distintos:
		Alemán, Francés, Inglés y Español

"""

import pandas as pd
import math
from aplicacion import Tokenization, Stemming



def crearEnegramas(n, documento):
	"""
	# Recibe como parámetros:
	#	n: tamaño del n-grama
	#	documento: nombre del archivo al que se obtendran los n-gramas
	# Salida:
	#	Una lista con todos los n-gramas
	"""
	texto = ''
	resultados = []

	#abre el archivo
	try:
		documento = open(documento,encoding = "utf8")

		for linea in documento.readlines():
			texto += linea
		documento.close()

		texto = texto.replace('\n',' ') #quita saltos de linea
		#print(texto)
		for caracter in texto: #quita caracteres que no sean A-Z o a-z
			if ord(caracter) in range(34,39) or ord(caracter) in range(40,63) or ord(caracter) in range(64,64) or ord(caracter) in range(91,96) or ord(caracter) in range(123,161) or ord(caracter) in range(162,255):
				texto = texto.replace(caracter, '')


		texto = texto.lower() #Pone en minúsculas el texto
		#print(texto)

		for i in range(0,len(texto)-n):
			resultados.append(texto[i:i+n])
	except IOError:
		resultados = []

	return resultados


def bolsaDePalabras(listaNgramas):
	"""
		Recibe una lista de Ngramas producida con la función crearEnegramas
		Devuelve la bolsa de palabras (BANCO)
	"""

	if len(listaNgramas) > 0:
		bolsa = {}
		for grama in listaNgramas:
			if grama not in bolsa.keys():
				bolsa[grama] = 1
			else:
				bolsa[grama] += 1
	else:
		bolsa = {}
	return bolsa


def tablaDataFrame(bolsaEsp,bolsaIng, bolsaAle, bolsaFran):
	"""
		Se construye una tabla con las frecuencias de cada n grama para cada idioma.
		Como parámetros recibe cada una de las bosas/Bancos de datos generados  con los n-gramas y la función de bolsa de palabras.
	"""

	#Todos los ngramas van como filas, se eliminan los repetidos
	filas = list(bolsaEsp.keys()) + list(bolsaIng.keys()) +  list(bolsaAle.keys()) +  list(bolsaFran.keys())
	filas = list(set(filas))
	filas.sort()

	#listas para cada idioma de coincidencias
	espanol = []
	ingles = []
	aleman = []
	frances = []

	#Se itera sobre cada fila para saber si hay coincidencias con ese idioma
	for fila in filas:
		#Para español
		if fila in bolsaEsp.keys():
			espanol.append(bolsaEsp[fila])
		else:
			espanol.append(0)


		#Para Inglés
		if fila in bolsaIng.keys():
			ingles.append(bolsaIng[fila])
		else:
			ingles.append(0)


		#Para Alemán
		if fila in bolsaAle.keys():
			aleman.append(bolsaAle[fila])
		else:
			aleman.append(0)


		#Para Francés
		if fila in bolsaFran.keys():
			frances.append(bolsaFran[fila])
		else:
			frances.append(0)

	tabla = {'espanol': espanol, 'ingles':ingles, 'aleman':aleman, 'frances':frances }
	tabla = pd.DataFrame(tabla)
	tabla.index = filas
	return tabla



def precreacionTabla(n,nombreArchivo):
	"""
		Es una función intermediaria para evitar tener que llamar cada función por separado en el menú
	"""
	ngramas = crearEnegramas(n, nombreArchivo)
	#print(ngramas)
	if len(ngramas) == 0:
		return None
	bolsa = bolsaDePalabras(ngramas)
	if len(bolsa.keys()) == 0:
		return None
	return bolsa



def determinaIdioma (ngramas,k,x,tabla,sumas):
	"""
		Calcula las probabilidades de que pertenezca a ese idioma el texto hecho n-gramas y devuelve cuál es
		el idioma más probable del que se trate dicho texto.
		Para ello se emplea  Laplace y una escala logarítmica porque muchas veces las probabilidades
		serán tan pequeñas que en otra escala podrían provocar pérdida de información, de esta forma se elimita ese problema.

		Datos de entrada:
			ngramas: es una lista con todos los n-gramas pertenecientes  a un texto
			k: constante para aplanar la curva. Generalmente se utiliza 1
			x: clases
			tabla: Es el dataframe donde aparecen todos los n-gramas como filas y los idiomas como columnas, y en cada celda está la frecuencia con la que apareció en el documento
			sumas: lista con la cantidad total de n-gramas en cada idioma

	"""

	prob_esp = math.log(tabla['Malas'].gt(0).sum()+k) - \
            math.log(len(list(tabla.index))+k*x)
	prob_ing = math.log(tabla['Regulares'].gt(0).sum()+k) - \
            math.log(len(list(tabla.index))+k*x)
	prob_ale = math.log(tabla['Buenas'].gt(0).sum()+k) - \
            math.log(len(list(tabla.index))+k*x)
	prob_fran = math.log(tabla['Excelentes'].gt(
		0).sum()+k)-math.log(len(list(tabla.index))+k*x)

	for grama in ngramas:
		if grama in list(tabla.index):
			prob_esp += math.log(list(tabla.loc[grama])[0]+k)-math.log(sumas[0]+k*len(list(tabla.index)))
			prob_ing += math.log(list(tabla.loc[grama])[1]+k)-math.log(sumas[1]+k*len(list(tabla.index)))
			prob_ale += math.log(list(tabla.loc[grama])[2]+k)-math.log(sumas[2]+k*len(list(tabla.index)))
			prob_fran  += math.log(list(tabla.loc[grama])[3]+k)-math.log(sumas[3]+k*len(list(tabla.index)))

	print([prob_esp, prob_ing, prob_ale, prob_fran])

	probaMax = max([prob_esp,prob_ing,prob_ale, prob_fran])
	if probaMax ==  prob_esp:
		return 'Malas'
	elif probaMax == prob_ing:
		return 'Regulares'
	elif probaMax == prob_ale:
		return 'Buenas'
	else:
		return 'Excelentes'

def main():
	"""
		Función principal del programa. Aquí se encuentra el menú.
	"""


	print("OPCIONES: \n 1) Crear DataFrame \n 2) Analizar texto \n 3) Probar el rendimiento del sistema  \n 4) Salir")
	try:
		res= int(input("¿Qué desea hacer? "))

		if res == 1: #OPCION 1: Crear Dataframe
			"""
				Se genera un dataframe y se guarda con la extensión .csv.
				Para ello se necesita el nombre del  los archivos con los que se construirán los n-gramas y posteriormente obtener la bolsa
				de palabras.
			"""
			print("\n\n Crear DataFrame")
			#Obtiene la N
			n = int(input("\nDetermine el tamaño del ngrama: "))
			#Obtienen nombres de archivo
			esp = input("\nNombre del archivo en  español: ")
			ing = input("\nNombre del archivo en  ingles: ")
			alem = input("\nNombre del archivo en  aleman: ")
			franc = input("\nNombre del archivo en  francés: ")

			#obtienen las bolsas
			bolsaEsp = precreacionTabla(n,esp)
			bolsaIng = precreacionTabla(n,ing)
			bolsaAlem = precreacionTabla(n,alem)
			bolsaFrac = precreacionTabla(n,franc)

			if bolsaEsp !=None and bolsaIng !=None and bolsaAlem!=None and bolsaFrac!= None:
				#Genera la tabla
				tabla = tablaDataFrame(bolsaEsp,bolsaIng,bolsaAlem,bolsaFrac)
				guardaArc = tabla.to_csv(input('\n\nNombre de su archivo de salida con extensión [.csv]:  '), index = True, header = True)
			else:
				print("\nERROR: Verifique si todos los archivos tienen texto o si existen")

			main()





		elif res == 2: #OPCION 2: Analizar un texto
			"""
				Necesita cargar el dataframe generado con la opción 1 y  un archivo de texto donde se ecuentree la información a clasificar
			"""
			print("\n\n Analizador de texto")
			try:
				tabla = pd.read_csv(input('Archivo con extension [.csv]'), index_col=0)
				textoAnalizar = input('Archivo a analizar con extensión: ')
				gramas =  crearEnegramas(len(list(tabla.index)[0]), textoAnalizar)

				print("Su documento está en " + determinaIdioma(gramas, int(input("Ingrese un valor para K: ")), 4, tabla, list(tabla.sum())))
			except IOError:
				print("\n Verifique que existen ambos archivos")

			main()




		elif res == 3: #OPCION 3: Probar desempeño del sistema
			"""
												DESEMPEÑO DEL SISTEMA

				Para evaluar el desempeño del sistema se utilizan 10 documentos (prueba1, prueba2, ..., prueba10) en donde cada uno de ellos tiene un texto
				en un idioma en específico.
				En la lista de la línea 208 se debe colocar de qué idioma es el texto, ya que hace la comparación entre el resultado del sistema contra
				dicha lista para poder generar la matriz de confusión.

				Al usuario se muestra un resumen  de la matriz de confusión así como los resultados de las mediciones de  Precision, Recall y F1.

				Cabe añadir que de manera predeterminada se generan las bolsas de palabras con los archivos espanol.txt, ingles.txt, aleman.txt y frances.txt, pero
				puede generar las propias con la opción 1 del menú principal.

			"""
			print("'\n*** EVALUAR DESEMPEÑO DEL SISTEMA ***")
			print("\n Nota: La matriz de confusión se hace con base en si están o no en ESPAÑOL los documentos de prueba")
			eval_sistema = []
			eval_real  = ['ESPANOL','FRANCES','INGLES','ESPANOL','FRANCES','ALEMAN','ESPANOL','ALEMAN','INGLES','ESPANOL'] #El usuario pone de qué es cada documento en esta lista

			for n in range (2,6): #Obtienen las bolsas de palabras y n-gramas
				bolsaEsp = precreacionTabla(n,"espanol.txt")
				bolsaIng = precreacionTabla(n,"ingles.txt")
				bolsaAlem = precreacionTabla(n,"aleman.txt")
				bolsaFrac = precreacionTabla(n,"frances.txt")
				tabla = tablaDataFrame(bolsaEsp,bolsaIng,bolsaAlem,bolsaFrac)
				guardaArc = tabla.to_csv(str(n)+"gramaPrueba.csv", index = True, header = True)


			for n in range (2,6): #PAra cada N se obtiene la matriz de confusion y mediciones
				print("\n\nPARA N = " +str(n))
				tabla = pd.read_csv(str(n)+"gramaPrueba.csv", index_col=0)
				eval_sistema = []

				for numDoc in range(1,11): #Lee cada uno de los archivos Prueba1.txt hasta prueba10.txt
					gramas =  crearEnegramas(len(list(tabla.index)[0]), "prueba" + str(numDoc)+".txt")
					eval_sistema.append(determinaIdioma(gramas, 1, 4, tabla, list(tabla.sum())))



				if len(eval_sistema) == len(eval_real): #Comprueba que tienen la misma longitud
					TN = 0
					FP = 0
					FN = 0
					TP = 0
					for i in range(0,len(eval_real)):
						if eval_real[i] ==  'ESPANOL' and eval_sistema[i]!= 'ESPANOL':
							#FALSO NEGATIVO (FN)
							FN += 1
							print("'\tFN")
						elif eval_real[i] ==  'ESPANOL' and eval_sistema[i] == 'ESPANOL':
							#VERDADERO POSITIVO
							TP += 1
							print("\tTP")
						elif eval_real[i] !=  'ESPANOL' and eval_sistema[i] != 'ESPANOL':
							#VERDADEROS NEGATIVOS
							TN += 1
							print("\tTN")
						else:
							#FALSO POSITIVO
							FP += 1
							print("\tFP")
					print("FN = " +str(FN) + " TP = " + str(TP) + " TN = " + str(TN) + " FP = " + str(FP))

					#Cálculo de las medidas de precision,  Recall y F1
					precision = (TP)/(TP+FP)
					recall =(TP)/(TP+FN)
					f1 = (2*precision*recall) / (precision + recall)
					print("\n\tPrecision: " + str(precision))
					print("\n\tRecall: " + str(recall))
					print("\n\tF1: " + str(f1))


				else:
					print("\n Verifique si llenó la lista de qué idioma es cada documento. LÍNEA 258")
			main()


		elif res == 4: #Opción 4, salir del programa
			exit()

		else:
			print("la opción no existe")
			main()

	except ValueError:
		print("\n\n\n\n")
		main()


sumas = [59.82426026140742, 55.449743905826224, 54.776973536408754, 57.91474330189863]

tf_idf_matrix = pd.read_csv("tablaTFIDF.csv", delimiter=",", index_col="Unnamed: 0")

filesPath = ["EjemplosExternos/buenaResult/Jim Carrey - The Un-Natural Act (1991).srt",
            "EjemplosExternos/excelentResult/Marvels.The.Punisher.S02E13.WEB.x264-STRiFE.srt",
            "EjemplosExternos/malaResult/2_English.srt",
            "EjemplosExternos/regularResult/Splice.en.srt"]

for filePath in filesPath:
	tk = Tokenization(filePath)
	tokens = tk.CreateTokens()

	st = Stemming(tokens)
	datosStemming = st.doStemming()

	words = list(datosStemming.values())[0]


	print(determinaIdioma(words, 1, 4, tf_idf_matrix, sumas))



# main()
