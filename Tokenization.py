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
			print(linesList)
            # allSubtitlesInMovie = ""
            # for line in linesList:
            #     if(line == "-----NewMovie-----"):
            #         allSubtitlesInMovie = ""
			#
			#
            # allTokens[file] = tokens
			return allTokens

tk = Tokenization(sys.argv[1])
tokens = tk.CreateTokens()
