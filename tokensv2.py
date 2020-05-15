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

    '''
        CreateTokens
        Devuelve: Diccionario 
            Clave: Nombre del archivo a tokenizar
            Valor: Lista con todos los tokens del archivo
    '''

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
            print(str(key),"\n")
            print(diccionario[key][len(diccionario[key])-1],"\n")
            print("Cantidad de peliculas de la categoria: ", len(diccionario[key]),"\n\n\n\n\n\n")


tk = Tokenization("PeliculasExcelentesResult.txt")
tokens = tk.CreateTokens()
tk.verTabla(tokens)
print("Cantidad de categorias: ", len(tokens))