"""
	Asignatura: Análisis y procesamiento digital de textos
	Grupo: 1
	Semestre: 2020-2
	
	Descripción:
		Programa que a partir de un conjunto de subtitulos limpia el conteniod no necesario para su posterior análisis

    Entradas:
        Nombre de las carpetas con los subtítulos a limpiar
        Ejemplo:
        C:/Users/Ivan/Anaconda3/python.exe c:/Users/Ivan/Documents/Textos/TextManager.py PeliculasBuenas,PeliculasExcelentes,PeliculasMalas,PeliculasRegulares 
		
"""

import glob, os
import sys

"""
    Find Files

    Recibe: Etension de los archivos a buscar
    Devuleve: Diccionario
        Clave: Nombre de la carpeta de los subtitulos
        Valor: Lista con todos los nombres de los subtítulos en dicha carpeta
"""

def FindFiles(extension):
    filesAllCategories = {}
    folders = sys.argv[1].split(",")
    for folder in folders:
        files = []
        dirname, filename = os.path.split(os.path.abspath(__file__))
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
def ProcessFiles(filesAllCategories):
    dirname, filename = os.path.split(os.path.abspath(__file__))
    for key in filesAllCategories:

        tokensForCategory = []
        os.chdir(dirname+"/"+key)

        for fileName in filesAllCategories[key]:
            actualFile = open(fileName, encoding="utf8", errors="ignore")
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
                line = line.strip()
                if not line.isdigit() and not '-->' in line:
                    linesForMovie.append(line)
                    for i in line.split():
                        tokensForCategory.append(i)

            path=dirname + "\\" + key + "Result.txt"
            with open(path, 'w+', encoding="utf8", errors="ignore") as f:
                for item in linesForMovie:
                    if item is not '':
                        f.write("%s\n" % item)


def Main():
    ProcessFiles(FindFiles("srt"))
Main()