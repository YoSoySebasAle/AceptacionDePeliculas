from flask import Flask, jsonify, request, render_template
import os
import pandas as pd
from clases_clasificador import Laplace, algoritmoKNeighbors, algoritmoSVM

UPLOAD_FOLDER = "./filesUploaded/"
tablaEntrenamientoLaplace = pd.read_csv("TFIDF-Laplace.csv", index_col=0)
svm_instancia = algoritmoSVM()
svm_entrenado = svm_instancia.entrenarSistema("TFIDF-SVM.csv", "linear")
n_neighbors = algoritmoKNeighbors()
neighbors_entrenado = n_neighbors.entrenarSistema("TFIDF-Nneig.csv", 3)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def KNNclassifier(filePath):
    return n_neighbors.analizarPelicula("TFIDF-Nneig.csv", filePath, neighbors_entrenado)

def SVMclassifier(filePath):
    return svm_instancia.analizarPelicula("TFIDF-SVM.csv",filePath,svm_entrenado)

def LaplaceClassifier(fileName):
    laplace = Laplace()
    tokensPelicula = laplace.tokenizarPeliculaAAnalizar(app.config['UPLOAD_FOLDER'], fileName)
    return laplace.determinarPuntuacion(tokensPelicula,1,4,tablaEntrenamientoLaplace,list(tablaEntrenamientoLaplace.sum()))

def cleanFile(fileName):
    actualFile = open(fileName, encoding="utf8", errors="ignore")
    linesList = actualFile.readlines()
    linesForMovie = []
    for line in linesList:
        line = line.lower()
        line = line.rstrip()
        line = line.replace('<i>', '')
        line = line.replace('</i>', '')
        line = line.replace('<b>', '')
        line = line.replace('</b>', '')
        line = line.replace('<font>', '')
        line = line.replace('</font>', '')
        line = line.replace('♪', '')
        line = line.replace('- ', '')
        line = line.replace('"', '')
        line = line.replace('..', '')
        line = line.replace('.', '')
        line = line.replace('\'s', '')
        line = line.replace(',', '')
        line = line.replace('?', '')
        line = line.replace('(', '')
        line = line.replace(')', '')
        line = line.replace('!', '')
        line = line.replace(':', '')
        line = line.replace('1', '')
        line = line.replace('[', '')
        line = line.replace(']', '')
        line = line.replace('\'', '')
        line = line.strip()
        if not line.isdigit() and not '-->' in line:
            linesForMovie.append(line)

        with open(fileName, 'w', encoding="utf8", errors="ignore") as f:
            for item in linesForMovie:
                if item is not '':
                    f.write("%s\n" % item)

@app.route("/")
def moviesPage():
    return render_template("index.html")

@app.route("/Laplace", methods = ["POST"])
def uploadFile1():
    f = request.files['inputFile']
    newFilePath = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
    f.save(newFilePath)
    cleanFile(newFilePath)
    classOfMovie = LaplaceClassifier(f.filename)
    return jsonify({
        "label" : classOfMovie
    })

@app.route("/SVM", methods = ["POST"])
def uploadFile2():
    f = request.files['inputFile']
    newFilePath = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
    f.save(newFilePath)
    cleanFile(newFilePath)
    classOfMovie = SVMclassifier(newFilePath)
    return jsonify({
        "label" : classOfMovie
    })


@app.route("/KNN", methods=["POST"])
def uploadFile3():
    f = request.files['inputFile']
    newFilePath = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
    f.save(newFilePath)
    cleanFile(newFilePath)
    classOfMovie = KNNclassifier(newFilePath)
    return jsonify({
        "label": classOfMovie
    })

if __name__ == '__main__':
    app.run(debug=True, port=4000)
