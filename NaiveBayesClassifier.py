import pandas as pd
import os
import math
from aplicacion import Tokenization, Stemming

class NaiveBayesClassifierWithTfIdf:
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

def main():
    classifier = NaiveBayesClassifierWithTfIdf()
    classifier.read_CSV_toObtainTfIdfMatrix("tablaTFIDF.csv", "Unnamed: 0")
    # nuestro clasificador tiene 100 documentos por cada clase y
    # 400 documentos en total
    classifier.buildPriors([100 for i in range(4)], 400)
    # # creación de la matriz de probabilidades
    classifier.buildClassifier()

def main1():
    classifier = NaiveBayesClassifierWithTfIdf()
    # matriz tf idf
    classifier.read_CSV_toObtainTfIdfMatrix("tablaTFIDF.csv", "Unnamed: 0")
    # probabilidades prior
    classifier.buildPriors([100 for i in range(4)], 400)
    # matriz de probabilidades condicionales
    classifier.read_CSV_toObtainClassifier("Probabilidades.csv", "Unnamed: 0")
    # pelicula a evuluar previamente limpiada
    classifier.getClassOfaMovie("Subs ejemplo/NewSubs/Justice.League.Dark.Apokolips.War.2020.HDRip.XviD.AC3-EVO.srt")

main1()
