import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

tabela = pd.read_csv("/content/sample_data/diabetes.csv", encoding="latin1")

tabela.rename(columns={
    'Pregnancies': 'Gravidezes',
    'Glucose': 'Glicose',
    'BloodPressure': 'Pressão Arterial',
    'SkinThickness': 'Espessura da Pele',
    'Insulin': 'Insulina',
    'BMI': 'IMC',
    'DiabetesPedigreeFunction': 'Função Pedigree de Diabetes',
    'Age': 'Idade',
    'Outcome': 'Resultado'
}, inplace=True)

dados = tabela.iloc[:,0:7].values
gabarito = tabela.iloc[:,8]  # A coluna 'Resultado' é o alvo

#0.25 divide em 4 partes
dados_treino, dados_teste, gabarito_treino, gabarito_teste = train_test_split(dados, gabarito, test_size=0.25, random_state=0)

def accuracy(confusion_matrix):
    sum, total = 0,0
    for i in range(len(confusion_matrix)):
        for j in range(len(confusion_matrix[0])):
            if i == j:
                sum += confusion_matrix[i,j]
            total += confusion_matrix[i,j]
    return sum/total

arvore_decisao = DecisionTreeClassifier()
arvore_decisao.fit(dados_treino, gabarito_treino)

previsoes = arvore_decisao.predict(dados_teste)

tree_cm = confusion_matrix(previsoes, gabarito_teste)
print('\nDecision Tree')
print('matriz de confusão:')
print(tree_cm,'\n')
print('Acurácia: ',accuracy(tree_cm)*100, '%')


knn = KNeighborsClassifier()
knn.fit(dados_treino, gabarito_treino)

knn_previsoes = knn.predict(dados_teste)

knn_cm = confusion_matrix(knn_previsoes, gabarito_teste)
print('K-Nearest Neighbors')
print('matriz de confusão:')
print(knn_cm,'\n')
print('acurácia: ',accuracy(knn_cm)*100,'%')

svm = SVC(kernel = 'linear')
svm.fit(dados_treino, gabarito_treino)

svm_previsoes = svm.predict(dados_teste)

svm_cm = confusion_matrix(svm_previsoes, gabarito_teste)
print('\nSupport Vector Machines')
print('matriz de confusão:')
print(svm_cm, '\n')
print('acurácia: ',accuracy(svm_cm)*100, '%')