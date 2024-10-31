# Classificação de Diabetes Usando Algoritmos de Machine Learning

Este projeto demonstra a aplicação de três algoritmos de machine learning — **Árvore de Decisão**, **K-Nearest Neighbors (KNN)** e **Máquinas de Vetores de Suporte (SVM)** — para classificar a presença de diabetes com base em parâmetros de saúde.

## 📋 Principais Funcionalidades

1. **Manipulação de Dados**: 
   - Utiliza a biblioteca **pandas** para ler e pré-processar o conjunto de dados de diabetes.
   - As colunas são renomeadas para português, facilitando a compreensão das variáveis e parâmetros.

2. **Pré-processamento de Dados**:
   - Divide o conjunto de dados em conjuntos de treino e teste para uma avaliação justa dos modelos.
   - Normaliza os dados para garantir que os algoritmos sejam treinados de forma eficaz e justa.

3. **Treinamento e Avaliação de Modelos**:
   - **Árvore de Decisão**: Treina um modelo de árvore de decisão e avalia seu desempenho usando uma matriz de confusão e acurácia.
   - **K-Nearest Neighbors (KNN)**: Aplica o algoritmo KNN, avaliando sua acurácia e métricas de desempenho.
   - **Máquinas de Vetores de Suporte (SVM)**: Implementa uma SVM com kernel linear, avaliando seu desempenho e interpretando os resultados.

4. **Métricas de Desempenho**:
   - Calcula e imprime a **matriz de confusão** e a **acurácia** para cada um dos modelos.
   - Compara as performances entre os três algoritmos para uma melhor compreensão dos resultados.

---

## 🛠️ Tecnologias e Bibliotecas Utilizadas

- **Python**: Linguagem de programação principal.
- **Pandas**: Para manipulação e pré-processamento de dados.
- **Scikit-Learn**: Para implementação dos algoritmos de classificação e avaliação de métricas.

---

## 🚀 Como Executar o Projeto

1. **Clone o repositório**:
   ```bash
   git clone https://github.com/seu-usuario/nome-do-repositorio.git
