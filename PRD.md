# 📄 PRD - Product Requirements Document

## 1. Visão Geral

Este projeto tem como objetivo construir uma trilha prática de aprendizado em regressão com dados tabulares, utilizando notebooks organizados por técnica. A base será o conjunto de dados da competição House Prices - Advanced Regression Techniques, e o foco será aplicar, comparar e entender diferentes modelos de regressão.

## 2. Objetivos

Criar um portfólio técnico com notebooks bem documentados.

Consolidar conhecimento em regressão e machine learning supervisionado.

Aprender a lidar com dados reais, pré-processamento, avaliação de modelos e tuning.

Produzir material que possa ser usado em entrevistas, competições ou como referência pessoal.

## 3. Escopo

5 notebooks principais cobrindo:

- Regressão Linear

- Regressão Polinomial

- Árvore de Decisão

- Random Forest

- XGBoost


Cada notebook deve conter:

- Carregamento e exploração dos dados

- Aplicação da técnica com explicações

- Avaliação dos resultados

- Visualizações e insights

Organização do repositório com estrutura clara de dados, notebooks, utilitários e resultados.

## 4. Público-Alvo

Estudantes e profissionais iniciando em machine learning.

Participantes de competições como o Kaggle.


## 5. Requisitos Funcionais

Repositório público no GitHub com estrutura:
```
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── utils/
├── outputs/
├── README.md
└── requirements.txt
```

Cada notebook deve ser executável do início ao fim.

Uso de bibliotecas como `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `xgboost`.

## 6. Requisitos Não Funcionais

Código limpo e comentado.

Visualizações claras e interpretáveis.

Explicações acessíveis para quem tem conhecimento básico de ML.

## 7. Critérios de Sucesso

Todos os notebooks executam sem erro.

O projeto é compreensível por terceiros.

O repositório é organizado e fácil de navegar.

O modelo final (XGBoost) gera uma submissão válida para o Kaggle.