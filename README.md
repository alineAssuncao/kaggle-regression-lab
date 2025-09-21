# 🏡 House Price Models — Trilha de Regressão com Notebooks

Este repositório é uma trilha prática de aprendizado em regressão com dados tabulares, usando como base o conjunto de dados da competição [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data). A proposta é aplicar, comparar e entender diferentes modelos de regressão, do mais simples ao mais avançado, por meio de notebooks bem documentados.

## 🎯 Objetivos

- Consolidar conhecimento em regressão supervisionada
- Construir um portfólio técnico com notebooks organizados
- Aprender a lidar com dados reais, pré-processamento e avaliação de modelos
- Produzir material útil para competições e estudos

## 🧭 Trilha de Notebooks

| Técnica                     | Notebook                          | Ferramenta Principal                      |
|----------------------------|-----------------------------------|-------------------------------------------|
| Regressão Linear           | `01-linear-regression.ipynb`      | `LinearRegression` (`sklearn`)            |
| Regressão Polinomial       | `02-polynomial-regression.ipynb`  | `PolynomialFeatures` + `LinearRegression` |
| Árvore de Decisão          | `03-decision-tree.ipynb`          | `DecisionTreeRegressor` (`sklearn`)       |
| Random Forest              | `04-random-forest.ipynb`          | `RandomForestRegressor` (`sklearn`)       |
| XGBoost                    | `05-xgboost.ipynb`                | `XGBRegressor` (`xgboost`)                |

## 📁 Estrutura do Repositório

```
├── data/
│   ├── raw/           # Dados originais do Kaggle
│   └── processed/     # Dados limpos e transformados
├── notebooks/         # Notebooks organizados por técnica
├── utils/             # Scripts auxiliares (ex: pré-processamento)
├── outputs/           # Gráficos, modelos salvos, submissões
├── README.md          # Este arquivo
└── requirements.txt   # Pacotes necessários para execução
```

## 🚀 Como Executar

1. Clone o repositório:
   ```bash
   git clone https://github.com/alineAssuncao/kaggle-regression-lab.git
   cd kaggle-regression-lab
   ```

2. Instale os pacotes:
   ```bash
   pip install -r requirements.txt
   ```

3. Execute os notebooks na pasta `notebooks/` usando Jupyter ou VS Code.

## 📦 Requisitos

- Python 3.8+
- `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `xgboost`

## 📌 Referência de Dados

Os dados utilizados são da competição [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data), disponível no Kaggle.


## 👩‍💻 Autora

Aline Assunção

Engenheira de Qualidade em transição para Inteligência Artificial

📫 [LinkedIn](https://www.linkedin.com/in/alineassuncaoai/)  

📬 aline.jassuncao@gmail.com
