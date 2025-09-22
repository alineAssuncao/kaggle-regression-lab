"""
Exemplo de uso da estrutura refatorada para regressão linear.
"""

# Importações organizadas por categoria
# Bibliotecas padrão
from pathlib import Path

# Bibliotecas de terceiros
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Módulos locais
from src.utils.data_processing import load_data, preprocess_data, create_preprocessor, split_data
from src.utils.evaluation import evaluate_model, plot_residuals


def main():
    # 1. Carregar dados
    print("Carregando dados...")
    data_path = Path("data/raw")
    train_df, test_df = load_data(data_path)
    
    # 2. Pré-processamento
    print("\nPré-processando dados...")
    train_df, numeric_features, categorical_features = preprocess_data(train_df)
    
    # Separar features e target
    X = train_df.drop('SalePrice', axis=1)
    y = np.log1p(train_df['SalePrice'])  # Aplicar log ao target para normalizar
    
    # 3. Criar pré-processador
    preprocessor = create_preprocessor(numeric_features, categorical_features)
    
    # 4. Dividir dados
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # 5. Criar e treinar modelo dentro de um pipeline
    print("\nTreinando modelo...")
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    
    model.fit(X_train, y_train)
    
    # 6. Fazer previsões
    print("\nFazendo previsões...")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # 7. Avaliar modelo
    print("\nAvaliando modelo no conjunto de treino:")
    train_metrics = evaluate_model(y_train, y_pred_train, "Regressão Linear (Treino)")
    
    print("\nAvaliando modelo no conjunto de teste:")
    test_metrics = evaluate_model(y_test, y_pred_test, "Regressão Linear (Teste)")
    
    # 8. Plotar resíduos
    plot_residuals(y_test, y_pred_test, "Regressão Linear")
    
    print("\nProcesso concluído!")


if __name__ == "__main__":
    main()
