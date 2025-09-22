"""
Módulo para avaliação de modelos de regressão.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = 'Modelo') -> dict:
    """
    Avalia um modelo de regressão com várias métricas.
    
    Args:
        y_true: Valores reais
        y_pred: Valores previstos
        model_name: Nome do modelo para exibição
        
    Returns:
        Dicionário com as métricas calculadas
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2
    }
    
    # Imprimir métricas formatadas
    print(f"\n{'='*50}")
    print(f"Avaliação do {model_name}")
    print(f"{'='*50}")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    print("="*50)
    
    return metrics


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = 'Modelo') -> None:
    """
    Plota os resíduos de um modelo de regressão.
    
    Args:
        y_true: Valores reais
        y_pred: Valores previstos
        model_name: Nome do modelo para o título do gráfico
    """
    residuals = y_true - y_pred
    
    plt.figure(figsize=(12, 5))
    
    # Gráfico de dispersão dos resíduos
    plt.subplot(1, 2, 1)
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title(f'Resíduos vs Valores Preditos - {model_name}')
    plt.xlabel('Valores Preditos')
    plt.ylabel('Resíduos')
    
    # Histograma dos resíduos
    plt.subplot(1, 2, 2)
    plt.hist(residuals, bins=30, edgecolor='black')
    plt.title(f'Distribuição dos Resíduos - {model_name}')
    plt.xlabel('Resíduos')
    plt.ylabel('Frequência')
    
    plt.tight_layout()
    plt.show()


def compare_models(metrics_dict: dict) -> pd.DataFrame:
    """
    Compara as métricas de vários modelos em um DataFrame.
    
    Args:
        metrics_dict: Dicionário com os nomes dos modelos como chaves e 
                     dicionários de métricas como valores
                     
    Returns:
        DataFrame com as métricas de todos os modelos
    """
    return pd.DataFrame(metrics_dict).T


def plot_feature_importance(model, feature_names: list, top_n: int = 20) -> None:
    """
    Plota as features mais importantes de um modelo.
    
    Args:
        model: Modelo treinado com atributo feature_importances_
        feature_names: Lista com os nomes das features
        top_n: Número de features mais importantes a serem exibidas
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]
    
    plt.figure(figsize=(10, 8))
    plt.title('Importância das Features')
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Importância Relativa')
    plt.tight_layout()
    plt.show()
