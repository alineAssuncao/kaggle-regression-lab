"""
Módulo com funções utilitárias para o projeto de regressão.
"""

from .data_processing import load_data, preprocess_data, create_preprocessor, split_data
from .evaluation import evaluate_model, plot_residuals
from .eda import analisar_dados, analisar_valores_ausentes, plot_distribuicao_numerica, plot_correlacao

__all__ = [
    'load_data',
    'preprocess_data',
    'create_preprocessor',
    'split_data',
    'evaluate_model',
    'plot_residuals',
    'analisar_dados',
    'analisar_valores_ausentes',
    'plot_distribuicao_numerica',
    'plot_correlacao'
]