"""
Pacote uteis - Módulos úteis para o projeto de regressão de preços de imóveis.

Este pacote contém módulos para pré-processamento, visualização e outras utilidades.
"""

# Importa as funções do módulo preprocessing para facilitar o acesso
from .preprocessing import (
    carregar_dados,
    analisar_dados,
    analisar_valores_ausentes,
    preencher_valores_numericos,
    converter_categorias,
    dividir_dados,
    salvar_dados
)

__all__ = [
    'carregar_dados',
    'analisar_dados',
    'analisar_valores_ausentes',
    'preencher_valores_numericos',
    'converter_categorias',
    'dividir_dados',
    'salvar_dados'
]
