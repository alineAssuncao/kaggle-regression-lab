"""
Módulo para Análise Exploratória de Dados (EDA).

Este módulo contém funções para análise e visualização de dados.
"""

from typing import Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def analisar_dados(df: pd.DataFrame, mostrar_amostra: bool = True) -> None:
    """
    Exibe informações básicas sobre o DataFrame.
    
    Args:
        df: DataFrame a ser analisado
        mostrar_amostra: Se True, mostra as primeiras linhas do DataFrame
    """
    print(f"\n{'='*50}")
    print(f"ANÁLISE INICIAL DOS DADOS")
    print(f"{'='*50}")
    
    print(f"\nDimensões do dataset: {df.shape}")
    print(f"\nTipos de dados:")
    print(df.dtypes.value_counts())
    
    if mostrar_amostra:
        print("\nAmostra dos dados:")
        display(df.head())
    
    print(f"\n{'='*50}")


def analisar_valores_ausentes(
    df: pd.DataFrame, 
    limite_porcentagem: float = 30.0,
    plotar_grafico: bool = True,
    limite_grafico: float = 5.0
) -> pd.DataFrame:
    """
    Analisa e retorna colunas com valores ausentes, com opção de visualização gráfica.
    
    Args:
        df: DataFrame para análise
        limite_porcentagem: Limiar para considerar colunas com muitos valores ausentes
        plotar_grafico: Se True, plota um gráfico das colunas com mais valores ausentes
        limite_grafico: Limite mínimo de porcentagem de valores ausentes para aparecer no gráfico
        
    Returns:
        DataFrame com informações sobre valores ausentes
    """
    # Cálculo dos valores ausentes
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum() / df.shape[0] * 100).sort_values(ascending=False)
    missing_data = pd.concat(
        [total, percent], 
        axis=1, 
        keys=['Total', 'Porcentagem']
    ).sort_values('Porcentagem', ascending=False)
    
    missing_data = missing_data[missing_data['Total'] > 0]
    
    # Cálculo de estatísticas resumidas
    total_linhas = df.shape[0]
    total_colunas = df.shape[1]
    colunas_com_ausentes = len(missing_data)
    percentual_colunas_com_ausentes = (colunas_com_ausentes / total_colunas) * 100
    
    print(f"\n{'='*60}")
    print(f"ANÁLISE DE VALORES AUSENTES")
    print(f"{'='*60}")
    
    print(f"\nVisão Geral:")
    print(f"- Total de linhas: {total_linhas}")
    print(f"- Total de colunas: {total_colunas}")
    print(f"- Colunas com valores ausentes: {colunas_com_ausentes} "
          f"({percentual_colunas_com_ausentes:.1f}% do total)")
    
    if not missing_data.empty:
        print(f"\nDetalhes dos Valores Ausentes:")
        display(missing_data.style.background_gradient(cmap='Reds', subset=['Porcentagem']))
        
        # Análise por tipo de dado
        print("\nAnálise por Tipo de Dado:")
        missing_by_type = df.isnull().sum().groupby(df.dtypes).sum()
        display(missing_by_type)
        
        # Recomendações
        print("\nRecomendações:")
        colunas_para_remover = missing_data[
            missing_data['Porcentagem'] > limite_porcentagem
        ].index.tolist()
        
        if colunas_para_remover:
            print(f"\n1. Considerar remover as seguintes colunas "
                  f"(mais de {limite_porcentagem}% de valores ausentes):")
            for col in colunas_para_remover:
                print(f"   - {col} ({missing_data.loc[col, 'Porcentagem']:.1f}% ausentes)")
        
        # Sugestões de preenchimento
        print("\n2. Para as demais colunas, considere:")
        print("   - Para colunas numéricas: preencher com mediana ou média")
        print("   - Para colunas categóricas: criar categoria 'Desconhecido' ou usar a moda")
        
        # Gráfico de barras
        if plotar_grafico and not missing_data.empty:
            plt.figure(figsize=(12, 6))
            missing_gt_limit = missing_data[missing_data['Porcentagem'] > limite_grafico]
            
            if not missing_gt_limit.empty:
                ax = sns.barplot(
                    x=missing_gt_limit.index, 
                    y='Porcentagem', 
                    data=missing_gt_limit,
                    palette='Reds_r'
                )
                
                plt.title(
                    f'Porcentagem de Valores Ausentes por Coluna (acima de {limite_grafico}%)',
                    fontsize=14,
                    pad=20
                )
                plt.xticks(rotation=45, ha='right')
                plt.axhline(
                    y=limite_porcentagem,
                    color='red',
                    linestyle='--',
                    label=f'Limite de {limite_porcentagem}%'
                )
                plt.legend()
                plt.tight_layout()
                plt.show()
            else:
                print(f"\nNenhuma coluna com mais de {limite_grafico}% "
                      f"de valores ausentes para plotar.")
    else:
        print("\nNenhum valor ausente encontrado no dataset!")
    
    print(f"\n{'='*60}\n")
    return missing_data


def plot_distribuicao_numerica(df: pd.DataFrame, colunas: list = None) -> None:
    """
    Plota a distribuição de variáveis numéricas.
    
    Args:
        df: DataFrame contendo os dados
        colunas: Lista de colunas para plotar (se None, plota todas as numéricas)
    """
    if colunas is None:
        colunas = df.select_dtypes(include=[np.number]).columns.tolist()
    
    n_cols = 3
    n_linhas = (len(colunas) + n_cols - 1) // n_cols
    
    plt.figure(figsize=(15, 4 * n_linhas))
    
    for i, col in enumerate(colunas, 1):
        plt.subplot(n_linhas, n_cols, i)
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribuição de {col}')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()


def plot_correlacao(df: pd.DataFrame, metodo: str = 'pearson', 
                   tamanho_figura: tuple = (12, 10)) -> None:
    """
    Plota a matriz de correlação das variáveis numéricas.
    
    Args:
        df: DataFrame contendo os dados
        metodo: Método de correlação ('pearson', 'spearman' ou 'kendall')
        tamanho_figura: Tamanho da figura do gráfico
    """
    # Selecionar apenas colunas numéricas
    df_numeric = df.select_dtypes(include=[np.number])
    
    # Calcular matriz de correlação
    corr = df_numeric.corr(method=metodo)
    
    # Configurar a máscara para o triângulo superior
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Configurar a figura
    plt.figure(figsize=tamanho_figura)
    
    # Gerar mapa de calor
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=.5,
        cbar_kws={"shrink": .8}
    )
    
    plt.title(f'Matriz de Correlação ({metodo.capitalize()})', fontsize=16)
    plt.tight_layout()
    plt.show()
