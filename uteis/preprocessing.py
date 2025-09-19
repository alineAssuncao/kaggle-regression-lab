"""
Módulo de pré-processamento de dados para o projeto de regressão de preços de imóveis.

Este módulo contém funções úteis para carregar, limpar e preparar dados para modelagem.
"""
import os
import numpy as np
import pandas as pd
from typing import Tuple, Union, List, Dict, Optional
from pathlib import Path


def carregar_dados(caminho_arquivo: str) -> pd.DataFrame:
    """
    Carrega os dados de um arquivo CSV.
    
    Args:
        caminho_arquivo: Caminho para o arquivo CSV
        
    Returns:
        DataFrame do pandas com os dados carregados
    """
    try:
        return pd.read_csv(caminho_arquivo)
    except Exception as e:
        print(f"Erro ao carregar o arquivo {caminho_arquivo}: {e}")
        raise


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
    missing_data = pd.concat([total, percent], axis=1, 
                           keys=['Total', 'Porcentagem']).sort_values('Porcentagem', ascending=False)
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
    print(f"- Colunas com valores ausentes: {colunas_com_ausentes} ({percentual_colunas_com_ausentes:.1f}% do total)")
    
    if not missing_data.empty:
        print(f"\nDetalhes dos Valores Ausentes:")
        display(missing_data.style.background_gradient(cmap='Reds', subset=['Porcentagem']))
        
        # Análise por tipo de dado
        print("\nAnálise por Tipo de Dado:")
        missing_by_type = df.isnull().sum().groupby(df.dtypes).sum()
        display(missing_by_type)
        
        # Recomendações
        print("\nRecomendações:")
        colunas_para_remover = missing_data[missing_data['Porcentagem'] > limite_porcentagem].index.tolist()
        if colunas_para_remover:
            print(f"\n1. Considerar remover as seguintes colunas (mais de {limite_porcentagem}% de valores ausentes):")
            for col in colunas_para_remover:
                print(f"   - {col} ({missing_data.loc[col, 'Porcentagem']:.1f}% ausentes)")
        
        # Sugestões de preenchimento
        print("\n2. Para as demais colunas, considere:")
        print("   - Para colunas numéricas: preencher com mediana ou média")
        print("   - Para colunas categóricas: criar categoria 'Desconhecido' ou usar a moda")
        
        # Gráfico de barras
        if plotar_grafico and not missing_data.empty:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            plt.figure(figsize=(12, 6))
            missing_gt_limit = missing_data[missing_data['Porcentagem'] > limite_grafico]
            
            if not missing_gt_limit.empty:
                ax = sns.barplot(x=missing_gt_limit.index, 
                                y='Porcentagem', 
                                data=missing_gt_limit,
                                palette='Reds_r')
                
                plt.title(f'Porcentagem de Valores Ausentes por Coluna (acima de {limite_grafico}%)', 
                         fontsize=14, pad=20)
                plt.xticks(rotation=45, ha='right')
                plt.axhline(y=limite_porcentagem, color='red', linestyle='--', 
                           label=f'Limite de {limite_porcentagem}%')
                plt.legend()
                plt.tight_layout()
                plt.show()
            else:
                print(f"\nNenhuma coluna com mais de {limite_grafico}% de valores ausentes para plotar.")
    else:
        print("\nNenhum valor ausente encontrado no dataset!")
    
    print(f"\n{'='*60}\n")
    return missing_data


def preencher_valores_numericos(df: pd.DataFrame, estrategia: str = 'mediana', colunas: list = None) -> pd.DataFrame:
    """
    Preenche valores ausentes em colunas numéricas.
    
    Args:
        df: DataFrame de entrada
        estrategia: Estratégia para preenchimento ('media', 'mediana', 'moda' ou 'constante')
        colunas: Lista de colunas para preencher (se None, preenche todas as numéricas)
        
    Returns:
        DataFrame com valores ausentes preenchidos
    """
    df = df.copy()
    
    if colunas is None:
        colunas = df.select_dtypes(include=[np.number]).columns
    
    for col in colunas:
        if df[col].isnull().any():
            if estrategia == 'media':
                fill_value = df[col].mean()
            elif estrategia == 'mediana':
                fill_value = df[col].median()
            elif estrategia == 'moda':
                fill_value = df[col].mode()[0]
            elif estrategia == 'constante':
                fill_value = 0  # ou outro valor padrão
            else:
                raise ValueError("Estratégia inválida. Use 'media', 'mediana', 'moda' ou 'constante'.")
                
            df[col] = df[col].fillna(fill_value)
            print(f"Preenchidos {df[col].isnull().sum()} valores ausentes na coluna '{col}' com {estrategia} = {fill_value:.2f}")
    
    return df


def converter_categorias(df: pd.DataFrame, colunas_categoricas: list = None, metodo: str = 'onehot') -> pd.DataFrame:
    """
    Converte colunas categóricas para formato numérico.
    
    Args:
        df: DataFrame de entrada
        colunas_categoricas: Lista de colunas categóricas para converter
        metodo: 'onehot' para one-hot encoding ou 'label' para label encoding
        
    Returns:
        DataFrame com as colunas categóricas convertidas
    """
    df = df.copy()
    
    if colunas_categoricas is None:
        colunas_categoricas = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if metodo == 'onehot':
        df = pd.get_dummies(df, columns=colunas_categoricas, drop_first=True)
    elif metodo == 'label':
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        for col in colunas_categoricas:
            df[col] = le.fit_transform(df[col].astype(str))
    else:
        raise ValueError("Método inválido. Use 'onehot' ou 'label'.")
    
    return df


def dividir_dados(X: pd.DataFrame, y: pd.Series, tamanho_teste: float = 0.2, seed: int = 42) -> tuple:
    """
    Divide os dados em conjuntos de treino e teste.
    
    Args:
        X: Variáveis independentes
        y: Variável alvo
        tamanho_teste: Proporção do conjunto de teste
        seed: Semente para reprodutibilidade
        
    Returns:
        Tupla com X_train, X_test, y_train, y_test
    """
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=tamanho_teste, random_state=seed)


def salvar_dados(df: pd.DataFrame, caminho_arquivo: str, **kwargs) -> None:
    """
    Salva o DataFrame em um arquivo.
    
    Args:
        df: DataFrame para salvar
        caminho_arquivo: Caminho do arquivo de saída
        **kwargs: Argumentos adicionais para o método to_csv()
    """
    try:
        # Garante que o diretório existe
        os.makedirs(os.path.dirname(caminho_arquivo), exist_ok=True)
        df.to_csv(caminho_arquivo, index=False, **kwargs)
        print(f"Dados salvos com sucesso em {caminho_arquivo}")
    except Exception as e:
        print(f"Erro ao salvar o arquivo {caminho_arquivo}: {e}")
        raise


if __name__ == "__main__":
    # Exemplo de uso
    print("Módulo de pré-processamento carregado com sucesso!")
    print("Funções disponíveis:")
    print("- carregar_dados(caminho_arquivo)")
    print("- analisar_dados(df, mostrar_amostra=True)")
    print("- analisar_valores_ausentes(df, limite_porcentagem=30.0)")
    print("- preencher_valores_numericos(df, estrategia='mediana', colunas=None)")
    print("- converter_categorias(df, colunas_categoricas=None, metodo='onehot')")
    print("- dividir_dados(X, y, tamanho_teste=0.2, seed=42)")
    print("- salvar_dados(df, caminho_arquivo, **kwargs)")
