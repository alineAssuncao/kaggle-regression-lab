"""
Módulo para processamento de dados do projeto de regressão.

Este módulo contém funções para carregar, limpar e transformar dados para modelagem.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, List, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def load_data(data_path: Union[str, Path], train_file: str = 'train.csv', 
             test_file: str = 'test.csv') -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carrega os dados de treino e teste.
    
    Args:
        data_path: Caminho para o diretório contendo os dados
        train_file: Nome do arquivo de treino
        test_file: Nome do arquivo de teste
        
    Returns:
        Tupla contendo os DataFrames de treino e teste
    """
    data_path = Path(data_path)
    train_df = pd.read_csv(data_path / train_file)
    test_df = pd.read_csv(data_path / test_file)
    return train_df, test_df


def preprocess_data(df: pd.DataFrame, target_column: Optional[str] = None,
                   drop_high_missing: bool = True, missing_threshold: float = 0.8) -> tuple[pd.DataFrame, list, list]:
    """
    Realiza o pré-processamento inicial dos dados.
    
    Args:
        df: DataFrame com os dados a serem pré-processados
        target_column: Nome da coluna alvo (opcional)
        drop_high_missing: Se True, remove colunas com muitos valores ausentes
        missing_threshold: Limiar para considerar colunas com muitos valores ausentes
        
    Returns:
        Tuple contendo o DataFrame processado, lista de features numéricas e categóricas
    """
    df = df.copy()
    
    # Identificar colunas numéricas e categóricas
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Remover a coluna alvo das features
    if target_column and target_column in numeric_features:
        numeric_features.remove(target_column)
    elif target_column and target_column in categorical_features:
        categorical_features.remove(target_column)
    
    # Remover colunas com muitos valores ausentes
    if drop_high_missing:
        missing_values = df.isnull().sum() / len(df)
        columns_to_drop = missing_values[missing_values > missing_threshold].index.tolist()
        df = df.drop(columns=columns_to_drop)
        
        # Atualizar listas de features
        numeric_features = [col for col in numeric_features if col in df.columns]
        categorical_features = [col for col in categorical_features if col in df.columns]
    
    return df, numeric_features, categorical_features


def create_preprocessor(numeric_features: list, categorical_features: list,
                       numeric_strategy: str = 'median', 
                       categorical_strategy: str = 'most_frequent',
                       scale_numeric: bool = True) -> ColumnTransformer:
    """
    Cria um pré-processador para as features numéricas e categóricas.
    
    Args:
        numeric_features: Lista de features numéricas
        categorical_features: Lista de features categóricas
        numeric_strategy: Estratégia para imputação de valores numéricos
        categorical_strategy: Estratégia para imputação de valores categóricos
        scale_numeric: Se True, aplica StandardScaler nas features numéricas
        
    Returns:
        ColumnTransformer configurado
    """
    # Pipeline para features numéricas
    numeric_steps = [
        ('imputer', SimpleImputer(strategy=numeric_strategy))
    ]
    
    if scale_numeric:
        numeric_steps.append(('scaler', StandardScaler()))
    
    numeric_transformer = Pipeline(steps=numeric_steps)
    
    # Pipeline para features categóricas
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=categorical_strategy, fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Criar o ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'  # Remove colunas não especificadas
    )
    
    return preprocessor


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, 
              random_state: int = 42, stratify: Optional[np.ndarray] = None) -> tuple:
    """
    Divide os dados em conjuntos de treino e teste.
    
    Args:
        X: Features
        y: Variável alvo
        test_size: Proporção do conjunto de teste (entre 0 e 1)
        random_state: Semente para reprodutibilidade
        stratify: Se não for None, os dados são divididos de forma estratificada
                 usando esta variável como classe
        
    Returns:
        Tuple com X_train, X_test, y_train, y_test
    """
    return train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=stratify
    )


def preencher_valores_numericos(df: pd.DataFrame, colunas: List[str] = None, 
                              estrategia: str = 'mediana', 
                              valor_constante: float = 0) -> pd.DataFrame:
    """
    Preenche valores ausentes em colunas numéricas.
    
    Args:
        df: DataFrame de entrada
        colunas: Lista de colunas para preencher (se None, preenche todas as numéricas)
        estrategia: Estratégia para preenchimento ('media', 'mediana', 'moda' ou 'constante')
        valor_constante: Valor a ser usado quando a estratégia for 'constante'
        
    Returns:
        DataFrame com valores ausentes preenchidos
    """
    df = df.copy()
    
    if colunas is None:
        colunas = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in colunas:
        if df[col].isnull().any():
            if estrategia == 'media':
                fill_value = df[col].mean()
            elif estrategia == 'mediana':
                fill_value = df[col].median()
            elif estrategia == 'moda':
                fill_value = df[col].mode()[0]
            elif estrategia == 'constante':
                fill_value = valor_constante
            else:
                raise ValueError(
                    "Estratégia inválida. Use 'media', 'mediana', 'moda' ou 'constante'."
                )
                
            df[col] = df[col].fillna(fill_value)
    
    return df


def converter_categorias(df: pd.DataFrame, colunas: List[str] = None, 
                        metodo: str = 'onehot',
                        drop_first: bool = True) -> pd.DataFrame:
    """
    Converte colunas categóricas para formato numérico.
    
    Args:
        df: DataFrame de entrada
        colunas: Lista de colunas categóricas para converter
        metodo: 'onehot' para one-hot encoding ou 'label' para label encoding
        drop_first: Se True, remove a primeira categoria no one-hot encoding
        
    Returns:
        DataFrame com as colunas categóricas convertidas
    """
    df = df.copy()
    
    if colunas is None:
        colunas = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if metodo == 'onehot':
        df = pd.get_dummies(df, columns=colunas, drop_first=drop_first)
    elif metodo == 'label':
        le = LabelEncoder()
        for col in colunas:
            df[col] = le.fit_transform(df[col].astype(str))
    else:
        raise ValueError("Método inválido. Use 'onehot' ou 'label'.")
    
    return df
