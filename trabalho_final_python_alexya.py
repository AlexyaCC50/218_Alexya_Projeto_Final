import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

df = pd.read_csv(r'C:\\Users\\Youth_Space_37\\Desktop\\TRABALHO FINAL\\trabalho_final.csv', encoding='latin1', sep=';', low_memory=False)
#Transformei os dados do power bi pelo excel online, por isso precisei acrescentar o encoding, sep e low memory

# Verificar valores faltantes

print(df.isnull())
print(df.isnull().sum())
print(df.info())

df.dropna(inplace=True)


#Criar os objetos LabelEncoder para cada coluna categórica
print(df.columns)

labelEncoder_order_status = LabelEncoder()
labelEncoder_price = LabelEncoder()
labelEncoder_payment_installments = LabelEncoder()
labelEncoder_payment_value = LabelEncoder()
labelEncoder_review_score = LabelEncoder()

df['order_status'] = labelEncoder_order_status.fit_transform(df['order_status'])
df['price'] = labelEncoder_price.fit_transform(df['price'])
df['payment_installments'] = labelEncoder_payment_installments.fit_transform(df['payment_installments'])
df['payment_value'] = labelEncoder_payment_value.fit_transform(df['payment_value'])
df['review_score'] = labelEncoder_review_score.fit_transform(df['review_score'])

#Normalizando

scaler = MinMaxScaler()
colunas_numericas = ['price', 'freight_value', 'payment_installments', 'payment_value', 'review_score']

df[colunas_numericas] = scaler.fit_transform(df[colunas_numericas])

print(df.head())


#Onehotencoder

categorias = [0,1,3,4,5,6,7,8,9,10,11,13,14,15,16,17,20,22,23]

oneHotEncoder= ColumnTransformer(transformers=[('onehot',OneHotEncoder(handle_unknown='ignore',sparse_output=False),categorias)], remainder='passthrough')

df_encoded_array = oneHotEncoder.fit_transform(df)

print('Formato dos dados codificados: ',df.shape)


df_encoded_array = oneHotEncoder.fit_transform(df)


#Recuperar nomes das colunas one-hot
onehot_columns = oneHotEncoder.named_transformers_['onehot'].get_feature_names_out(input_features=df.columns[categorias])

#Colunas que não foram transformadas
remainder_cols = [col for i, col in enumerate(df.columns) if i not in categorias]

#Combinar nomes das colunas
all_columns = list(onehot_columns) + remainder_cols

#Converter array para DataFrame com nomes das colunas
df_encoded = pd.DataFrame(df_encoded_array, columns=all_columns)

#Converter colunas numéricas para float
for col in colunas_numericas:
    if col in df_encoded.columns:
        df_encoded[col] = pd.to_numeric(df_encoded[col], errors='coerce')


#CLassificação
from sklearn.model_selection import train_test_split


#Previsores: todas as colunas exceto 'problema'
X_classificacao = df_encoded.drop('problema', axis=1)

#classe: coluna 'problema'
y_classificacao = df_encoded['problema']

# Dividir em treino e teste
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_classificacao, y_classificacao, test_size=0.2, random_state=42, stratify=y_classificacao
)

print(f"Classificação - Treino: {X_train_clf.shape}, Teste: {X_test_clf.shape}")

#regressao

df_regressao = df_encoded[df_encoded['problema'] == 0].copy()

X_regressao = df_regressao.drop('price', axis=1)

# Classe price
y_regressao = df_regressao['price']

# Dividir em treino e teste
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_regressao, y_regressao, test_size=0.2, random_state=42
)

print(f"Regressão - Treino: {X_train_reg.shape}, Teste: {X_test_reg.shape}")

#Dataframe para classificação
df_classificacao = df_encoded.copy()

# Separar previsores e classe para classificação
X_classificacao = df_classificacao.drop('problema', axis=1)
y_classificacao = df_classificacao['problema']

#Dataframe para regressão
df_regressao = df_encoded[df_encoded['problema'] == 0].copy()

# Separar previsores e classe para regressão
X_regressao = df_regressao.drop('price', axis=1) 
y_regressao = df_regressao['price']



