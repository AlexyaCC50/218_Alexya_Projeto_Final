import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

df = pd.read_csv(r'C:\\Users\\Youth_Space_37\\Desktop\\TRABALHO FINAL\\olist_df_unificado.csv', encoding='latin1', sep=';', low_memory=False)

#Transformei os dados do power bi pelo excel online, por isso precisei acrescentar o encoding, sep e low memory

#...Criação do dataset contendo cidade, estado, categoria, valor, frete, datas, tipo de pagamento, número de parcelas, avaliação do cliente

df2= df[['customer_city','customer_state','product_category_name','payment_type','payment_installments','order_status','order_delivered_customer_date','order_estimated_delivery_date','price','freight_value','review_score']]


# #Criar a variavel problema baeado no atraso com a entrega e a nota menor<2

df2['atraso_entrega'] = np.where(df2['order_delivered_customer_date'] >df2['order_estimated_delivery_date'], 1, 0)


# print(df2[['order_delivered_customer_date','order_estimated_delivery_date','atraso_entrega']].head())

df2['problema'] = np.where((df2['atraso_entrega'] ==1) | (df2['review_score'] <= 2),1 ,0)

print(df2.head())

# print(df2[['problema','atraso_entrega','review_score']].head(15))

# #Verificar dados faltantes

print(df2.isnull().sum())

#Exportar o dataset tratado para um novo csv

df2.to_csv('df2.csv', sep=';',index=False, encoding='latin1')  #df2=dados tratados

