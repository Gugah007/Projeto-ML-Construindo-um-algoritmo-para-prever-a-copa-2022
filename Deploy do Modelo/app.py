# Importando as bibliotecas
import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import joblib

# Trazendo nossa imagem para a página
image = Image.open("logo-FIFA-world-cup-qatar-2022-vector-e-png.png")
st.image(image)

# Titulo e Subtitulo da página
st.title("Copa do Mundo 2022")
st.text("Algoritmo de Machine Learning capaz de prever quem vai ganhar a Copa do Mundo 2022")

# Importando nosso csv de seleções
df_selecoes = pd.read_csv("Selecoes2022 - Página1.csv")

# Valores unicos do nosso dataset
todas_selecoes = sorted(df_selecoes['Selecoes'].unique())

# Selecionamos a Primeira Seleção
selecionar_primeira_selecao = st.selectbox('Primeira Seleção', todas_selecoes)
# Esse linha aqui faz com que não escolhemos a mesma seleção 2x
selecionar_b = df_selecoes[df_selecoes['Selecoes'] != selecionar_primeira_selecao]
# Selecionamos a Primeira Seleção
selecionar_segunda_selecao = st.selectbox('Segunda Seleção', selecionar_b)

# Puxar nosso modelo
model = joblib.load('model.pk1')

# Todas as seleções
nome_time = {'France': 0,'Mexico': 1, 'USA': 2,'Belgium': 3,'Yugoslavia': 4,'Brazil': 5,'Romania': 6,'Peru': 7,
            'Argentina': 8,'Chile': 9,'Bolivia': 10,'Paraguay': 11,'Uruguay': 12,'Austria': 13,'Hungary': 14,'Egypt': 15,
            'Switzerland': 16,'Netherlands': 17,'Sweden': 18,'Germany': 19,'Spain': 20,'Italy': 21,'Czechoslovakia': 22,
            'Dutch East Indies': 23,'Cuba': 24,'Norway': 25,'Poland': 26,'England': 27,'Scotland': 28,'Turkey': 29,'Korea Republic': 30,
            'Soviet Union': 31,'Wales': 32,'Northern Ireland': 33,'Colombia': 34,'Bulgaria': 35,'Korea DPR': 36,'Portugal': 37,
            'Israel': 38,'Morocco': 39, 'El Salvador': 40, 'Australia': 41,'Zaire': 42,'Haiti': 43,'Tunisia': 44,'IR Iran': 45,
            'Iran': 46,'Cameroon': 47, 'New Zealand': 48,'Algeria': 49,'Honduras': 50,'Kuwait': 51,'Canada': 52,'Iraq': 53,
            'Denmark': 54,'rn">United Arab Emirates': 55,'Costa Rica': 56,'rn">Republic of Ireland': 57,'Saudi Arabia': 58,
            'Russia': 59,'Greece': 60,'Nigeria': 61,'South Africa': 62,'Japan' : 63,'Jamaica': 64,'Croatia': 65,'Senegal': 66,
            'Slovenia': 67,'Ecuador': 68,'China PR': 69,'rn">Trinidad and Tobago': 70,"Côte d'Ivoire": 71,'rn">Serbia and Montenegro': 72,
            'Angola': 73,'Czech Republic': 74,'Ghana': 75,'Togo': 76,'Ukraine': 77,'Serbia': 78,'Slovakia': 79,
            'rn">Bosnia and Herzegovina': 80,'Iceland': 81,'Panama': 82}

# Importando os dados de campeoes e fazendo a contagem
df_campeoes = pd.read_csv("Campeoes - Campeoes.csv")
campeoes = df_campeoes['Vencedor'].value_counts()

# Função de predição
def predicao(timeA, timeB):
  idA = nome_time[timeA]
  idB = nome_time[timeB]
  campeaoA = campeoes.get(timeA) if campeoes.get(timeA) != None else 0
  campeaoB = campeoes.get(timeB) if campeoes.get(timeB) != None else 0

  x = np.array([idA, idB, campeaoA, campeaoB]).astype('float64')
  x = np.reshape(x, (1,-1))
  _y = model.predict_proba(x)[0]

  text = ('Chance de ' +timeA+' vencer '+timeB+' é {}\nChance de '+timeB+' vencer '+timeA+' e {}\nChance de '+timeA+' e '+timeB+' empatar é {}').format(_y[1]*100,_y[2]*100,_y[0]*100)
  return _y[0], text

# Predict do modelo
prob, text1 = predicao(selecionar_primeira_selecao,selecionar_segunda_selecao)

# Reakuzar predição
if st.button("Realizar predição do Jogo"):
    st.text(text1)
