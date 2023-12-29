from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pylab as plb
import pandas as pd

#Linhas e Colunas MiniSom = 5√n


#Carregando dados e separando classes
dados = pd.read_csv('wines.csv')
classe = dados.Class
dados = dados.drop('Class', axis=1)

#Normalizando dados
normalizador = MinMaxScaler()
dados = normalizador.fit_transform(dados)

#Criando modelo
som = MiniSom(x=20, y=20, input_len=13, learning_rate=2, sigma=3.5, random_seed=5)

#Inicializando pesos com base nos dados
som.random_weights_init(dados)

#Treinando modelo
som.train_random(data=dados, num_iteration=250)

#Plotando mapa
markers = ['o', 's', 'D']
color = ['r', 'g', 'b']
plb.pcolor(som.distance_map().T)
plb.colorbar()

#Alterando valores da classe para corresponder a lista de cor e marcadores
classe[classe == 1] = 0
classe[classe == 2] = 1
classe[classe == 3] = 2

for i, x in enumerate(dados):
    w = som.winner(x)
    plb.plot(w[0] + 0.5, w[1] + 0.5, markers[classe[i]], markerfacecolor ='None',
             markersize=10, markeredgecolor=color[classe[i]], markeredgewidth=2)

plb.show()
