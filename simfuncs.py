import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.impute import KNNImputer
#Leitura do arquivo teste
tabela = pd.read_csv('matrizteste.txt',header=None)
print(tabela)
#Ajuste da tabela de testes para rotina
tabela = tabela.to_numpy()
#calculo da media de cada linha (usuario)
media_user = np.nanmean(tabela, axis=1)
#retirar a media de cada usuario pra nao ter bias
tabelasembias = tabela - media_user.reshape(-1,1)

#Imputer para substituir os valores NaN usando nan_euclidean para achar NN e\
    #utilizando media ponderada dos valores dos NNs para a predicao
imputer = KNNImputer(n_neighbors=2, weights='distance')
imputer.fit_transform(tabelasembias)
tabela_predicao = imputer.transform(tabelasembias)

#adicionando de volta a bias dos usuarios
tabela_final = tabela_predicao + media_user.reshape(-1,1)

print(pd.DataFrame(tabela_final))






#falta implementar as contas para calcular as predicoes
 #usar função tabela.corrwith(method='pearson') pra achar as correlações\
    #entre as colunas






