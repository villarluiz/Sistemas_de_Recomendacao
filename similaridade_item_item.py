import pandas as pd
import numpy as np

#tabela precisa ser invertida -item<->item-
tabela = pd.read_csv('matrizteste.txt', header=None)
tabela = tabela.fillna(0)
tabela = tabela.to_numpy()
tabela = tabela.tolist()

#i -> linha
#j -> coluna

#retorna media de um vetor sem contar os NA
def media_nan(x):
    linha = []
    for j in range(len(x)):
            if x[j] != 0.0:
                linha.append(x[j])
    return (sum(linha)/len(linha))



def remove_bias(x):
    tabelasembias = []
    for i1 in range(len(x)):
        linha1 = []
        for j1 in range(len(x[0])):
            if x[i1][j1] != 0.0:
                linha1.append(round(x[i1][j1] - media_nan(x[i1]),4))
            else:
                linha1.append(x[i1][j1])
        tabelasembias.append(linha1)    
    return tabelasembias

#x e y sao vetores
#ja adaptado para matrizes item-item?
def pearson(x,y):
    numerador = []
    vetor1 = []
    vetor2 = []
    for i2 in range(len(x)):
        for j2 in range(len(x[0]))
            if x[j2] and y[j2] != 0.0:
                numerador.append((x[j2] - media_nan(x))*(y[j2] - media_nan(y)))
                vetor1.append(x[j2] - media_nan(x))
                vetor2.append(y[j2] - media_nan(y))
    return sum(numerador)/(np.sqrt(sum(np.power(vetor1,2)))*np.sqrt(sum(np.power(vetor2,2))))

def cosine(x,y):
    numerador = []
    vetor1 = []
    vetor2 = []
    for i3 in range(len(x)):
        if x[i3] and y[i3] != 0.0:
            numerador.append(x[i3]*y[i3])
            vetor1.append(x[i3])
            vetor2.append(y[i3])
    return sum(numerador)/(np.sqrt(sum(np.power(vetor1,2)))*np.sqrt(sum(np.power(vetor2,2))))


#entradas -> x=tabela , y=similaridade, z=centrada ou raw
#funcao procura os valores NA(nesse caso 0) e usa da matriz de similaridade pra prever NA
def substituir(x,y,z='raw'):
    tabela_final = []
    for i4 in range(len(x)):
        linha_final = []
        for j4 in range(len(x[0])):
            numerador_final = []
            denominador_final = []
            if x[i4][j4] == 0.0:
                if z == 'centrada':
                    for k in range(len(x)):
                        w = y(x[i4],x[k])
                        if round(w,4) != 1.0:
                            if round(w,4) >= 0.8:
                                numerador_final.append(w*     (x[k][j4] - media_nan(x[k]))     )
                                denominador_final.append(w)
                    linha_final.append(round(sum(numerador_final)/sum(denominador_final) + media_nan(x[i4]),2))
                else:
                    for k in range(len(x)):
                        w = y(x[i4],x[k])
                        if round(w,4) != 1.0:
                            if round(w,4) >= 0.8:
                                numerador_final.append(w*(x[k][j4]))
                                denominador_final.append(w)
                    linha_final.append(round(sum(numerador_final)/sum(denominador_final),2))
            else:
                linha_final.append(x[i4][j4])
        tabela_final.append(linha_final)
    return tabela_final           
