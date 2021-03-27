import pandas as pd
import numpy as np

#####SVD Via NumPy:
#matriz de teste tirada do exemplo na dissertação de mestrado
matriztest2 = {'Insterstelar' :[-1,-1,0,-1,0,1,1,1,1,1,1,1,1],
               'A Origem':[-1,-1,-1,-1,0,1,1,1,1,1,0,1,1],
               'Matrix':[0,-1,-1,0,0,1,0,1,1,1,1,1,1],
               'Alien':[1,1,1,1,1,0,-1,-1,0,1,1,0,1],
               'Halloween':[1,1,1,1,1,0,0,-1,-1,1,1,1,1]}
#dataframe pra visualizar
dt = pd.DataFrame(data=matriztest2)
#decompondo em svd
Q,E,S = np.linalg.svd(dt,full_matrices=False)
Q = pd.DataFrame(Q)
#necessario fazer a diagonalizacao de E pois ele retorna apenas o array com \
#os valores singulares ao invés da matriz
E = np.diag(E)
E = pd.DataFrame(E)
S = pd.DataFrame(S)
#reconstrucao:
Vt = E.dot(S)
M = Q.dot(Vt)
#arredondamento pra sumir com as aproximacoes infinitas:
M = M.round()
M = M.astype(int)
#atribuindo os filmes as respectivas colunas::
M.columns = dt.columns


#####SVD Manual:
##calcular a transposta da matriz:
dtT = dt.T

##autovalores e autovetores:
MTM = dtT.dot(dt)
val_MTM,vet_MTM = np.linalg.eigh(MTM)
#raiz dos autovalores para obter valores singulares:
E = np.sqrt(val_MTM)
#diagonalizando e formatando:
E = np.diag(E)
E = pd.DataFrame(E)
S = pd.DataFrame(vet_MTM)
#invertendo a matriz "S" (autovetores MTM)
S = S.T

##definindo Q da fatoracao
#Q representa os autovetores normalizados de MMT:
MMT = dt.dot(dtT)
val_MMT,vet_MMT = np.linalg.eigh(MMT)

#por definicao, a funcao retorna os vetores normalizados, ou seja, ortogonais
#vamos buscar as colunas utilizadas pelos autovalores (E):
Q = pd.DataFrame(vet_MMT)
colunas = val_MMT.round(3)
Q.columns = colunas.astype(str)

#definido os índices das colunas como seus respectivos autovalores,
#vamos seleciona-los:
ind = []
for i in val_MTM.round(3):
    if i in colunas:
        ind.append(i)

#selecionando agora apenas os autovetores associados a E:
ind = np.array(ind)
Q = Q[ind.astype(str)]

#acertando os indices das colunas pra funcao nao confundir as dimensoes:
Q.columns = range(0,len(ind))

#obtemos a decomposicao QESt, vamos reconstruir a matriz original:
U = Q.dot(E)
m_final = U.dot(S)
#percebe-se que esse método não retorna uma matriz satisfatória,
#pois são feitas MUITAS aproximações e tal matriz possui calculos complicados,
#para matrizes muito esparças, o método np.linalg.svd e satisfatorio.



