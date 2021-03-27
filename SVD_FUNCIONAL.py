import pandas as pd
import numpy as np

matriz = [[1,0],[0,np.sqrt(2)],[0,np.sqrt(2)]]
A = pd.DataFrame(matriz)
AT = A.T
ATA = AT.dot(A)
AAT = A.dot(AT)

val_ATA,vet_ATA = np.linalg.eigh(ATA)
E = np.sqrt(val_ATA)
E = np.diag(E)
E = pd.DataFrame(E)
S = pd.DataFrame(vet_ATA)
S = S.T

val_AAT,vet_AAT = np.linalg.eigh(AAT)
Q = pd.DataFrame(vet_AAT)
col = val_AAT.round(3)
Q.columns = col.astype(str)

#tentando igualar os indices dos vetores com valores::::
ind = []
for i in val_ATA.round(3):
    if i in col:
        ind.append(i)

#deve retornar apenas os autovetores dos autovalores encontrados
ind = np.array(ind)
Q = Q[ind.astype(str)]

#acertando as colunas de Q para a multiplicacao::
Q.columns = range(0,len(ind))
U = Q.dot(E)
m_final = U.dot(S)
#####matriz reconstruida

        

