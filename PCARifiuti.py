import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


#Regione Veneto
# Produzione di rifiuti [migliaia di tonnellate]
# reddito disponibile delle famiglie consumatrici per abitante https://statistica.regione.veneto.it/excel/PIL/Indicatori_pro_capite_20210127.xlsx
# età media https://www.tuttitalia.it/veneto/statistiche/indici-demografici-struttura-popolazione/

name_dict = {
            'Target': ['2019','2018','2017','2016'],
            'Produzione_Rifiuti': [2403.3,2363.2,2334.8,2389.2],
            'Popolazione_Residente_Media': [4881862,4882763,4883475,4888332],
            'Redditi_Disponibili': [20746.1,20598.8,20143.7,19589.4],
            'Età_media': [45.6,45.4,45.1,44.8],
            'Tasso_Disoccupazione': [5.6,6.4,6.3,6.8]
          }

df = pd.DataFrame(name_dict)

features = ['Produzione_Rifiuti', 'Popolazione_Residente_Media', 'Redditi_Disponibili', 'Età_media']
# Separating out the features
x = df.loc[:, features].values

# Separating out the target
y = df.loc[:,['Target']].values

# Standardizing the features
x = StandardScaler().fit_transform(x)

pca = PCA(n_components=2)

principalComponents = pca.fit_transform(x)


principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

finalDf = pd.concat([principalDf, df[['Target']]], axis = 1)


fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = ['2019','2018','2017','2016']
colors = ['r', 'g', 'b', 'y']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['Target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
plt.show()