import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

data = {
            'Name': ['2019','2018','2017','2016','2015'],
            'Produzione_Rifiuti': [2403.3,2363.2,2334.8,2389.2, 1800],
            'Popolazione_Residente_Media': [4881862,4882763,4883475,4888332, 4500000],
            'Redditi_Disponibili': [20746.1,20598.8,20143.7,19589.4, 18000],
            'Età_Media': [45.6,45.4,45.1,44.8, 33],
            'Tasso_Disoccupazione': [5.6,6.4,6.3,6.8, 3.1]
          }

df = pd.DataFrame(data,columns=['Name','Produzione_Rifiuti','Popolazione_Residente_Media','Redditi_Disponibili','Età_media','Tasso_Disoccupazione'])

corrMatrix = df.corr()
sn.heatmap(corrMatrix, annot=True)
plt.show()

