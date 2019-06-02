import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dji_df = pd.read_csv('database/DJIndex.csv')
perc_cng = np.array(dji_df.Close)[1:] / np.array(dji_df.Close)[:-1]
perc_cng = (perc_cng - np.mean(perc_cng)) / np.std(perc_cng, ddof=1)
print(perc_cng)
# perc_cng = np.insert(perc_cng, 0, 0)
# dji_df['perc_cng'] = perc_cng
#
# print(dji_df['perc_cng'])

# plt.scatter(x=perc_cng, y=range(0, len(dji_df)-1))
print(perc_cng.min(), perc_cng.max())
plt.hist(x=perc_cng, bins=10, density=True, log=True)
plt.show()
