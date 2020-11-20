import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

df = pd.read_csv('heart.csv')
model = KMeans(4,max_iter=50)
df.age = np.array(df.age).reshape(-1,1)
print(df.age.shape)
preds = model.fit_predict(df.age)
print(model)

# model.predict(df.age)

print(model.cluster_centers_)

df['preds'] = preds

plt.scatter(df['preds'],df['age'],c=df['preds'])
plt.show()
