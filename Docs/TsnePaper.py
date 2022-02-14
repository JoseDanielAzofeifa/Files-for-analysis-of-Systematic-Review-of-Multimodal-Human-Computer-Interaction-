from __future__ import print_function
import time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

#load data
mnist = pd.read_csv('./papers.csv')
X = mnist.drop("Domain",axis=1)
y = mnist['Domain']
#print(X.shape, y.shape)


def domains(r):
    k=[]
    for i in r:
        if i==1:
            k.append("Basic Research")
        elif i==2:
            k.append("Medicine")
        elif i==3:
            k.append("Transportation")
        elif i==4:
            k.append("Physics")
        elif i==5:
            k.append("User Experience")
        elif i==6:
            k.append("Cultural Heritage")
        else:
            k.append("Industry")

    return k

ar= X.drop("Year",axis=1)
array = np.array(ar)
weight = np.sum(array, axis=1)

#convert to pandas dataframes
feat_cols = ['Year','Haptic','VR','AR','2D','NA','Touchpad','Vibration','Wind','Temperature','Audio','Gizmo','Tracking','NA','Object','Eye','Hand','Head','Body','NA','Survey','StateOfTheArt','Review','System','Simulation','Experiment','Analysis']
df = pd.DataFrame(X,columns=feat_cols)
df['y'] = y
df['Domain'] = df['y'].apply(lambda i: str(i))
df['Domains']=domains(y)
df['weight']=weight
X, y = None, None
print('Size of the dataframe: {}'.format(df.shape))

#randomize for reproducability of the results
np.random.seed(112)
rndperm = np.random.permutation(df.shape[0])
N = 112
df_subset = df.loc[rndperm[:N],:].copy()
data_subset = df_subset[feat_cols].values

"""
#first look
fig = plt.figure( figsize=(16,7) )
for i in range(0,15):
    ax = fig.add_subplot(3,5,i+1, title="Digit: {}".format(str(df.loc[rndperm[i],'label'])) )
    ax.matshow(df.loc[rndperm[i],feat_cols].values.reshape((28,28)).astype(float))
plt.show()

#PCA
pca = PCA(n_components=3)
pca_result = pca.fit_transform(data_subset)
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

#look pca results 2D
df_subset['pca-one'] = pca_result[:,0]
df_subset['pca-two'] = pca_result[:,1] 
df_subset['pca-three'] = pca_result[:,2]
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="pca-one", y="pca-two",
    hue="y",
    palette=sns.color_palette("hls", 10),
    data=df_subset,
    legend="full",
    alpha=0.3
)
plt.show()
"""

#T-sne
time_start = time.time()
#tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne = TSNE(n_components=2, verbose=1, perplexity=1500, n_iter=100000)
tsne_results = tsne.fit_transform(data_subset)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

#look T-sne results 2D
df_subset['tsne-2d-one'] = tsne_results[:,0]
df_subset['tsne-2d-two'] = tsne_results[:,1]
plt.figure(figsize=(25,7))
sns.set(rc={'figure.figsize':(25,7)})

tsne_plt=sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="Domains",
    #size="weight",
    #sizes=(500, 1000),
    palette=sns.color_palette("hls", 7),
    data=df_subset,
    alpha=0.8,
 
)
"""
tsne_plt=sns.kdeplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="Domains",
    #size="weight",
    #sizes=(500, 1000),
    palette=sns.color_palette("hls", 7),
    data=df_subset,
    alpha=0.8, 
)
"""


plt.show()
fig = tsne_plt.get_figure()
fig.savefig("tsne.pdf")




