from __future__ import print_function
import time
import pandas as pd
from sklearn.cluster import KMeans
from yellowbrick.cluster.elbow import kelbow_visualizer


#convert to pandas dataframes
feat_cols = ['Domain','Year','Haptic','VR','AR','2D','NA','Touchpad','Vibration','Wind','Temperature','Audio','Gizmo','Tracking','NA','Object','Eye','Hand','Head','Body','NA','Survey','StateOfTheArt','Review','System','Simulation','Experiment','Analysis']
df = pd.DataFrame(pd.read_csv('./papers.csv'),columns=feat_cols)

print('Size of the dataframe: {}'.format(df.shape))

# Instantiate the clustering model and visualizer

#min not found
elbow=kelbow_visualizer(KMeans(), df, k=(1,6))

#first found
elbow2=kelbow_visualizer(KMeans(), df, k=(1,7))

#second, different than first, found
elbow3=kelbow_visualizer(KMeans(), df, k=(1,15))


#https://www.scikit-yb.org/en/latest/api/cluster/elbow.html
