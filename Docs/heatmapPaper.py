import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
import pandas as pd

feat_cols = ['Haptic','VR','AR','2D','NA','Touchpad','Vibration','Wind','Temp.','Audio','Gizmo','Tracking','NA','Object','Eye','Hand','Head','Body','NA','Survey','StOfArt','Review','System','Simu.','Exp.','Analysis']
feat_row=['Basic Research','Medicine','Transportation','Physics','UX', 'Cultural Heritage', 'Industry']
dat=[[4,9,2,0,5,0,1,0,0,0,2,2,13,0,0,2,1,1,13,1,3,8,2,0,1,1],
     [16,14,1,1,0,0,2,0,0,3,9,9,8,0,0,9,2,2,8,3,1,4,6,3,0,0],
     [6,10,0,2,0,1,2,0,0,2,10,8,1,0,1,6,5,2,3,0,0,0,2,6,0,3],
     [14,15,1,3,0,1,3,0,0,1,16,17,0,0,0,16,2,1,0,0,0,0,5,1,10,1],
     [12,20,0,2,0,0,4,1,2,6,19,18,3,1,1,16,3,3,4,0,0,1,9,2,3,7],
     [3,9,6,0,0,3,0,0,0,7,7,7,2,0,0,5,3,4,5,1,0,0,7,1,1,2],
     [12,16,5,1,0,2,1,0,0,4,10,12,6,3,0,11,5,3,6,3,0,2,8,3,0,2]]
color=["darkRed","crimson","indianRed","coral","lightSalmon","yellow","lawnGreen","limeGreen","olive","blue","darkBlue"]
data=pd.DataFrame(dat,index=feat_row, columns=feat_cols)


hmap=sns.heatmap(data, linewidth=0.1, linecolor="black", vmin=0, vmax=20, annot=True, cmap="coolwarm")
sns.set(rc={'figure.figsize':(25,7)})
plt.show()
fig = hmap.get_figure()
fig.savefig("heatmap.pdf")
