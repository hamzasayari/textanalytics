import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == '__main__':
	
   df1=pd.DataFrame.from_csv('hilinger_distance_2D_sp.csv')
   df2=pd.DataFrame.from_csv('users_categ.csv')
   df=df1.join(df2)
   print(df.iloc[1:,:].head())
#y=df2["user_sentiment_label"].cat.codes
    #df2.user_sentiment_label = pd.Categorical(df2.user_sentiment_label).codes
#y=pd.get_dummies(df2["user_sentiment_label"])
#df2.rename(columns={'positive': 'user_sentiment_label'}, inplace=True)
   #df=df[df.user_sentiment_label != 'irrelevant']

   #df=df[df.user_sentiment_label != 'neutral']
#y=df2["user_sentiment_label"].cat.codes
   #df.user_sentiment_label = pd.Categorical(df.user_sentiment_label).codes
#y=pd.get_dummies(df2["user_sentiment_label"])
#df2.rename(columns={'positive': 'user_sentiment_label'}, inplace=True)

   
   label=df.loc[:,"5"].tolist()
   x=df.loc[:,"hil_dis_cor1"].tolist()
   y=df.loc[:,"hil_dis_cor2"].tolist()
   

   print(label)
   
   fig = plt.figure(figsize=(8,8))
   color= ['green' if l == 7 else 'red' if l==6 else 'blue' if l==5 else 'purple' if l==4 else 'white' if l==3 else 'black'if l==2 else 'yellow' if l==1 else 'silver'  for l in label]
   plt.scatter(x,y, color=color)
  
  # plt.scatter(x,y, c=label, cmap=matplotlib.colors.ListedColormap(colors))

   #cb = plt.colorbar()
   #loc = np.arange(0,max(label),max(label)/float(len(colors)))
   #cb.set_ticks(loc)
   #cb.set_ticklabels(colors)
   plt.show()