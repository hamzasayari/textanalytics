import pandas as pd
import csv
import numpy as np
import codecs
import json_lines
import matplotlib.pyplot as plt



###############  Scipt to explore and visualize users data  ######################################################################




def get_users_informations(): 

    ####### retrieve sentiment labeled twitter users information from 'setiment_labeled_tweets.jsonl' file #######"#######"

    users_descriptions=[]
    users_names=[]
    users_ids=[]
    users_locations=[]
    num_of_followers=[]
    num_of_friends=[]
    users_tweets=[]
    tweets_ids=[]
    with open('users_data.jsonl', 'r') as f:
        for item in json_lines.reader(f):
            tweets_ids.append(item["id"])
            users_tweets.append(item["full_text"])
            users_descriptions.append(item["user"]["description"])
            users_names.append(item["user"]["name"])
            users_ids.append(item["user"]["id"])
            users_locations.append(item["user"]["location"]) 
            num_of_followers.append(item["user"]["followers_count"])
            num_of_friends.append(item["user"]["friends_count"])
    return users_descriptions,users_names,users_ids,users_locations,num_of_followers,num_of_friends,tweets_ids,users_tweets

def get_users_df(): 

    ####### match the sentiment labeled tweets in 'corpus.csv' file with users information from 'users_data.jsonl' file #####
 
    df1=pd.DataFrame.from_csv('corpus.csv')
    df2=df1.index.values
    df1.loc[:,'campany_name']=df2
    df1.rename(columns={'126415614616154112':'tweet_id','positive':'user_sentiment_label'}, inplace=True)

    users_descriptions,users_names,users_ids,users_locations,num_of_followers,num_of_friends,tweets_ids,users_tweets=get_users_informations()
    df2= pd.DataFrame(np.column_stack([users_descriptions,users_names,users_ids,users_locations,num_of_followers,num_of_friends,tweets_ids,users_tweets]), 
                               columns=['user_id', 'user_name', 'user_description','user_location','num_of_followers','num_of_friends','tweet_id','user_tweet'])
    
    df2['tweet_id']=df2['tweet_id'].apply(int)
    df3=pd.DataFrame.merge(df1, df2, on="tweet_id")
    return df3

def get_users_tweets():
   df=get_users_df()
   return df["full_text"]


def plot_users_locations_histogramm(n): 

    ########  function to visualize n random users_locations frequency in the dataset #################
    
    df=pd.DataFrame.from_csv('twitter_users_data.csv')
    df=df.sample(n=n)
    users_locations=df["user_location"].value_counts().index.values
    locations_counts=df["user_location"].value_counts().values
    
    
    s = pd.Series(
    locations_counts,
    index = [users_locations]
    )
    ax = plt.gca()
    ax.tick_params(axis='x', colors='blue')
    ax.tick_params(axis='y', colors='red')
    s.plot( 
    kind='bar'
    #color=my_colors,
    )
    plt.show()


def get_top_n_influential_users(top_n_influential_users): 

    ############# function returning the top_n_influential_users list ########################

    df=pd.DataFrame.from_csv('twitter_users_data.csv')
    df=df.sort_values(['num_of_followers'],ascending=False)
    df=df.iloc[:top_n_influential_users,:]
    return df


def plot_most_influential_users_histogramm(top_n_influential_users):

     ############# function to plot the top_n_influential_users histogramm ########################

    most_influential_users_df=get_top_n_influential_users(top_n_influential_users)
    most_influential_users_followers_counts=most_influential_users_df.loc[:,"num_of_followers"].values
    most_influential_users_names=most_influential_users_df.loc[:,"user_name"].values
    
    
    s = pd.Series(
     most_influential_users_followers_counts,
    index = [most_influential_users_names]
    )
    ax = plt.gca()
    ax.tick_params(axis='x', colors='blue')
    ax.tick_params(axis='y', colors='red')
    s.plot( 
    kind='bar'
    #color=my_colors,
    )
    plt.xlabel("User_name")
    plt.ylabel("User_followers_count")
 
    plt.xticks(range(0, 7))
    plt.yticks(range(1, 20))
    plt.title('Most_followed_users_histogramm')
    plt.show()

if __name__ == '__main__':
    
    ################ usage example: visualise the top 10 influential users histogramm ####################

    plot_most_influential_users_histogramm(10)
    