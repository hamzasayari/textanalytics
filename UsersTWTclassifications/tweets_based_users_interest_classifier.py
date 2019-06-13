try:
    import twitter
except ImportError:
    print("python-twitter not installed. Will not be able to do real-time classification")
import pickle
#import cPickle
import credentials  # You'll have to fill in the credentials in the credentials file here
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords  # for using english stopwords
from gensim.models.phrases import Phrases
from gensim.utils import deaccent, decode_htmlentities, lemmatize
from sklearn.metrics import accuracy_score, confusion_matrix
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_20newsgroups
#twenty_train = fetch_20newsgroups(subset='train', shuffle=True) 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB      #return df
from sklearn.pipeline import Pipeline
import re
from nltk.corpus import stopwords  # for using english stopwords
from gensim.models.phrases import Phrases
from sklearn.metrics import accuracy_score, confusion_matrix
from matplotlib import pyplot as plt
from gensim.utils import deaccent, decode_htmlentities, lemmatize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import json_lines
import matplotlib.pyplot as plt
def preprocess_text(tweet):
    """
    Function to process an aggregated user profile. This does the following:
    1. Decode html entities. eg. "AT&amp;T" will become "AT&T"
    2. Deaccent
    3. Remove links.
    4. Remove any user mentions (@name).
    5. Lemmatize and remove stopwords.
    
    Parameters:
    ----------
    text : String. If train_texts is a list of tweets, ' '.join and pass
    
    Returns:
    -------
    text : preprocessed (tokenized) tweet.
    """
    tweet = decode_htmlentities(tweet)
    tweet = deaccent(tweet)
    tweet = tweet.encode('ascii', 'ignore')  # To prevent UnicodeDecodeErrors later on
    tweet = re.sub(r'http\S+', '', str(tweet))  # Step 3
    tweet = re.sub(r'@\w+', '', str(tweet) ) # Step 4
    tweet = tweet.split()
    tweet = lemmatize(' '.join(tweet), re.compile('(NN)'), stopwords=stopwords.words('english'), min_length=3, max_length=15)
    tweet = [word.split('/')[0] for word in tweet]
    return tweet
#Creating training and test sets
def get_dataframes(pycon_dict):
    """
    Function to get train and test dataframes (without any preprocessing).
    
    Parameters:
    ----------
    pycon_dict: The twitter user dictionary being used.
    
    Returns:
    -------
    train, test: Train and test dataframes.
    """
    columns = ['message', 'category']
    categories_map = {0: u'Business & CEOs',
                  1: u'Music',
                  2: u'Entertainment',
                  3: u'Fashion, Travel & Lifestyle',
                  4: u'Sports',
                  5: u'Tech',
                  6: u'Politics',
                  7: u'Science',
                  u'Business & CEOs': 0,
                  u'Entertainment': 2,
                  u'Fashion, Travel & Lifestyle': 3,
                  u'Music': 1,
                  u'Politics': 6,
                  u'Science': 7,
                  u'Sports': 4,
                  u'Tech': 5}

    train = pd.DataFrame(columns=columns)
    test = pd.DataFrame(columns=columns)
    
    for category in pycon_dict:
        for entity in pycon_dict[category]:
            train_texts = []
            test_texts = []
            num_texts = len(pycon_dict[category][entity])  # To get number of tweets
            train_indices = np.random.choice(num_texts, int(0.9 * num_texts), replace=False)  # Random selection
            test_indices = [i for i in range(num_texts) if i not in train_indices]  # Rest go into test set
            train_texts.extend(pycon_dict[category][entity][i].text for i in train_indices)  # Add to train texts
            test_texts.extend(pycon_dict[category][entity][i].text for i in test_indices)  # Add to test texts
            #### Create train dataframe ####
            train_texts = ' '.join(train_texts)
            df_train = pd.DataFrame([[train_texts, categories_map[category]]], columns=columns)
            train = train.append(df_train, ignore_index=True)
            #### Create test dataframe ####
            test_texts = ' '.join(test_texts)
            df_test = pd.DataFrame([[test_texts, categories_map[category]]], columns=columns)
            test = test.append(df_test, ignore_index=True)
            
    return train, test

from pprint import pprint

def get_processed_tweets():
    pycon_dict = pickle.load(open("data/pycon_dict.pkl", "rb"))
    train,test=get_dataframes(pycon_dict)
    processed_train_texts = train['message'].apply(preprocess_text)
    bigram = Phrases(processed_train_texts)  # For collocation detection
    processed_train_texts= [bigram[profile] for profile in processed_train_texts]
    processed_test_texts = test['message'].apply(preprocess_text)

    processed_test_texts= [bigram[message] for message in processed_test_texts]
    my_tags = pycon_dict.keys()
    #test_texts = [bigram[message] for message in test_texts]
    return processed_train_texts,processed_test_texts, my_tags

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(my_tags))
    target_names = my_tags
    plt.xticks(tick_marks, target_names, rotation=90)
    plt.yticks(tick_marks, target_names)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def evaluate_prediction(predictions, target, title="Confusion matrix"):
    print('accuracy %s' % accuracy_score(target, predictions))
    cm = confusion_matrix(target, predictions)
    print('confusion matrix\n %s' % cm)
    print('(row=expected, col=predicted)')
    
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plot_confusion_matrix(cm_normalized, title + ' Normalized') 

def most_influential_words(clf, vectorizer, category_index=0, num_words=10):
    features = vectorizer.get_feature_names()
    max_coef = sorted(enumerate(clf.coef_[category_index]), key=lambda x:x[1], reverse=True)
    return [features[x[0]] for x in max_coef[:num_words]]

def lr_classifier(processed_train_texts,processed_test_texts,texts_categories):
    categories = texts_categories
    categories=categories.astype('int')
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    train_tfidf_features = tfidf_vectorizer.fit_transform(' '.join(text) for text in processed_train_texts)
    clf_tfidf = LogisticRegression()
    clf_tfidf = clf_tfidf.fit(train_tfidf_features, categories)
    test_tfidf_features = tfidf_vectorizer.transform(' '.join(text) for text in processed_test_texts)
    predictions = clf_tfidf.predict(test_tfidf_features)
    f = open('lr_clf_tfidf.pickle', 'wb')
    pickle.dump(clf_tfidf, f)
    f.close()
    return predictions

def nb_classifier(processed_train_texts,processed_test_texts,texts_categories):
    categories = texts_categories
    categories=categories.astype('int')
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    train_tfidf_features = tfidf_vectorizer.fit_transform(' '.join(text) for text in processed_train_texts)
    clf_tfidf = MultinomialNB()
    clf_tfidf = clf_tfidf.fit(train_tfidf_features, categories)
    test_tfidf_features = tfidf_vectorizer.transform(' '.join(text) for text in processed_test_texts)
    predictions = clf_tfidf.predict_proba(test_tfidf_features)
    #f = open('nb_clf_tfidf.pickle', 'wb')
    #pickle.dump(clf_tfidf, f)
    #f.close()

    return predictions


def get_users_histograms():
    
    users_histograms =nb_classifier(processed_train_texts,processed_test_texts,texts_categories)
     
    return users_histograms


def plot_user_histogramm(user_histogramm):
    
    categories_names=["Tech","Business & CEOs","Entertainment","Science","Fashion, Travel & Lifestyle","Sports","Music","Politics"]
    user_dict= dict(zip(categories_names, user_histogramm))
    
    probabilities= list(user_dict.values())
    s = pd.Series(
    probabilities,
    index = [categories_names]
    )
    ax = plt.gca()
    ax.tick_params(axis='x', colors='blue')
    ax.tick_params(axis='y', colors='red')

#Plot the data:
    my_colors = ['red', 'green', 'blue', 'black','yellow','purple','white','silver']  #red, green, blue, black, etc.

    s.plot( 
    kind='bar', 
    color=my_colors,
    )
    
    plt.show()

def getTweets(category_dict, category): 
    """ Function to get the tweets for each handle in the dictionary in the particular category.
     Parameters: ---------- category_dict: User category dictionary consisting of categories and user handles. 
     category: String. Name of the category. 
     Returns: ------- category_dict: Dictionary with the most recent 200 tweets of all user handles. """ 
    for handle in category_dict[category]:
        category_dict[category][handle] = api.GetUserTimeline(screen_name=handle, count=200)
    return category_dict


if __name__ == '__main__':

    ######## usage example: load, process data to create and visualize users intrests histgramms based on their tweets #########
    
    
    
    pycon_dict = pickle.load(open("data/pycon_dict.pkl", "rb")) ####### load tweets of differents intersts categories
    

    
    train,test=get_dataframes(pycon_dict)  ###### build train and test dataframes
    
    print(train.head())
    
    texts_categories=train['category']
    
    processed_train_texts,processed_test_texts, my_tags=get_processed_tweets() ########### build processed train and test dataframes
    
    users_histogramms=get_users_histograms()   ######### create users histogramms based on naive bayes classifier predicted probabilities
    
    first_user_histgramm=users_histogramms[1]
    
    plot_user_histogramm(first_user_histgramm) ####### visualise the first user histogramm
    