import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from __future__ import print_function
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def fit_LDA(data_samples, n_samples, n_features):
    # Use tf (raw term count) features for LDA.
    print("Extracting tf features for LDA...")
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                    max_features=n_features,
                                    stop_words='english')
    
    tf = tf_vectorizer.fit_transform(data_samples)
    print("Fitting LDA models with tf features, "
          "n_samples=%d and n_features=%d..."
          % (n_samples, n_features))
    lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=5,
                                    learning_method='online',
                                    learning_offset=50.,
                                    random_state=0)
    lda.fit(tf)

    print("\nTopics in LDA model:")
    tf_feature_names = tf_vectorizer.get_feature_names()
    print_top_words(lda, tf_feature_names, n_top_words)

def explore_data(file):
    data = None
    # Format json data to be read by pandas
    with open(file, 'rb') as f:
        data = f.readlines()

    data = map(lambda x : x.rstrip(), data)

    data_json_str = "[" + ','.join(data) + "]"

    # read in pandas
    data = pd.read_json(data_json_str)

    # shuffle data
    data = data.sample(frac = 1).reset_index(drop=True)
    
    # remove stop words from dataset
    #stop_words = stopwords.words('english')
    #data['body'] = data['body'].apply(lambda comment: [word for word in str(comment.encode('utf-8')) if word not in stop_words])
    #print data['body']
    #print data.describe()

   
    

explore_data('sample.json')