import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()


def fit_LDA(data_samples, n_samples, n_features):
    n_top_words = 20
    n_topics = 5
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

    # Originally had open with rb. Why read as bytes? Does pandas prefer bytes?
    with open(file, 'r') as f:
        data = f.readlines()

    # Remove excess
    data = map(lambda x : x.rstrip(), data)

    data_json_str = "[" + ','.join(data) + "]"

    # read in pandas
    data = pd.read_json(data_json_str)

    # shuffle data
    data = data.sample(frac = 1).reset_index(drop=True)

    # Lead the stop words in to a local variable
    stop_words = stopwords.words('english')

    # Removes the stop words, but treats them as individual characters, instead of a full string.
    # Will continue looking at in the morning.
    for word in stop_words: data['body'] = data['body'].str.replace(word, "")
    print(data['body'])

    # This splits data['body'] in to rows of characters. Not sure that's what you were trying to do.
    # print(data['body'].apply(lambda comment: [word for word in str(comment.encode('utf-8')) if word not in stop_words]))

    # print(data['body'])
    # print(data.describe())


explore_data('sample.json')