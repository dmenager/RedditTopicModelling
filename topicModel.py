import pandas as pd
import numpy as np
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


def print_top_words(sub_name, model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        results[sub_name] = " ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(results[sub_name])
    print()


def fit_LDA(data_samples, n_samples, topics, name):
    n_features = 1000
    n_topics = topics
    n_top_words = 10
    # Use tf (raw term count) features for LDA.
    print("n_samples=%d" % (n_samples))
    print("Extracting tf features for LDA...")
    tf_vectorizer = CountVectorizer(max_df=5, min_df=1,
                                    max_features=None,
                                    stop_words='english')

    # tf_vectorizer = CountVectorizer(max_df=.3, min_df=1,
    #                                max_features=None,
    #                                stop_words='english')
    # Pass in a list of strings here
    tf = tf_vectorizer.fit_transform(data_samples)
    print("Fitting LDA models with tf features, "
          "n_samples=%d, subreddit_name=%s, n_features=%d..."
          % (n_samples, name, n_features))
    lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=5,
                                    learning_method='online',
                                    learning_offset=50.,
                                    random_state=0)
    lda.fit(tf)

    print("\nTopics in LDA model:")
    tf_feature_names = tf_vectorizer.get_feature_names()
    print_top_words(name, lda, tf_feature_names, n_top_words)


def explore_data(file):
    data = None
    # Format json data to be read by pandas

    # Originally had open with rb. Why read as bytes? Does pandas prefer bytes?
    with open(file, 'r') as f:
        data = f.readlines()

    # Remove excess
    data = map(lambda x: x.rstrip(), data)
    data_json_str = "[" + ','.join(data) + "]"

    # read in pandas
    data = pd.read_json(data_json_str)

    # set the index to be this and don't drop
    data.set_index(keys=['subreddit'], drop=False, inplace=True)
    # get a list of names
    subreddits = data['subreddit'].unique().tolist()

    # samples = [x for x in data['body']]
    # fit_LDA(samples, len(samples), len(subreddits))
    print(subreddits)
    for idx, subredditName in enumerate(subreddits):
        subDF = data.loc[data.subreddit == subreddits[idx]]
        # print subDF
        # shuffle data
        subDF = subDF.sample(frac=1).reset_index(drop=True)
        samples = [x for x in subDF['body']]
        if len(samples) > 1:
            fit_LDA(samples, len(samples), 1, subredditName)

results = {}
explore_data('2016-08-15000.json')
print(results)