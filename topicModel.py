import pandas as pd
import numpy as np
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.corpus import stopwords


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
    stop = stopwords.words('english')
    stop.extend(['www', 'http', 'https', 'com', 'net', 'org', 'edu', '://', 'jpg', 'png', 'gif', 'href'])
    # Use tf (raw term count) features for LDA.
    print("Extracting tf features for LDA...")
    tf_vectorizer = CountVectorizer(max_df=5, min_df=1,
                                    max_features=None,
                                    stop_words=set(stop))

    # Pass in a list of strings here
    tf = tf_vectorizer.fit_transform(data_samples)
    mask = np.random.rand(len(tf_vectorizer.vocabulary_)) < 0.8
    print mask
    print "Orig:"
    print tf
    tfTrain = tf[0,mask]
    tfHold = tf[0,~mask]
    print
    print tfTrain
    print
    print tfHold
    print a
    print("Training LDA model with tf features, "
          "n_samples=%d, subreddit_name=%s, n_features=%d..."
          % (n_samples, name, n_features))
    lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=10,
                                    learning_method='online',
                                    learning_offset=50.,
                                    random_state=0)
    lda.fit(tfTrain)
    score = lda.score(tfHold)
    perp = lda.perplexity(tfHold)
    print "Log Likihood: " + str(perp)
    print "Score: " + str(score)
    print("Topics in LDA model:")
    tf_feature_names = tf_vectorizer.get_feature_names()
    print_top_words(name, lda, tf_feature_names, n_top_words)

def explore_data(file):
    files = ['2010.json', '2011.json']
    datas = []
    subreddits = []
    for file in files:
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
            
        subreddits.append(data['subreddit'].unique().tolist())

        # Shuffle the data
        shuffled = data.iloc[np.random.permutation(len(data))]
        shuffled.reset_index(drop=True)
        # Make training and holdout
        if (shuffled.empty == False):
            datas.append(shuffled)
    
    res = list(set(subreddits[0]).intersection(*subreddits))
    subreddits = res
    #print subreddits
    #print a
    for data in datas:
        data = data.query('@subreddits in subreddit')
        for idx, subredditName in enumerate(subreddits):
            subDF = data.loc[data.subreddit == subreddits[idx]]
            #print subDF
            if (subDF.empty == False):
                #print subDF
                subDF = subDF.sample(frac=1).reset_index(drop=True)
                samples = [x for x in subDF['body']]
                if len(samples) >= 1:
                    fit_LDA(samples, len(samples), 1, subredditName,)
        print "------------------------------------------------------"
        print
        print "------------------------------------------------------"

results = {}
explore_data('2016-08-15000.json')
print(results)


# Remove Stop words
# LDA on subreddits for each year
# Evaluate the quality of the topics detected for each year
