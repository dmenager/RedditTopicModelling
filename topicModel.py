import pandas as pd
import numpy as np
import string

from time import time
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.random_projection import SparseRandomProjection
from sklearn.random_projection import johnson_lindenstrauss_min_dim
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, scale
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import word_tokenize          
from nltk.stem.porter import PorterStemmer
from scipy.sparse import csr_matrix

stemmer = PorterStemmer()
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

lemmatizer = WordNetLemmatizer()
def lemmatize_tokens(tokens, lemmatizer):
    lemma = []
    for item in tokens:
        lemma.append(lemmatizer.lemmatize(item))
    return lemma
    
def tokenize(text):
    text = "".join([ch for ch in text.lower() if ch not in string.punctuation])
    tokens = word_tokenize(text)
    lemmas = lemmatize_tokens(tokens, lemmatizer)
    return lemmas
    #stems = stem_tokens(tokens, stemmer)
    #return stems

def print_top_words(sub_name, model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        results[sub_name] = " ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(results[sub_name])
    print()

def preProcess(data):
    stop = stopwords.words('english')
    stop.extend(['www', 'http', 'https', 'com', 'net', 'org', 'edu', '://', 'jpg', 'png', 'gif', 'href', 'deleted', 'just', 'like', 'im', 'dont', 'wa', 'u', 'ha', 'get', 'would', 'thats', 'thing', 'even', 'one', 'well', 'see', 'got', 'could', 'should', 'also', 'go', 'make', 'sure'])

    tf_vectorizer = TfidfVectorizer(tokenizer=tokenize,
                                    max_df=.9, min_df=1,
                                    #stop_words='english',
                                    stop_words=set(stop),
                                    binary=False,
                                    norm = None,
                                    use_idf=True,
                                    sublinear_tf=False,
                                    max_features=10)
    
    # Pass in a list of strings here
    term_matrix = tf_vectorizer.fit_transform(data).tocsr()
    term_matrix = scale(term_matrix, with_mean=False, with_std=False, axis=0, copy=False)
    #term_matrix = MaxAbsScaler((0,1)).fit_transform(term_matrix)
    print term_matrix
    return (term_matrix, tf_vectorizer.get_feature_names())

def preProcessBaseline(data):
    stop = stopwords.words('english')
    tf_vectorizer = CountVectorizer(max_df=1, min_df=1,
                                    max_features=None,
                                    stop_words='english')
    # Pass in a list of strings here
    return (tf_vectorizer.fit_transform(data).tocsr(), tf_vectorizer.get_feature_names())
    
def fit_LDA(tf, tf_feature_names, n_samples, n_features, topics, name):
    n_topics = topics
    n_top_words = 10
    print("Training LDA model with tf features, "
          "n_samples=%d, subreddit_name=%s, n_features=%d..."
          % (n_samples, name, n_features))
    lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=10,
                                    learning_method='online',
                                    learning_offset=50.,
                                    random_state=0)
    lda.fit(tf)
    print("Topics in LDA model:")    
    print_top_words(name, lda, tf_feature_names, n_top_words)
    return lda

def explore_data(file):
    #files = ['funny.json', '24hoursupport.json', 'todayilearned.json', 'politics', 'uspolitics.json']
    files = ['politics.json']
    datas = []
    preprocessed = []
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
    
    print("Extracting tf features for LDA...")
    yearModels = []
    for data in datas:
        models = {}
        data = data.query('@subreddits in subreddit')
        for subredditName in subreddits:
            subDF = data.loc[data.subreddit == subredditName]
            if(subDF.empty != True):
                samples = [x for x in subDF['body']]
                (termMatrix, vectorizer) = preProcess(samples)
                # minus 1 because we want to use this as index
                rowsIndex = termMatrix.shape[0] - 1
                split = int(round(.7 * rowsIndex))
                Train = termMatrix[:split]
                Hold = termMatrix[split +1:]
                if Train.shape[0] == 0 or Hold.shape[0] == 0:
                    continue
                model = fit_LDA(Train, vectorizer, termMatrix.shape[0], termMatrix.shape[1], 1, subredditName)
                models[subredditName] = model
                print model.perplexity(Hold)
                print
                preprocessed.append((termMatrix, vectorizer, subredditName))
        print "-----------------------------------------------"

results = {}
explore_data('2016-08-15000.json')
print(results)


# Remove Stop words
# LDA on subreddits for each year
# Evaluate the quality of the topics detected for each year
