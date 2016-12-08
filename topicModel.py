import pandas as pd
import numpy as np
import string
import matplotlib.pyplot as plt

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
from scipy.sparse import csr_matrix, vstack


def plotPerplexities(m1, m2):
    xaxis = np.arange(len(m1.keys()))
    y_base_m = []
    y_base_d = []
    for subreddit, res in m1.iteritems():
        y_base_m.append(res[0])
        y_base_d.append(res[1])

    y_prop_m = []
    y_prop_d = []
    for subreddit, res in m2.iteritems():
        y_prop_m.append(res[0])
        y_prop_d.append(res[1])

    plt.figure(1)
    plt.subplot(2,1,1)
    plt.errorbar(xaxis, y_base_m, y_base_d)
    plt.title("Average Perplexity Score for Baseline", fontsize=35)
    plt.xlabel("Subreddits", fontsize=20)
    plt.ylabel("Perplexity", fontsize=20)
    plt.yticks(fontsize=20)
    plt.xticks(xaxis, m1.keys(), fontsize=20)

    plt.subplot(2,1,2)
    plt.errorbar(xaxis,y_prop_m, y_prop_d)
    plt.title("Average Perplexity Score for Tuned Model", fontsize=35)
    plt.xlabel("Subreddits", fontsize=20)
    plt.ylabel("Perplexity", fontsize=20)
    plt.yticks(fontsize=20)
    plt.xticks(xaxis, m2.keys(), fontsize=20)
    plt.show()
    
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

def crossValidate(folds, terms, vocabulary, subredditName, avgPerplexities):
    print "Performing",
    print folds,
    print "fold cross validation."
    perplexities = []
    terms = terms.todense()
    n = terms.shape[0]/folds
    print "n =",
    print n
    termsList = [terms[i:i + n] for i in range(0, terms.shape[0], n)]
    terms = csr_matrix(terms)
    for i, Hold in enumerate(termsList):
        if i == 10:
            continue
        Hold = csr_matrix(Hold)
        print "Fold: ",
        print i
        Train = csr_matrix((0, terms.shape[1]))
        for j, train in enumerate(termsList):
            if i != j:
                Train = vstack([Train, csr_matrix(train)])
        if Train.shape[0] == 0 or Hold.shape[0] == 0:
            return 1
        model = fit_LDA(Train, vocabulary, Train.shape[0], Train.shape[1], 1, subredditName)
        #models[subredditName] = model
        perplexities.append(model.perplexity(Hold))
        print model.perplexity(Hold)
        print
    avgPerplexities[subredditName] = (np.mean(perplexities), np.std(perplexities))

def preProcess(data):
    stop = stopwords.words('english')
    stop.extend(['www', 'http', 'https', 'com', 'net', 'org', 'edu', '://', 'jpg', 'png', 'gif', 'href', 'deleted', 'just', 'like', 'im', 'dont', 'wa', 'u', 'ha', 'get', 'would', 'thats', 'thing', 'even', 'one', 'well', 'see', 'got', 'could', 'should', 'also', 'go', 'make', 'sure', 'ive', 'think', 'time', 'good', 'uve', 'much'])

    tf_vectorizer = TfidfVectorizer(tokenizer=tokenize,
                                    max_df=.9, min_df=1,
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
    #print term_matrix
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
    #files = ['24hoursupport.json']
    files = ['2010.json', '2011.json', '2012.json']
    datas = []
    subreddits = []
    print "Loading Datasets"
    for file in files:
        # Format json data to be read by pandas
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
        if (shuffled.empty == False):
            datas.append(shuffled)

    res = list(set(subreddits[0]).intersection(*subreddits))
    subreddits = res
    baselineAvgModelPerplexities = {}
    proposedAvgModelPerplexities = {}
    yearModels = []
    for data in datas:
        print("Extracting proposed tf features for LDA...")
        models = {}
        data = data.query('@subreddits in subreddit')
        for subredditName in subreddits:
            subDF = data.loc[data.subreddit == subredditName]
            if(subDF.empty != True):
                samples = [x for x in subDF['body']]
                (termMatrix, vocabulary) = preProcess(samples)
                crossValidate(10, termMatrix, vocabulary, subredditName, proposedAvgModelPerplexities)
        print "-----------------------------------------------"

    for data in datas:
        print("Extracting baseline tf features for LDA...")
        models = {}
        data = data.query('@subreddits in subreddit')
        for subredditName in subreddits:
            subDF = data.loc[data.subreddit == subredditName]
            if(subDF.empty != True):
                samples = [x for x in subDF['body']]
                (termMatrix, vocabulary) = preProcessBaseline(samples)
                crossValidate(10, termMatrix, vocabulary, subredditName, baselineAvgModelPerplexities)
        print "-----------------------------------------------"
    plotPerplexities(baselineAvgModelPerplexities, proposedAvgModelPerplexities)

results = {}
explore_data('2016-08-15000.json')
#print(results)


# Remove Stop words
# LDA on subreddits for each year
# Evaluate the quality of the topics detected for each year
