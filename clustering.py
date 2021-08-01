# Clustering python File
''' Comment on the inputs and outputs of the functions later
    as well as the lines of codes where necessary.'''

## Importing modules
import nltk
import numpy as np
import re
import pandas as pd
import seaborn as sns
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn import metrics
from sklearn.manifold import TSNE
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.cluster import KMeans, MiniBatchKMeans
import umap
import umap.plot
import hdbscan
import sklearn.cluster as cluster
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.neighbors import KNeighborsClassifier


## A function to handle pronouns
def rem_prnoun(sent):
    tagged_sentence = nltk.tag.pos_tag(sent.split())
    edited_sentence = [word for word, tag in tagged_sentence if tag != 'NNP' and tag != 'NNPS']
    return ' '.join(edited_sentence)


## A function only tokenize text
def tokenize_only(text):
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens


## A function to tokenize and lemmatize
def tokenize_and_lemmatize(text, lemmatizer):
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    lemmas = [lemmatizer.lemmatize(t) for t in filtered_tokens]
    return lemmas


# A function to obtain the tfidf matrix of a text
def tfidf(frame):
    frame['synopsis'] = frame['synopsis'].apply(rem_prnoun)
    words = []
    for i in frame['synopsis']:
        words.append(i.split(' '))
    ranks = []
    for i in range(0, len(frame['synopsis'])):
        ranks.append(i)
    MAX_NUM_WORDS = np.unique(np.asarray(words)).shape[0]
    nltk.download('stopwords', quiet = True)
    nltk.download('punkt', quiet = True)
    nltk.download('averaged_perceptron_tagger', quiet = True)
    stopwords = nltk.corpus.stopwords.words('english') + ["'d", "'ll", "'re", "'s", "'ve", 'could', 'doe', 'ha', 'might', 'must', "n't", 'need', 'sha', 'wa', 'wo', 'would']
    stemmer = SnowballStemmer('english')
    lemmatizer = WordNetLemmatizer()
    totalvocab_stemmed = []
    totalvocab_tokenized = []
    for i in frame['synopsis']:
        allwords_stemmed = tokenize_and_lemmatize(i, lemmatizer)
        totalvocab_stemmed.extend(allwords_stemmed)
        allwords_tokenized = tokenize_only(i)
        totalvocab_tokenized.extend(allwords_tokenized)
    vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)
    tfidf_vectorizer = TfidfVectorizer(max_df = 0.6, max_features = 200000, min_df = 20,
                                        stop_words = stopwords, use_idf = True, sublinear_tf = True,
                                        tokenizer = (lambda x: tokenize_and_lemmatize(x, lemmatizer)),
                                        ngram_range = (1, 2))
    tfidf_matrix = tfidf_vectorizer.fit_transform(frame['synopsis'])
    return tfidf_matrix


# A function to obtain the tfidf matrix of empath text
def tfidf_empath(frame):
    frame['empath_themes'] = frame['empath_themes'].apply(rem_prnoun)
    words = []
    for i in frame['empath_themes']:
        words.append(i.split(' '))
    ranks = []
    for i in range(0, len(frame['empath_themes'])):
        ranks.append(i)
    MAX_NUM_WORDS = np.unique(np.asarray(words)).shape[0]
    nltk.download('stopwords', quiet = True)
    nltk.download('punkt', quiet = True)
    nltk.download('averaged_perceptron_tagger', quiet = True)
    stopwords = nltk.corpus.stopwords.words('english') + ["'d", "'ll", "'re", "'s", "'ve", 'could', 'doe', 'ha', 'might', 'must', "n't", 'need', 'sha', 'wa', 'wo', 'would']
    stemmer = SnowballStemmer('english')
    lemmatizer = WordNetLemmatizer()
    totalvocab_stemmed = []
    totalvocab_tokenized = []
    for i in frame['empath_themes']:
        allwords_stemmed = tokenize_and_lemmatize(i, lemmatizer)
        totalvocab_stemmed.extend(allwords_stemmed)
        allwords_tokenized = tokenize_only(i)
        totalvocab_tokenized.extend(allwords_tokenized)
    vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)
    tfidf_vectorizer = TfidfVectorizer(max_df = 0.6, max_features = None, min_df = 20,
                                        stop_words = stopwords, use_idf = True, sublinear_tf = True,
                                        tokenizer = (lambda x: tokenize_and_lemmatize(x, lemmatizer)),
                                        ngram_range = (1, 2), strip_accents = 'unicode', analyzer = 'word')
    tfidf_matrix = tfidf_vectorizer.fit_transform(frame['empath_themes'])
    return tfidf_matrix

## A function to reduce dimension
def dim_reduc(tfidf_matrix, dim = 50):
    svd = TruncatedSVD(dim)
    normalizer = Normalizer(copy = False)
    lsa = make_pipeline(svd, normalizer)
    X = lsa.fit_transform(tfidf_matrix)
    tfidf_embedding = umap.UMAP(metric = 'euclidean').fit(X)
    standard_embedding = umap.UMAP(metric = 'euclidean').fit_transform(X)
    clusterable_embedding = umap.UMAP(
    n_neighbors = 5,
    min_dist = 0.0,
    n_components = 10,
    random_state = 42
    ).fit_transform(X)
    return standard_embedding, clusterable_embedding


## A function to cluster content
def cluster(clusterable_embedding, standard_embedding):
    clusterer = hdbscan.HDBSCAN(
    min_samples = 5,
    min_cluster_size = 10,
    cluster_selection_epsilon = 0.1
    ).fit(clusterable_embedding)
    labels = clusterer.labels_
    clustered = (labels >= 0)
    neigh = KNeighborsClassifier(n_neighbors = 5)
    X_train = standard_embedding[clustered]
    y_train = labels[clustered]
    X_pred = standard_embedding[~clustered]
    neigh.fit(X_train, y_train)
    labels[~clustered] = neigh.predict(X_pred)
    return labels


## A function to get a frame that is plottable
def get_plottable_df(content, size, synopsis, x_coord, y_coord, labels):
    num_labels = len(set(labels))
    colors = sns.color_palette('hls', num_labels).as_hex()
    color_lookup = {v:k for k, v in zip(colors, set(labels))}
    df = pd.DataFrame({
    'uid': content,
    'text': content,
    'label': labels,
    'x_val': x_coord,
    'y_val': y_coord,
    'size': size/10
    })
    df['color'] = list(map(lambda x: color_lookup[x], labels))
    return df
