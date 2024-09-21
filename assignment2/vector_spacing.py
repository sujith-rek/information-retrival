# n this assignment, you will be implementing ranked retrieval using vector space model. To
# implement the VSM, you may choose to implement your dictionary and postings lists in the
# following format. The only difference between this format and that in the textbook, is that
# you encode term frequencies in the postings for the purpose of computing tf×idf. The tuple in
# each posting represents (doc ID, term freq).
# Term doc freq (df) → postings lists
# Ambitious 5 → (1, 5)→ (7,2) → (21, 7) → ...
# ... ... ...
# In addition to the standard dictionary and postings file, you will need to store information at
# indexing time about the document length, in order to do document normalization. In the
# textbook this is referred to as Length[N]. You may store this information with the postings,
# dictionary or as a separate file.
# In the searching step, you will need to rank documents by cosine similarity based on tf×idf. In
# terms of SMART notation of ddd.qqq, you will need to implement the lnc.ltc ranking scheme
# (i.e., log tf and idf with cosine normalization for queries documents, and log tf, cosine
# normalization but no idf for documents. Compute cosine similarity between the query and each
# document, with the weights follow the tf×idf calculation, where term freq = 1 + log(tf) and
# inverse document frequency idf = log(N/df) (for queries). That is,
# tf-idf = (1 + log(tf)) * log(N/df).
# It's suggested that you use log base 10 for your logarithm calculations. The queries we provide
# are now free text queries, i.e., you don't need to use query operators like AND, OR, NOT and
# parentheses, and there will be no phrasal queries. These free text queries are similar to those
# you type in a web search engine's search bar.
# Your searcher should output a list of up to 10 most relevant (less if there are fewer than ten
# documents that have matching stems to the query) docIDs in response to the query. These
# documents need to be ordered by relevance, with the first document being most relevant. For
# those with marked with the same relevance, further sort them by the increasing order of the
# docIDs.

import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

SPACE = ' '
EMPTY = ''
READ = 'r'
UTF_8 = 'utf-8'
CORPUS = 'corpus/'


def case_fold(string: str) -> str:
    """Converts the string to lowercase"""
    return string.lower()


def remove_stopwords(string: str) -> str:
    """Removes the stopwords from the string using NLTK's stopwords"""
    stop_words = set(stopwords.words('english'))
    return SPACE.join([word for word in string.split() if word not in stop_words])


def remove_punctuation(string: str) -> str:
    """Removes the punctuation from the string"""
    return EMPTY.join([char for char in string if char.isalnum() or char.isspace()])


def expand_contractions(string: str) -> str:
    """Expands the contractions in the string"""
    return string.replace("'", SPACE)


def stem_string(tokens: list) -> list:
    """Stems the tokens using Porter Stemmer"""
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in tokens]


def lemmatize(tokens: list) -> list:
    """Lemmatizes the tokens using WordNet Lemmatizer"""
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in tokens]


def tokenize(string: str) -> list:
    """Tokenizes the string"""
    return word_tokenize(string)


def preprocess(string: str) -> list:
    """Preprocesses the string using the following steps:
    1. Case Folding
    2. Expanding Contractions
    3. Removing Punctuation
    4. Removing Stopwords
    5. Tokenization
    6. Stemming
    7. Lemmatization
    """
    string = case_fold(string)
    string = expand_contractions(string)
    string = remove_punctuation(string)
    string = remove_stopwords(string)
    tokens = tokenize(string)
    tokens = stem_string(tokens)
    tokens = lemmatize(tokens)
    return tokens


# Document Processing Functions
def read_document_as_tokens(file: str) -> list:
    """Reads the document from the file and returns it as a string"""
    with open(file, READ, encoding=UTF_8) as f:
        return preprocess(f.read())


def read_document_in_directory(directory: str) -> dict:
    """Reads the documents in the directory and returns them as a dictionary"""
    documents = {}
    for file in os.listdir(directory):
        if file.endswith('.txt'):
            file_path = os.path.join(directory, file)
            documents[file] = read_document_as_tokens(file_path)
    return documents
