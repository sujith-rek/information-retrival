import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from contractions import get_contraction

# import nltk
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download('wordnet')

# Constants
INVERTED_INDEX_FILE = 'inverted_index.txt'
UTF_8 = 'utf-8'
READ = 'r'
WRITE = 'w'
DOCUMENT_PATH = 'corpus/'
AND = 'and'
OR = 'or'
NOT = 'not'
SPACE = ' '
EMPTY = ''
INPUT_MESSAGE = 'Enter the search query: '
QUERY_SUCCESS_MESSAGE = 'Documents satisfying the query: '
QUERY_FAILURE_MESSAGE = 'No documents satisfy the query'
APHO_IS = "'s"
IS = " is"
APHO_CAUSE = "'cause"
BECAUSE = " because"


# Document Preprocessing Functions
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
    contraction = get_contraction()
    if APHO_IS in string:
        string = string.replace(APHO_IS, IS)
    if APHO_CAUSE in string:
        string = string.replace(APHO_CAUSE, BECAUSE)

    return SPACE.join([contraction[word] if word in contraction else word for word in string.split()])


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
def read_document_as_string(file: str) -> str:
    """Reads the document from the file and returns it as a string"""
    with open(file, READ, encoding=UTF_8) as f:
        return f.read()


def read_documents_as_strings(files: list) -> list:
    """Reads the documents from the files and returns them as strings"""
    return [read_document_as_string(file) for file in files]


def create_inverted_index(documents: list) -> dict:
    """Creates the inverted index from the documents"""
    inverted_index = {}
    for i, document in enumerate(documents):
        for word in document:
            if word not in inverted_index:
                inverted_index[word] = {i}
            else:
                inverted_index[word].add(i)
    return dict(sorted(inverted_index.items()))


def process_data(files: list) -> list:
    """Processes the data from the files"""
    documents = read_documents_as_strings(files)
    processed_documents = [preprocess(document) for document in documents]
    return processed_documents


def reconstruct_inverted_index_from_index_file() -> dict:
    """Reconstructs the inverted index from the index file"""
    inv_index_file = open(INVERTED_INDEX_FILE, READ, encoding=UTF_8)
    inverted_index = eval(inv_index_file.read())
    inv_index_file.close()
    return inverted_index


def index_documents() -> tuple:
    """Indexes the documents"""
    files = [DOCUMENT_PATH + f for f in os.listdir(DOCUMENT_PATH)]

    if os.path.exists(INVERTED_INDEX_FILE):
        inverted_index = reconstruct_inverted_index_from_index_file()
    else:
        processed_documents = process_data(files)
        inverted_index = create_inverted_index(processed_documents)

        inv_index_file = open(INVERTED_INDEX_FILE, WRITE, encoding=UTF_8)
        inv_index_file.write(str(inverted_index))
        inv_index_file.close()

    return inverted_index, files


# Query Processing Functions
def preprocess_query(query: str) -> list:
    """Preprocesses the query"""
    query = query.split()
    for i in range(len(query)):
        if query[i].lower() not in [AND, OR, NOT]:
            query[i] = preprocess(query[i])[0]
    return query


def validate_query(query: list, inverted_index: dict) -> bool:
    """Validates the query by looking if word exists in the inverted index"""
    for i in range(len(query)):
        if query[i].lower() not in [AND, OR, NOT]:
            if query[i] not in inverted_index:
                return False
    return True


# Search Functions
def search(query: str, inverted_index: dict) -> set:
    """Searches the query in the inverted index"""
    query = preprocess_query(query)
    if not validate_query(query, inverted_index):
        return set()
    result = set()
    for i, word in enumerate(query):
        if word.lower() == AND:
            result = result.intersection(inverted_index[query[i + 1]])
        elif word.lower() == OR:
            result = result.union(inverted_index[query[i + 1]])
        elif word.lower() == NOT:
            result = result.difference(inverted_index[query[i + 1]])
        elif not result:
            result = inverted_index[word]
        else:
            pass

    return result


# Utility Functions
def get_document_name(index: int, files: list) -> str:
    """Returns the name of the document at the given index"""
    if index < 0 or index >= len(files):
        return EMPTY
    return os.path.basename(files[index])


def main():
    inverted_index, files = index_documents()
    query = input(INPUT_MESSAGE)
    result = search(query, inverted_index)
    if not result:
        print(QUERY_FAILURE_MESSAGE)
        return
    else:
        print(QUERY_SUCCESS_MESSAGE)
        for i in result:
            print(get_document_name(i, files))


if __name__ == '__main__':
    while True:
        main()
