import os
import math
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Constants
SPACE = ' '
EMPTY = ''
READ = 'r'
UTF_8 = 'utf-8'
CORPUS = 'corpus/'
INDEX_FILE = 'index.txt'


# Preprocessing Functions
def case_fold(string: str) -> str:
    """Converts the string to lowercase."""
    return string.lower()


def remove_stopwords(string: str) -> str:
    """Removes stopwords using NLTK's stopwords."""
    stop_words = set(stopwords.words('english'))
    return SPACE.join([word for word in word_tokenize(string) if word not in stop_words])


def remove_punctuation(string: str) -> str:
    """Removes punctuation from the string."""
    return EMPTY.join([char for char in string if char.isalnum() or char.isspace()])


def expand_contractions(string: str) -> str:
    """Expands contractions in the string."""
    return string.replace("'", SPACE)


def stem_tokens(tokens: list) -> list:
    """Applies Porter Stemmer to the tokens."""
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in tokens]


def lemmatize_tokens(tokens: list) -> list:
    """Lemmatizes the tokens using WordNet Lemmatizer."""
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in tokens]


def tokenize(string: str) -> list:
    """Tokenizes the string."""
    return word_tokenize(string)


def preprocess(string: str) -> list:
    """
    Preprocesses the string by performing:
    1. Case Folding
    2. Expanding Contractions
    3. Removing Punctuation
    4. Removing Stopwords
    5. Tokenization
    6. Lemmatization
    7. Stemming
    """
    string = case_fold(string)
    string = expand_contractions(string)
    string = remove_punctuation(string)
    # string = remove_stopwords(string)
    tokens = tokenize(string)
    # tokens = lemmatize_tokens(tokens)
    # tokens = stem_tokens(tokens)
    return tokens


# Document Processing Functions
def read_document_as_tokens(file: str) -> list:
    """Reads a document and returns the tokens after preprocessing."""
    with open(file, READ, encoding=UTF_8) as f:
        return preprocess(f.read())


def read_documents(directory: str) -> tuple:
    """Reads all documents from the directory, returns their contents and IDs."""
    documents = {}
    doc_ids = {}

    for i, file in enumerate(os.listdir(directory)):
        if file.endswith('.txt'):
            doc_ids[i] = file
            file_path = os.path.join(directory, file)
            documents[i] = read_document_as_tokens(file_path)
    return documents, doc_ids


# Inverted Index and Document Length Calculation
def create_index_with_tf_df_and_lengths() -> tuple:
    """Creates an inverted index with term frequencies and document lengths."""
    documents, doc_ids = read_documents(CORPUS)
    inverted_index = {}
    doc_lengths = {}

    for doc_id, document in documents.items():
        term_freqs = {}
        # Calculate term frequencies
        for term in document:
            term_freqs[term] = term_freqs.get(term, 0) + 1

        # Populate the inverted index with tf information
        for term, freq in term_freqs.items():
            if term not in inverted_index:
                inverted_index[term] = {}
            inverted_index[term][doc_id] = calculate_tf(freq)

        # Calculate document length for normalization
        doc_lengths[doc_id] = calculate_document_length(term_freqs)

    return inverted_index, doc_ids, doc_lengths


def calculate_tf(term_freq) -> float:
    """Calculates term frequency using logarithmic weighting."""
    return 1 + math.log10(term_freq) if term_freq > 0 else 0


def calculate_idf(total_docs, doc_freq) -> float:
    """Calculates inverse document frequency."""
    return math.log10(total_docs / doc_freq) if doc_freq > 0 else 0


def calculate_document_length(doc_vector) -> float:
    """Calculates the length of a document vector for normalization."""
    return math.sqrt(sum(weight ** 2 for weight in doc_vector.values()))


def calculate_cosine_similarity(query_vector, doc_vector, doc_length) -> float:
    """Calculates cosine similarity between query and document vectors."""
    dot_product = sum(query_vector[term] * doc_vector.get(term, 0) for term in query_vector)
    return dot_product / doc_length if doc_length > 0 else 0


# Query Processing and Ranked Retrieval
def process_query(query, index, doc_lengths, total_docs) -> list:
    """Processes the query and computes ranked retrieval using cosine similarity."""
    query_tokens = preprocess(query)

    # Create query vector with tf-idf weighting
    query_vector = {}
    for term in query_tokens:
        if term in index:
            df = len(index[term])
            tf = query_tokens.count(term)
            query_vector[term] = calculate_tf(tf) * calculate_idf(total_docs, df)

    # Normalize the query vector
    query_length = calculate_document_length(query_vector)
    if query_length > 0:
        query_vector = {term: weight / query_length for term, weight in query_vector.items()}

    # Compute cosine similarity scores for all documents
    scores = {}
    for term in query_vector:
        if term in index:
            for doc_id, tf_weight in index[term].items():
                scores[doc_id] = scores.get(doc_id, 0) + query_vector[term] * tf_weight

    # Normalize by document lengths and rank
    ranked_docs = [(doc_id, score / doc_lengths[doc_id]) for doc_id, score in scores.items() if doc_lengths[doc_id] > 0]
    ranked_docs.sort(key=lambda x: (-x[1], x[0]))

    return ranked_docs[:10]  # Return top 10 relevant documents


# Main Search Function
def search(query) -> None:
    """Main search function to process the query and display ranked results."""
    index, doc_ids, doc_lengths = create_index_with_tf_df_and_lengths()
    total_docs = len(doc_ids)

    ranked_results = process_query(query, index, doc_lengths, total_docs)

    print("Top 10 relevant documents for your query:")
    for doc_id, score in ranked_results:
        print(f"Document ID: {doc_ids[doc_id]}, Score: {score}")


def main() -> None:
    """Entry point for the search functionality."""
    query = input("Enter your query: ")
    search(query)


# Run the main function
if __name__ == "__main__":
    main()