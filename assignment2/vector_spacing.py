import os
import math
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Constants
SPACE = ' '
EMPTY = ''
READ = 'r'
UTF_8 = 'utf-8'
CORPUS = 'corpus/'

# Preprocessing Functions
def case_fold(string: str) -> str:
    return string.lower()

def remove_stopwords(string: str) -> str:
    stop_words = set(stopwords.words('english'))
    return SPACE.join([word for word in word_tokenize(string) if word not in stop_words])

def remove_punctuation(string: str) -> str:
    return EMPTY.join([char for char in string if char.isalnum() or char.isspace()])

def expand_contractions(string: str) -> str:
    return string.replace("'", SPACE)

def tokenize(string: str) -> list:
    return word_tokenize(string)

def preprocess(string: str) -> list:
    string = case_fold(string)
    string = expand_contractions(string)
    string = remove_stopwords(string)
    string = remove_punctuation(string)
    return tokenize(string)

# Document Processing Functions
def read_document_as_tokens(file: str) -> list:
    with open(file, READ, encoding=UTF_8) as f:
        return preprocess(f.read())

def read_documents(directory: str) -> tuple:
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
    documents, doc_ids = read_documents(CORPUS)
    inverted_index = {}
    doc_vectors = {}
    doc_lengths = {}

    for doc_id, document in documents.items():
        term_freqs = {}
        for term in document:
            term_freqs[term] = term_freqs.get(term, 0) + 1

        doc_vector = {}
        for term, freq in term_freqs.items():
            tf_weight = calculate_tf(freq)
            doc_vector[term] = tf_weight

            if term not in inverted_index:
                inverted_index[term] = {}
            inverted_index[term][doc_id] = tf_weight

        # Store document vector for later use in calculating similarity
        doc_lengths[doc_id] = calculate_document_length(doc_vector)

        for term in doc_vector:
            doc_vector[term] /= doc_lengths[doc_id]

        doc_vectors[doc_id] = doc_vector

    return inverted_index, doc_ids, doc_lengths, doc_vectors

def calculate_tf(term_freq) -> float:
    return 1 + math.log10(term_freq) if term_freq > 0 else 0

def calculate_idf(total_docs, doc_freq) -> float:
    return math.log10(total_docs / doc_freq) if doc_freq > 0 else 0

def calculate_document_length(doc_vector) -> float:
    return math.sqrt(sum(weight ** 2 for weight in doc_vector.values()))


def calculate_cosine_similarity(query_vector, doc_vector) -> float:
    dot_product = sum(query_vector[term] * doc_vector.get(term, 0) for term in query_vector)
    return dot_product

# Query Processing and Ranked Retrieval
def process_query(query, index, doc_lengths, doc_vectors, total_docs) -> list:
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
    for doc_id, doc_vector in doc_vectors.items():
        similarity = calculate_cosine_similarity(query_vector, doc_vector)
        scores[doc_id] = similarity

    ranked_docs = [(doc_id, score) for doc_id, score in scores.items() if score > 0]
    ranked_docs.sort(key=lambda x: (-x[1], x[0]))

    return ranked_docs[:10]

# Main Search Function
def search(query) -> None:
    index, doc_ids, doc_lengths, doc_vectors = create_index_with_tf_df_and_lengths()
    total_docs = len(doc_ids)

    ranked_results = process_query(query, index, doc_lengths, doc_vectors, total_docs)

    print("Top 10 relevant documents for your query:")
    for doc_id, score in ranked_results:
        print(f"Document ID: {doc_ids[doc_id]}, Score: {score}")

def main() -> None:
    query = input("Enter your query: ")
    search(query)

# Run the main function
if __name__ == "__main__":
    main()
