import os

from assignment1.question1 import read_documents_as_strings, preprocess
from assignment1.question1 import UTF_8, READ, WRITE, SPACE, DOCUMENT_PATH

ORIGINAL_BI_WORD_INDEX_FILE = "original_bi_word_index.txt"


def read_dir(dir_path: str) -> list:
    """Reads the directory and returns the files in it"""
    return read_documents_as_strings([dir_path + f for f in os.listdir(dir_path)])


def read_original_documents() -> list:
    """Reads the original documents"""
    return read_dir(DOCUMENT_PATH)


def process_original_documents() -> list:
    """Processes the original documents"""
    documents = read_original_documents()
    processed_documents = [preprocess(document) for document in documents]
    return processed_documents


def create_bi_word_index(documents: list) -> dict:
    """Creates the bi-word index from the documents"""
    bi_word_index = {}
    for i, document in enumerate(documents):
        for j in range(len(document) - 1):
            bi_word = document[j] + " " + document[j + 1]
            if bi_word not in bi_word_index:
                bi_word_index[bi_word] = {i}
            else:
                bi_word_index[bi_word].add(i)
    return dict(sorted(bi_word_index.items()))


def index_bi_words_with_original() -> dict:
    """Indexes the bi-words with the original documents"""
    processed_documents = process_original_documents()
    bi_word_index = create_bi_word_index(processed_documents)

    bi_word_index_file = open(ORIGINAL_BI_WORD_INDEX_FILE, WRITE, encoding=UTF_8)
    bi_word_index_file.write(str(bi_word_index))
    bi_word_index_file.close()

    return bi_word_index


def reconstruct_bi_word_index_from_file(file: str) -> dict:
    """Reconstructs the bi-word index from the file"""
    bi_word_index_file = open(file, READ, encoding=UTF_8)
    bi_word_index = eval(bi_word_index_file.read())
    bi_word_index_file.close()
    return bi_word_index


def preprocess_bi_word_query(query: str) -> str:
    """Preprocesses the bi-word query"""
    return SPACE.join(preprocess(query))


def search_bi_word_index(query: str, bi_word_index: dict) -> set:
    """Searches the bi-word index for the query"""
    query = preprocess_bi_word_query(query)
    if query in bi_word_index:
        return bi_word_index[query]
    return set()


def get_documents_from_index(indices: set, directory: str):
    """Prints the document names from the indices"""
    files = os.listdir(directory)
    for i in indices:
        print(files[i])


def main():
    """Main function"""
    original_bi_word_index = index_bi_words_with_original()

    # if os.path.exists(ORIGINAL_BI_WORD_INDEX_FILE):
    #     original_bi_word_index = reconstruct_bi_word_index_from_file(ORIGINAL_BI_WORD_INDEX_FILE)

    query = input("Enter the bi-word query: ")
    original_result = search_bi_word_index(query, original_bi_word_index)

    print("Result: ")
    get_documents_from_index(original_result, DOCUMENT_PATH)


if __name__ == "__main__":
    while True:
        main()
