from assignment1.question1 import preprocess, get_documents_from_index, process_documents, \
    reconstruct_index_from_file, write_index_to_file
from assignment1.question1 import SPACE, DOCUMENT_PATH

BI_WORD_INDEX_FILE = "bi_word_index.txt"


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


def index_bi_words() -> dict:
    """Indexes the bi-words with the original documents"""
    processed_documents = process_documents(DOCUMENT_PATH)
    bi_word_index = create_bi_word_index(processed_documents)

    write_index_to_file(bi_word_index, BI_WORD_INDEX_FILE)

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


def main():
    """Main function"""
    bi_word_index = index_bi_words()

    # if os.path.exists(BI_WORD_INDEX_FILE):
    #     bi_word_index = reconstruct_index_from_file(BI_WORD_INDEX_FILE)

    query = input("Enter the bi-word query: ")
    result = search_bi_word_index(query, bi_word_index)

    if not result:
        print("No results found!")
    else:
        get_documents_from_index(result, DOCUMENT_PATH)


if __name__ == "__main__":
    while True:
        main()
