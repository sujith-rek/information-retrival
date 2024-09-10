from assignment1.question1 import preprocess, process_documents, create_inverted_index, search, \
    get_documents_from_index, write_index_to_file, preprocess_query, AND, OR, SPACE
from assignment1.question1 import DOCUMENT_PATH

POSITIONAL_INDEX_FILE = "positional_index.txt"


def create_positional_index(documents: list) -> dict:
    """Creates the positional index from the documents"""
    positional_index = {}
    for i, document in enumerate(documents):
        for j, word in enumerate(document):
            if word not in positional_index:
                positional_index[word] = {i: [j]}
            else:
                if i not in positional_index[word]:
                    positional_index[word][i] = [j]
                else:
                    positional_index[word][i].append(j)
    return dict(sorted(positional_index.items()))


def index_documents() -> tuple:
    processed_documents = process_documents(DOCUMENT_PATH)
    positional_index = create_positional_index(processed_documents)
    inverted_index = create_inverted_index(processed_documents)

    write_index_to_file(positional_index, POSITIONAL_INDEX_FILE)

    return inverted_index, positional_index


def search_query(query: str, proximity: int, inverted_index: dict, positional_index: dict) -> set:
    query = preprocess_query(query)
    common_docs = search(query, inverted_index)
    query = preprocess(SPACE.join(query))
    result = set()
    for doc in common_docs:
        for i in range(len(positional_index[query[0]][doc])):
            for j in range(len(positional_index[query[1]][doc])):
                if abs(positional_index[query[0]][doc][i] - positional_index[query[1]][doc][j]) <= proximity:
                    result.add(doc)
                    break

    return result


def main():
    """Main function"""
    inverted_index, positional_index = index_documents()
    proximity = int(input("Enter the proximity: "))
    query = input("Enter the query: ")

    result = search_query(query, proximity, inverted_index, positional_index)
    if not result:
        print("No results found!")
    else:
        get_documents_from_index(result, DOCUMENT_PATH)


if __name__ == "__main__":
    while True:
        main()
