from assignment1.question1 import EMPTY, VOWELS_ZERO, ZERO, PHONETIC_DICTIONARY, FOUR, DOCUMENT_PATH, \
    read_dir, case_fold, remove_punctuation, expand_contractions, tokenize, write_index_to_file, search, \
    remove_stopwords, AND, OR, NOT, get_documents_from_index

SOUNDEX_INDEX_FILE = 'soundex_index.txt'


def replace_vowels(word: str) -> str:
    """Replaces the vowels in the word with 0"""
    return word[0] + EMPTY.join([ZERO if char in VOWELS_ZERO else char for char in word[1:]])


def remove_consecutive_duplicates(word: str) -> str:
    """Removes the consecutive duplicates in the word"""
    result = word[0]
    for i in range(1, len(word)):
        if word[i] != word[i - 1]:
            result += word[i]
    return result


def soundex(word: str, length: int) -> str:
    """Returns the soundex code of the word"""
    if word.isnumeric():
        return EMPTY

    word = word.upper()
    word = replace_vowels(word)

    result = ""
    result += word[0]
    for i in range(1, len(word)):
        if word[i] not in PHONETIC_DICTIONARY:
            result += word[i]
        else:
            result += PHONETIC_DICTIONARY[word[i]]

    result = remove_consecutive_duplicates(result)
    result = result.replace(ZERO, EMPTY)

    if len(result) < length:
        result += ZERO * (length - len(result))
    else:
        result = result[:length]

    return result


def create_soundex_index(documents: list) -> dict:
    """Creates the soundex index from the documents"""
    soundex_index = {}
    for i, document in enumerate(documents):
        for word in document:
            soundex_code = soundex(word, FOUR)
            if soundex_code not in soundex_index:
                soundex_index[soundex_code] = {i}
            else:
                soundex_index[soundex_code].add(i)
    return dict(sorted(soundex_index.items()))


def preprocess_for_soundex(string: str) -> list:
    """Preprocesses the string for soundex"""
    string = case_fold(string)
    string = expand_contractions(string)
    string = remove_punctuation(string)
    string = remove_stopwords(string)
    return tokenize(string)


def index_soundex() -> dict:
    documents = read_dir(DOCUMENT_PATH)
    processed_documents = [preprocess_for_soundex(document) for document in documents]

    soundex_index = create_soundex_index(processed_documents)

    write_index_to_file(soundex_index, SOUNDEX_INDEX_FILE)

    return soundex_index


def preprocess_query(query: str) -> list:
    """Preprocesses the query"""
    query = query.split()
    for i in range(len(query)):
        if query[i].lower() not in [AND, OR, NOT]:
            query[i] = preprocess_for_soundex(query[i])[0]
    return query


def search_soundex_index(query: str, soundex_index: dict) -> set:
    """Searches the soundex index for the query"""
    query = preprocess_query(query)
    # ig word is  "AND" or "OR" or "NOT" then it will not be converted to soundex
    query = [soundex(word, FOUR) if word not in [AND, OR, NOT] else word for word in query]
    return search(query, soundex_index)


def main():
    """Main function"""
    soundex_index = index_soundex()

    query = input("Enter the soundex query: ")
    result = search_soundex_index(query, soundex_index)

    if not result:
        print("No results found!")
    else:
        get_documents_from_index(result, DOCUMENT_PATH)


if __name__ == '__main__':
    while True:
        main()
