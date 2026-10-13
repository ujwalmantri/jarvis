import string

def tokenize(text):
    """
    Removes punctuations and casing from a given sentence.
    Also removes unicode apostrophe (’).
    Replaces hyphens with whitespaces.  

    Args: 
    text (string) - plain text

    Returns:
    cleaned_words (list) - clean list of words ignoring casing, punctuation, hyphens, and unicodeapostrophe.
    """
    text = text.replace('-', ' ')
    text = text.replace('’', "'").replace('‘', "'")
    text = text.replace('“', '"').replace('”', '"')
    words = text.split()
    cleaned_words = []
    for word in words:
        cleaned = word.translate(str.maketrans('', '', string.punctuation)).lower()
        if cleaned and any(char.isalnum() for char in cleaned):
            cleaned_words.append(cleaned)
    return cleaned_words

def sentence_to_vector(sentence, vocab_list):
    """
    Turns a sentence into bag of words vector.

    Args:
    sentence (list) - a list of tokenized words that needs to be vectorized.
    vocab_list (list)- a vocabulary list on which the model is trained.

    Returns:
    vector (list) - vectorized form of the sentence.
    """

    vector = []

    for word in vocab_list:
        count = sentence.count(word)
        vector.append(count)

    return vector

def build_vocabulary(texts):
    """
    Turns a data into a sorted vocabulary list.

    Args:
    texts (list) - data i.e. list of sentences.

    Returns:
    vocabulary (list) - a list of unique words in the data.
    """

    vocab = set()
    for sentence in texts:
        words = tokenize(sentence)
        vocab.update(words)

    vocabulary = sorted(vocab)

    return vocabulary