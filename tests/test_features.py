import pytest
from src.features import tokenize, build_vocabulary, sentence_to_vector

def test_tokenize_works():
    assert tokenize("Hello, World!") == ["hello", "world"]

def test_tokenize_convert_hyphens_to_whitespaces(): 
    assert tokenize("open-browser") == ["open", "browser"]

def test_tokenize_works_for_unicode_apostrophe():
    assert tokenize("what’s the time") == ["whats", "the", "time"]

def test_build_vocabulary_works():
    assert build_vocabulary(["hello", "world", "im", "jarvis", "bye", "bye"]
) == ["bye", "hello", "im","jarvis","world"]

def test_sentence_to_vector_works():
    vocabulary = ["bye", "hello", "im","jarvis","world"]
    assert sentence_to_vector(["hello", "world", "jarvis", "bye", "bye", "sayonara"], vocab_list=vocabulary) == [2, 1, 0, 1, 1]