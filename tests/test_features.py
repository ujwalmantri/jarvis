import pytest
from src.features import tokenize, build_vocabulary, sentence_to_vector

def test_tokenize_works():
    assert tokenize("Hello, World! I’m Jarvis! Bye-Bye") == ["hello", "world", "im", "jarvis", "bye", "bye"]

def test_build_vocabulary_works():
    assert build_vocabulary(["hello", "world", "im", "jarvis", "bye", "bye"]
) == ["bye", "hello", "im","jarvis","world"]

def test_sentence_to_vector_works():
    vocabulary = ["bye", "hello", "im","jarvis","world"]
    assert sentence_to_vector(["hello", "world", "jarvis", "bye", "bye", "sayonara"], vocab_list=vocabulary) == [2, 1, 0, 1, 1]