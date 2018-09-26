import nltk
from nltk.corpus import brown
import numpy as np

# Step 1: Import the data, which is the tagged and untagged brown corpus
# Step 2: Find the trigram probabilties using the tagged corpus
# Transition probability: P(tag|prev 2 tags)
# Emission probability: P(word|tag, prev word)

def get_corpus(str):
    corpus = brown.words()
    tagged_corpus = brown.tagged_words()

    if str is 'tagged':
        return tagged_corpus
    else:
        return corpus

def get_corpus_counts(tagged_corpus):
    words_with_tags = dict()

    for (word, tag) in tagged_corpus:

        if word not in words_with_tags:
            words_with_tags[word] = {tag : 1}
        elif tag not in words_with_tags[word]:
            words_with_tags[word][tag] = 1
        else:
            words_with_tags[word][tag] += 1


    return words_with_tags


def get_tag_trigrams(corpus):
    
    tags = list()
    trigrams = {}
    
    for (word, tag) in corpus:
        tags.append(tag)

    for index, tag in enumerate(tags):
        if index > 1:
            three = (tags[index - 2], tags[index - 1], tag)
            if three not in trigrams:
                trigrams[three] = 0
            trigrams[three] += 1

    return trigrams

def get_word_bigram_tag(corpus):
    word_bigram_tag = dict()
    bigrams = list()
    words = list()
    tags = list()

    for (word, tag) in corpus:
        words.append(word)
        tags.append(tag)

    for index, word in enumerate(words):
        if index > 0:
            two = (words[index - 1], word)
        else:
            two = ('<s>', word)
        bigrams.append(two)

    for index, two in enumerate(bigrams):
        tag = tags[index]
        pair = (two, tag)
        if pair not in word_bigram_tag:
            word_bigram_tag[pair] = 0
        word_bigram_tag[pair] += 1

    print(word_bigram_tag)
    return word_bigram_tag

tagged_corpus = get_corpus('tagged')
corpus = get_corpus('X')
word_tag_counts = get_corpus_counts(tagged_corpus)
trans_counts = get_tag_trigrams(tagged_corpus)
emission_counts = get_word_bigram_tag(tagged_corpus)