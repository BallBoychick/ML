import numpy as np
import pandas as pd
import collections

def bag_of_words(texts):
    word_count = collections.defaultdict(int)
    tokenized_text = [text.split() for text in texts]

    for text in tokenized_text:
        for word in text:
            word_count[word] +=1
    
    word_index = {word: i for i, word in enumerate(word_count.keys())}

    bag_of_word_matrix = []
    for text in tokenized_text:
        word_vector = [0] * len(word_index)
        for word in text:
            if word in word_index:
                word_vector[word_index[word]] += 1
        bag_of_word_matrix.append(word_vector)
    
    return bag_of_word_matrix, word_index

