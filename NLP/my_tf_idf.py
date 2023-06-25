from collections import Counter
import numpy as np
import pandas as pd
import math
def compute_tf(text):
    tf_text = Counter(text)
    for i in tf_text:
        tf_text[i] = tf_text[i]/float(len(text))
    return tf_text

def compute_idf(word, texts):
    N = len(texts)
    idf_value = 0

    for text in texts:
        if word in text:
            idf_value += 1

    idf_value = math.log10(N / float(idf_value))

    return idf_value

def compute_tfidf(texts):
    texts_list = []

    for text in texts:
        words = text.split()
        tf_idf_dict = {}
        computed_tf = compute_tf(words)

        for word in computed_tf:
            tf_idf_dict[word] = computed_tf[word] * compute_idf(word, texts)

        texts_list.append(tf_idf_dict)

    return texts_list
