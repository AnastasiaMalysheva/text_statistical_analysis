import numpy as np
import re
import pymorphy2
import string
from nltk.corpus import stopwords
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt


class TextStatistics(object):

    def __init__(self, text):
        self.text = text

    def _preprocess_text(self, drop_stop=True, normalize=True):
        regex = re.compile('[%s]' % re.escape(string.punctuation))
        norm_text = regex.sub('', self.text).lower()
        if drop_stop:
            norm_text = ' '.join([word for word in norm_text.split() if word not in stopwords.words('russian')])
        if normalize:
            morph = pymorphy2.MorphAnalyzer()
            norm_text = ' '.join([morph.parse(word)[0].normal_form for word in norm_text.split()])
        return norm_text

    def _get_n_gramms(self, n, text=False):
        if not text:
            words = self.text.split()
        else:
            words = text.split()
        n_gramms = []
        for i in range(len(words) - n):
            n_gramms.append('_'.join(words[i:i + n]))
        return n_gramms

    def calculate_main_statics(self):
        sentences = self.text.split('.')
        st = {}
        st['mean_sentence_length'] = np.mean([len(sent) for sent in sentences])
        words = self.text.split()
        st['mean_words_length'] = np.mean([len(word) for word in words])
        st['n_unique_forms'] = len(set(words))
        norm_text = self._preprocess_text()
        st['n_unique'] = len(set(norm_text.split()))
        return st

    def get_frequent_statistics(self, max_n_gram=3, preprocess=True, plot_most_frequent=True, n_freq=50):
        norm_text = self._preprocess_text()
        words = norm_text.split()
        n_grams_counts = []
        n_grams_counts += words
        for i in range(2, max_n_gram + 1):
            n_grams_counts += self._get_n_gramms(i, norm_text)
        if plot_most_frequent:
            pd.Series(dict(Counter(n_grams_counts).most_common(n_freq))).plot(kind='bar')
            plt.show()
        return Counter(n_grams_counts)

    def get_morph_statistics(self):
        morph = pymorphy2.MorphAnalyzer()
        morph_statistics = {}
        parts_of_speech = [morph.parse(word)[0].tag.POS for word in self.text.split()]
        morph_statistics['parts_of_speech'] = Counter([word for word in parts_of_speech if word != None])
        animacy = [morph.parse(word)[0].tag.animacy for word in self.text.split()]
        morph_statistics['animacy'] = Counter([word for word in animacy if word != None])
        aspect = [morph.parse(word)[0].tag.aspect for word in self.text.split()]
        morph_statistics['aspect'] = Counter([word for word in aspect if word != None])
        case = [morph.parse(word)[0].tag.case for word in self.text.split()]
        morph_statistics['case'] = Counter([word for word in case if word != None])
        gender = [morph.parse(word)[0].tag.gender for word in self.text.split()]
        morph_statistics['gender'] = Counter([word for word in gender if word != None])
        number = [morph.parse(word)[0].tag.number for word in self.text.split()]
        morph_statistics['number'] = Counter([word for word in number if word != None])
        tense = [morph.parse(word)[0].tag.tense for word in self.text.split()]
        morph_statistics['tense'] = Counter([word for word in tense if word != None])
        return morph_statistics

    def get_lexical_density(self, terms_list):
        normal_text = self._preprocess_text()
        words = normal_text.split()
        n_words = len(words)
        n_terms = 0
        for term in terms_list:
            n_terms += sum([1 for word in words if word==term])
        return n_terms/n_words