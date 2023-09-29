import os
import math
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

class TextProcessor:
    def __init__(self, corpusroot):
        self.corpusroot = corpusroot
        self.docs = self._load_documents()
        self.tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.stemmed_docs = self._preprocess_documents()
        self.tf_docs = {filename: self._compute_tf(tokens) for filename, tokens in self.stemmed_docs.items()}

    def _load_documents(self):
        docs = {}
        for filename in os.listdir(self.corpusroot):
            if filename.startswith('0') or filename.startswith('1'):
                try:
                    with open(os.path.join(self.corpusroot, filename), "r", encoding='windows-1252') as file:
                        doc = file.read().lower()
                        docs[filename] = doc
                except FileNotFoundError:
                    print(f"Error: {filename} not found.")
                except Exception as e:
                    print(f"Error reading {filename}: {e}")
        return docs

    def _preprocess_documents(self):
        tokenized_docs = {filename: self.tokenizer.tokenize(doc) for filename, doc in self.docs.items()}
        filtered_docs = {filename: [word for word in tokens if word not in self.stop_words] for filename, tokens in tokenized_docs.items()}
        return {filename: [self.stemmer.stem(word) for word in tokens] for filename, tokens in filtered_docs.items()}

    def _compute_tf(self, tokens):
        tf = {token: tokens.count(token) for token in set(tokens)}
        for token, count in tf.items():
            tf[token] = 1 + math.log(count)
        return tf

    def _compute_idf(self, token):
        df = sum(1 for doc_tokens in self.stemmed_docs.values() if token in doc_tokens)
        if df == 0:
            return 0
        return math.log(len(self.stemmed_docs) / df)

    def txt6312_getidf(self, token):
        return self._compute_idf(token)

    def txt6312_getweight(self, filename, token):
        tf = self.tf_docs[filename].get(token, 0)
        idf = self.txt6312_getidf(token)
        return tf * idf

    def txt6312_query(self, q):
        q = q.lower()
        tokens = self.tokenizer.tokenize(q)
        tokens = [word for word in tokens if word not in self.stop_words]
        tokens = [self.stemmer.stem(word) for word in tokens]
        tf_query = self._compute_tf(tokens)
        
        max_sim = -float('inf')
        best_doc = None
        for filename, doc_tokens in self.stemmed_docs.items():
            dot_product = sum(tf_query[token] * self.txt6312_getweight(filename, token) for token in tokens)
            doc_length = math.sqrt(sum(self.txt6312_getweight(filename, token)**2 for token in doc_tokens))
            q_length = math.sqrt(sum(tf**2 for tf in tf_query.values()))
            cosine_sim = dot_product / (doc_length * q_length)
            if cosine_sim > max_sim:
                max_sim = cosine_sim
                best_doc = filename
        return best_doc, max_sim


if __name__ == "__main__":
    processor = TextProcessor('./US_Inaugural_Addresses')
    print("%.12f" % processor.txt6312_getidf('british'))
    print("%.12f" % processor.txt6312_getidf('union'))
    print("%.12f" % processor.txt6312_getidf('war'))
    print("%.12f" % processor.txt6312_getidf('military'))
    print("%.12f" % processor.txt6312_getidf('great'))
    print("--------------")
    print("%.12f" % processor.txt6312_getweight('02_washington_1793.txt','arrive'))
    print("%.12f" % processor.txt6312_getweight('07_madison_1813.txt','war'))
    print("%.12f" % processor.txt6312_getweight('12_jackson_1833.txt','union'))
    print("%.12f" % processor.txt6312_getweight('09_monroe_1821.txt','british'))
    print("%.12f" % processor.txt6312_getweight('05_jefferson_1805.txt','public'))
    print("--------------")
    print("(%s, %.12f)" % processor.txt6312_query("pleasing people"))
    print("(%s, %.12f)" % processor.txt6312_query("british war"))
    print("(%s, %.12f)" % processor.txt6312_query("false public"))
    print("(%s, %.12f)" % processor.txt6312_query("people institutions"))
    print("(%s, %.12f)" % processor.txt6312_query("violated willingly"))


    # Readme.txt
    # Author:Taksha Sachin Thosani
    # Student Id : 1002086312
    # References :-https://github.com/mayank408/TFIDF
    # https://github.com/bbc/Similarity
    # https://iyzico.engineering/how-to-calculate-tf-idf-term-frequency-inverse-document-frequency-from-the-beatles-biography-in-c4c3cd968296
    # Github Program link : https://github.com/taksha17/CSE5334_DM_Assignment1