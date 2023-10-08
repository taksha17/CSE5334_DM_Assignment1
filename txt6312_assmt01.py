import os
import math
import operator
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from collections import Counter

stemmer = PorterStemmer()
tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
corpusroot = './US_Inaugural_Addresses'
vector = {}
df = Counter()
tfs = {}
lengths = Counter()
postings_list = {}


def txt6312_root_files_count(directory):
    """Count the number of files in the directory."""
    return len(os.listdir(directory))


def txt6312_preprocess_document(doc):
    """Preprocess a document: tokenize, remove stopwords, and stem."""
    tokens = tokenizer.tokenize(doc.lower())
    sw = stopwords.words('english')
    return [stemmer.stem(token) for token in tokens if token not in sw]


def txt6312_calculate_weight(filename, token):
    """Calculate the weight of a token in a document without normalizing."""
    idf = txt6312_getidf(token)
    return (1 + math.log10(tfs[filename][token])) * idf


def txt6312_getidf(token):
    """Calculate IDF for a token."""
    if df[token] == 0:
        return -1
    return math.log10(len(tfs) / df[token])


def txt6312_initialize_data_structures():
    for filename in os.listdir(corpusroot):
        if filename.startswith(('0', '1')):
            with open(os.path.join(corpusroot, filename), "r", encoding='windows-1252') as f:
                tokens = txt6312_preprocess_document(f.read())
                tf = Counter(tokens)
                df.update(list(set(tokens)))
                tfs[filename] = tf.copy()

    for filename in tfs:
        vector[filename] = Counter()
        length = 0
        for token in tfs[filename]:
            weight = txt6312_calculate_weight(filename, token)
            vector[filename][token] = weight
            length += weight ** 2
        lengths[filename] = math.sqrt(length)

    for filename in vector:
        for token in vector[filename]:
            vector[filename][token] /= lengths[filename]
            if token not in postings_list:
                postings_list[token] = Counter()
            postings_list[token][filename] = vector[filename][token]


def txt6312_getweight(filename, token):
    """Get the normalized weight of a token in a document."""
    return vector[filename][stemmer.stem(token)]


def txt6312_query(qstring):
    tokens = txt6312_preprocess_document(qstring)
    tokens = dict(Counter(tokens))
    normalizingfactor = sum([(1 + math.log10(val)) ** 2 for val in tokens.values()]) ** 0.5

    for token in tokens:
        tokens[token] /= normalizingfactor

    if not any(token in postings_list for token in tokens):
        return ("None", 0)

    document_scores = {}
    not_in_topten = []
    toptenpostings_list = {token: postings_list[token] for token in tokens if token in postings_list}

    for token in toptenpostings_list:
        for filename in toptenpostings_list[token]:
            if filename not in document_scores:
                document_scores[filename] = 0

    for token in toptenpostings_list:
        for filename in document_scores:
            if filename in toptenpostings_list[token]:
                document_scores[filename] += toptenpostings_list[token][filename] * tokens[token]
            elif token in postings_list and filename in postings_list[token]:
                document_scores[filename] += postings_list[token][filename] * tokens[token]
                not_in_topten.append(filename)

    sorted_scores = sorted(document_scores.items(), key=operator.itemgetter(1), reverse=True)

    if sorted_scores[0][0] in not_in_topten:
        return ("fetch more", 0)
    else:
        return sorted_scores[0]

# Main execution
txt6312_initialize_data_structures()

print("%.12f" % txt6312_getidf('british'))
print("%.12f" % txt6312_getidf('union'))
print("%.12f" % txt6312_getidf('war'))
print("%.12f" % txt6312_getidf('power'))
print("%.12f" % txt6312_getidf('great'))
print("--------------")
print("%.12f" % txt6312_getweight('02_washington_1793.txt', 'arrive'))
print("%.12f" % txt6312_getweight('07_madison_1813.txt', 'war'))
print("%.12f" % txt6312_getweight('12_jackson_1833.txt', 'union'))
print("%.12f" % txt6312_getweight('09_monroe_1821.txt', 'british'))
print("%.12f" % txt6312_getweight('05_jefferson_1805.txt', 'public'))
print("--------------")
print("(%s, %.12f)" % txt6312_query("pleasing people"))
print("(%s, %.12f)" % txt6312_query("british war"))
print("(%s, %.12f)" % txt6312_query("false public"))
print("(%s, %.12f)" % txt6312_query("people institutions"))
print("(%s, %.12f)" % txt6312_query("violated willingly"))




    # Readme.txt
    # Author:Taksha Sachin Thosani
    # Student Id : 1002086312
    # References :-https://github.com/mayank408/TFIDF
    # https://github.com/bbc/Similarity
    # https://iyzico.engineering/how-to-calculate-tf-idf-term-frequency-inverse-document-frequency-from-the-beatles-biography-in-c4c3cd968296
    # Github Program link : https://github.com/taksha17/CSE5334_DM_Assignment1