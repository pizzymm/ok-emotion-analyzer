import numpy as np
from collections import defaultdict
import pandas as pd
import gensim

def load_data_and_cv(data_folder, cv):
    """
    Loads data and split into 10 folds.
    """
    revs = []
    pos_file = data_folder[0]
    neu_file = data_folder[1]
    neg_file = data_folder[2]
    vocab = defaultdict(float)  # 初始化为0.0
    with open(pos_file, "rb") as f:
        for line in f:
            rev = []
            rev.append(line.strip())
            orig_rev = line.strip()
            words = set(orig_rev.split())  # 默认以空格分割
            for word in words:
                vocab[word] += 1
            datum = {"y":0,  # 0代表积极
                      "text": orig_rev,
                      "num_words": len(orig_rev.split()),
                      "split": np.random.randint(0,cv)}
            revs.append(datum)
    with open(neu_file, "rb") as f:
        for line in f:
            rev = []
            rev.append(line.strip())
            orig_rev = line.strip()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum = {"y":1,  # 1代表中性
                      "text": orig_rev,
                      "num_words": len(orig_rev.split()),
                      "split": np.random.randint(0,cv)}
            revs.append(datum)
    with open(neg_file, "rb") as f:
        for line in f:       
            rev = []
            rev.append(line.strip())
            orig_rev = line.strip()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum = {"y":2,  # 2代表消极
                      "text": orig_rev,                             
                      "num_words": len(orig_rev.split()),
                      "split": np.random.randint(0,cv)}
            revs.append(datum)
    return revs, vocab
    

def load_w2v_than_count(w2v_file, vocab):
    w2v_model = gensim.models.Word2Vec.load_word2vec_format(w2v_file, binary=True, encoding='utf8', unicode_errors='ignore')
    count = 0
    for word in vocab.keys():
        if word.decode() in w2v_model:
            count += 1
    return count


if __name__=="__main__":
    w2v_file = "/home/lvchao/vectors/vectors_300.bin"
    data_folder = ["./data/1.txt", "./data/0.txt", "./data/-1.txt"]

    print("loading data...")
    revs, vocab = load_data_and_cv(data_folder, cv=10)
    max_l = np.max(pd.DataFrame(revs)["num_words"])
    print("data loaded!")

    print("number of sentences: " + str(len(revs)))
    print("vocab size: " + str(len(vocab)))
    print("max sentence length: " + str(max_l) + "\n")

    print("loading word2vec...",)
    count = load_w2v_than_count(w2v_file, vocab)
    print("word2vec loaded!")

    print("num words already in word2vec: " + str(count))
    print("vocabulary coverage rate of the data in word2vec:" + str(count / len(vocab)))


