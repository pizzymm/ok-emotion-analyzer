import numpy as np
import gensim

"""
Convert words to its matrix
"""
def NC_wordsToMatrix(words, w2vModel, Suqence_length):
    matrix = list()
    zeroVector = [0.0] * 300
    count = 0
    for i in range(len(words)):
        count += 1
        if words[i] in w2vModel:
            vec = w2vModel[words[i]]
        else:
            vec = np.random.uniform(-0.25, 0.25, 300)

        matrix.extend(vec)

    if count > Suqence_length:
        return matrix
    else:
        while count < Suqence_length:
            matrix.extend(zeroVector)
            count += 1
    return matrix

def LC_wordsToMatrix(words, w2vModel, Suqence_length):
    matrix = list()
    zeroVector = [0.0] * 600
    count = 0
    for i in range(len(words)):
        count += 1
        if words[i] in w2vModel:
            vec = w2vModel[words[i]]
        else:
            vec = np.random.uniform(-0.25, 0.25, 300)

        if i - 1 < 0:
            vec = np.hstack(([0.0] * 300, vec))
        else:
            if words[i - 1] in w2vModel:
                vec = np.hstack((w2vModel[words[i - 1]], vec))
            else:
                vec = np.hstack((np.random.uniform(-0.25, 0.25, 300), vec))

        matrix.extend(vec)

    if count > Suqence_length:
        return matrix
    else:
        while count < Suqence_length:
            matrix.extend(zeroVector)
            count += 1
    return matrix


def RC_wordsToMatrix(words, w2vModel, Suqence_length):
    matrix = list()
    zeroVector = [0.0] * 600
    count = 0
    for i in range(len(words)):
        count += 1
        if words[i] in w2vModel:
            vec = w2vModel[words[i]]
        else:
            vec = np.random.uniform(-0.25, 0.25, 300)

        if i + 1 < len(words):
            if words[i + 1] in w2vModel:
                vec = np.hstack((vec, w2vModel[words[i + 1]]))
            else:
                vec = np.hstack((vec, np.random.uniform(-0.25, 0.25, 300)))
        else:
            vec = np.hstack((vec, [0.0] * 300))

        matrix.extend(vec)

    if count > Suqence_length:
        return matrix
    else:
        while count < Suqence_length:
            matrix.extend(zeroVector)
            count += 1
    return matrix


def LRC_wordsToMatrix(words, w2vModel, Suqence_length):
    matrix = list()
    zeroVector = [0.0] * 900
    count = 0
    for i in range(len(words)):
        count += 1
        if words[i] in w2vModel:
            vec = w2vModel[words[i]]
        else:
            vec = np.random.uniform(-0.25, 0.25, 300)

        if i - 1 < 0:
            vec = np.hstack(([0.0] * 300, vec))
        else:
            if words[i-1] in w2vModel:
                vec = np.hstack((w2vModel[words[i-1]], vec))
            else:
                vec = np.hstack((np.random.uniform(-0.25, 0.25, 300), vec))

        if i + 1 < len(words):
            if words[i+1] in w2vModel:
                vec = np.hstack((vec, w2vModel[words[i+1]]))
            else:
                vec = np.hstack((vec, np.random.uniform(-0.25, 0.25, 300)))
        else:
            vec = np.hstack((vec, [0.0] * 300))

        matrix.extend(vec)

    if count > Suqence_length:
        return matrix
    else:
        while count < Suqence_length:
            matrix.extend(zeroVector)
            count += 1
    return matrix


"""
#  Return the feature matrix and its label matrix
"""
def load_data_label(positive_data_file, negtive_data_file, neutral_data_file, w2vModelPath, sequence_length, cxt_type):
    neg_file = negtive_data_file
    neu_file = neutral_data_file
    pos_file = positive_data_file
    # w2vModel = gensim.models.Word2Vec.load_word2vec_format(w2vModelPath, binary=True, encoding='utf8', unicode_errors='ignore')
    w2vModel = gensim.models.KeyedVectors.load_word2vec_format(w2vModelPath, binary=True, encoding='utf8', unicode_errors='ignore')
    X = list()
    Y = list()
    with open(neg_file, 'r',encoding='utf8') as f:
        for line in f:
            label = [1, 0, 0]
            words = line.split()
            if cxt_type== "LC":
                matrix = LC_wordsToMatrix(words, w2vModel, sequence_length)
            elif cxt_type == "RC":
                matrix = RC_wordsToMatrix(words, w2vModel, sequence_length)
            elif cxt_type == "LRC":
                matrix = LRC_wordsToMatrix(words, w2vModel, sequence_length)
            else:
                matrix = NC_wordsToMatrix(words, w2vModel, sequence_length)
            X.append(matrix)
            Y.append(label)
    with open(neu_file, 'r',encoding='utf8') as f:
        for line in f:
            label = [0, 1, 0]
            words = line.split()
            if cxt_type== "LC":
                matrix = LC_wordsToMatrix(words, w2vModel, sequence_length)
            elif cxt_type == "RC":
                matrix = RC_wordsToMatrix(words, w2vModel, sequence_length)
            elif cxt_type == "LRC":
                matrix = LRC_wordsToMatrix(words, w2vModel, sequence_length)
            else:
                matrix = NC_wordsToMatrix(words, w2vModel, sequence_length)
            X.append(matrix)
            Y.append(label)
    with open(pos_file, 'r',encoding='utf8') as f:
        for line in f:
            label = [0, 0, 1]
            words = line.split()
            if cxt_type== "LC":
                matrix = LC_wordsToMatrix(words, w2vModel, sequence_length)
            elif cxt_type == "RC":
                matrix = RC_wordsToMatrix(words, w2vModel, sequence_length)
            elif cxt_type == "LRC":
                matrix = LRC_wordsToMatrix(words, w2vModel, sequence_length)
            else:
                matrix = NC_wordsToMatrix(words, w2vModel, sequence_length)
            X.append(matrix)
            Y.append(label)

    # change the list format to matrix format
    x_data = np.array(X)
    y_data = np.array(Y)

    return x_data, y_data


"""
Generate a batch of data for training
"""
def batch_iter(data, batch_size, shuffle=True):
    data = np.array(data)
    data_size = data.shape[0]
    num_batchs_each_epoch = int(data_size / batch_size) + 1

    if shuffle:
        shuffle_indices = np.random.permutation(data_size)
        shuffle_data = data[shuffle_indices]
    else:
        shuffle_data = data
    for num_batch in range(num_batchs_each_epoch):
        start_index = num_batch * batch_size
        end_index = min((num_batch + 1) * batch_size, data_size)
        yield shuffle_data[start_index: end_index]
