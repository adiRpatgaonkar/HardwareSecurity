# main.py
#
# Created by Aditya Patgaonkar
# 03/03/2020
#
# Updates the years a farmer market was active
#
# Usage:python main_1.py
#
# Training, validation, testing code
# 
# To create train.npz run DataPreprocessing.py first 
# If already existing ignore
#

import numpy as np
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier

PATH = "./train.npz"


def show_data_info(np_array, data=False, shape=False):
    print("Features:")
    if data:
        print("{} -- ".format(np_array['data']), end=" ")
    if shape: 
        print("{}".format(np_array['data'].shape))
    print("Labels:")
    # print("{} -- ".format(np_array['label'], end=" ")
    if shape:
        print("{}".format(np_array['label'].shape))

def train_val_split(dataset):
    # Training, validation split
    trainset = {"data":np.zeros((80*26, 3000)), "label":np.zeros((80*26, 1))}
    validset = {"data":np.zeros((20*26, 3000)), "label":np.zeros((20*26, 1))}
    data_ptr = train_ptr = valid_ptr = 0
    idx = 0
    while (data_ptr <= 2500):
        # print(validset['data'][valid_ptr:valid_ptr+20, :])
        # print("IDX: {}, DPR: {}, TPR: {}, VPR: {}".format(idx, data_ptr, train_ptr, valid_ptr))
        trainset['data'][train_ptr:train_ptr+80, :] = dataset['data'][data_ptr:data_ptr+80, :]
        trainset['label'][train_ptr:train_ptr+80, :] = dataset['label'][data_ptr:data_ptr+80]
        data_ptr += 80
        validset['data'][valid_ptr:valid_ptr+20, :] = dataset['data'][data_ptr:data_ptr+20, :]
        validset['label'][valid_ptr:valid_ptr+20, :] = dataset['label'][data_ptr:data_ptr+20]
        data_ptr += 20
        train_ptr += 80
        valid_ptr += 20
        idx += 1
    return (trainset, validset)


def main():

    # Dataset
    print("Loading dataset ... ", end="")
    dataset = dict(np.load(PATH))
    dataset['label'] = dataset['label'][:, np.newaxis]
    print("done.")
    show_data_info(dataset, shape=True)

    print("Splitting dataset ... ", end="")
    trainset, validset = train_val_split(dataset)
    print("done.")
    # show_data_info(trainset, data=True)
    # show_data_info(validset, data=True)

    # Transforms
    print("Applying data transforms ... ", end="")
    scaler = preprocessing.StandardScaler().fit(trainset['data'])
    trainset['data'] = scaler.transform(trainset['data'])
    # print(scaler.mean_, scaler.scale_)
    validset['data'] = scaler.transform(validset['data'])
    print("done.")
    # show_data_info(trainset, data=True, shape=True)
    # show_data_info(validset, data=True, shape=True)

    # Model
    print("Creating model ... ", end="")
    net = MLPClassifier(solver='sgd', alpha=1e-5, 
                        hidden_layer_sizes=(100, 26),
                        max_iter=300, 
                        random_state=1)
    print("done.")

    # Train
    print("Training model ... ", end="")
    net.fit(trainset['data'], np.squeeze(trainset['label']))
    print("done.")
    # print(net.predict(validset['data']))

    # Validation score
    score = net.score(validset['data'], np.squeeze(validset['label']))
    print("Score on validation set: {}".format(score*100))







if __name__ == "__main__":
    main()
