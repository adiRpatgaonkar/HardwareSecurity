# main.py
#
# Created by Aditya Patgaonkar
# 03/03/2020
#
# Training, validation, hyper-parameter tuning
# and testing key stroke prediction code
# using .wav files
#
# Usage:python main_1.py
#
# Training, validation, testing code
# 
# To create train.npz run DataPreprocessing.py first 
# If already existing ignore

import numpy as np
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix
import pickle
import string

from DataProcessing import KeyExtracter

PATH = "./train.npz" ## PATH of training set
n_folds = 5
filename = "./model.pkl" ## PATH of saved model


def show_data_info(dataset, data=False, shape=False):
    print("Features:")
    if data:
        print("{} -- ".format(dataset['data']), end=" ")
    if shape: 
        print("{}".format(dataset['data'].shape))
    print("Labels:")
    # print("{} -- ".format(np_array['label'], end=" ")
    if shape:
        print("{}".format(dataset['label'].shape))

def train_val_split(dataset, skl=False):
    '''
    Preps a seperate training and validation set by:
    For every 100 samples:
    1st 80: trainset
    Remaining 20: validation set

    '''
    # Training, validation split
    if skl:
        trainset, validset = {}, {}
        trainset['data'], validset['data'], \
        trainset['label'], validset['label'] = train_test_split(dataset['data'], 
                                                                dataset['label'], 
                                                                test_size=0.2, 
                                                                random_state=1)
    else:
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

def get_transformed_data(dataset):
    scaler = preprocessing.StandardScaler().fit(dataset['data'])
    dataset['data'] = scaler.transform(dataset['data'])
    return (dataset, scaler)

def grid_search_hyp(model, dataset):
     ## Grid search hyper parameters
    hidden_layer_sizes = [(10, 10), (100, 100)]#, (200, 200), 
                        #(300, 300), (400, 400)]
    param_grid = dict(hidden_layer_sizes=hidden_layer_sizes)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, 
                            n_jobs=2, cv=n_folds)
    grid_result = grid.fit(dataset['data'], dataset['label'])
    return grid_result

def train(model, dataset, save=True, fname="model.pkl"):
    model.fit(dataset['data'], np.squeeze(dataset['label']))
    if save:
        ## Save trained model
        pickle.dump(model, open(fname, 'wb'))
    return model


def main():

    ## Dataset
    print("Loading dataset ... ", end="")
    dataset = dict(np.load(PATH))
    print("done.")
    show_data_info(dataset, shape=True)

    ## Split train, validation sets using sklearn
    # print("Splitting dataset ... ", end="")
    # trainset, validset = train_test_split(dataset, skl=True)
    # print("done.")
    # show_data_info(trainset, shape=True)
    # show_data_info(validset, shape=True)

    ## Transforms
    print("Applying data transforms ... ", end="")
    dataset, scaler = get_transformed_data(dataset)
    print("done.")
    # show_data_info(trainset, data=True, shape=True)
    # show_data_info(validset, data=True, shape=True)

    # Model
    print("Creating model ... ", end="")
    net = MLPClassifier(solver='sgd', alpha=1e-5, 
                        hidden_layer_sizes=(10, 10),
                        max_iter=200, random_state=1,
                        verbose=True)
    print("done.")

    # ## Grid search hyper parameters
    # grid_result = grid_search_hyp(net, dataset)
    # print("Grid searching hyper parameters done.\nSummary:")

    # ## Summarize results
    # print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    # means = grid_result.cv_results_['mean_test_score']
    # stds = grid_result.cv_results_['std_test_score']
    # params = grid_result.cv_results_['params']
    # ## Cross validation score
    # for mean, stdev, param in zip(means, stds, params):
    #     print("%f (%f) with: %r" % (mean, stdev, param))
    # idx = np.argmax(means)
    # print("Best K-fold cross validation score with ", end="")
    # print("K = {}: {:2f} (+/- {:2f}) for params: {}".format(n_folds,
    #                                                         means[idx],
    #                                                         stds[idx],
    
    #                                                         params[idx]))
    
    ## Train on the best hyperparmameters
    # net.hidden_layer_sizes = params[idx]['hidden_layer_sizes']
    ## To skip grid search
    net.hidden_layer_sizes = (300, 300)
    print("Training model ... ")
    net = train(net, dataset, save=True, fname=filename)
    print("Training finished")

    ## Load trained model
    print("Loading model from {} ... ".format(filename), end="")
    net = pickle.load(open(filename, 'rb'))
    print("done.")

    ## Validation score
    # score = net.score(validset['data'], np.squeeze(validset['label']))
    # print("Score on validation set: {}".format(score*100))

    ## Cross validation score
    # scores = cross_val_score(net, dataset['data'], dataset['label'])
    # print("Cross validation scores: \n{}".format(scores))
    
    ## Predictions on dataset & corresponding "Confusion Matrix"
    preds = net.predict(dataset['data'])

    ## To plot
    ## https://stackoverflow.com/questions/19233771/sklearn-plot-confusion-matrix-with-labels
    print(confusion_matrix(dataset['label'], preds))

    ## Inference the trained model
    print("Processing -- secret.wav ... ")
    key_secret = KeyExtracter(file_name='./data/secret.wav', num_clicks=8)
    key_secret.run()

    ## Prep (unseen) testing data
    testset = {}
    testset['data'] = key_secret.arrs
    print(testset['data'].shape)
    # Apply same transform
    testset['data'] = scaler.transform(testset['data'])

    ## PREDICTIONS
    preds = net.predict(testset['data'])
    print(preds)
    print("Key strokes: ", end="")
    for p in preds:
        print("{}".format(string.ascii_lowercase[p]), end=" ")
    print()


if __name__ == "__main__":
    main()
