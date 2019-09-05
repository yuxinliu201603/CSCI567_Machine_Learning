import json
import numpy as np


###### Q1.1 ######
def objective_function(X, y, w, lamb):
    """
    Inputs:
    - Xtrain: A 2 dimensional numpy array of data (number of samples x number of features)
    - ytrain: A 1 dimensional numpy array of labels (length = number of samples )
    - w: a numpy array of D elements as a D-dimension weight vector
    - lamb: lambda used in pegasos algorithm

    Return:
    - obj_value: the value of objective function in SVM primal formulation
    """
    # you need to fill in your solution here
    X = np.array(X)
    N = X.shape[0]
    y= np.array(y)
    # print(X.shape,'x')
    # print(w.shape, 'w')
    temp_1 = (np.linalg.norm(w, ord=2))
    # print(temp_1,'temp_1')
    temp_2 = y* (np.dot(X,w))
    # print(temp_2.shape)
    ta = np.ones((N,1))
    # print(ta)
    taa = ta - temp_2
    # print(taa)
    obj_value = np.sum(np.where(taa > 0, taa, 0)) / N
    # print(obj_value,'obj')
    # temp_1 = min((lamb/2)*(np.linalg.norm(w, ord=2)))


    # obj_value = []
    return obj_value



###### Q1.2 ######
def pegasos_train(Xtrain, ytrain, w, lamb, k, max_iterations):
    """
    Inputs:
    - Xtrain: A list of num_train elements, where each element is a list of D-dimensional features.
    - ytrain: A list of num_train labels
    - w: a numpy array of D elements as a D-dimension vector, which is the weight vector and initialized to be all 0s
    - lamb: lambda used in pegasos algorithm
    - k: mini-batch size
    - max_iterations: the total number of iterations to update parameters

    Returns:
    - learnt w
    - train_obj: a list of the objective function value at each iteration during the training process, length of 500.
    """
    np.random.seed(0)
    Xtrain = np.array(Xtrain)
    ytrain = np.array(ytrain)
    N = Xtrain.shape[0]
    D = Xtrain.shape[1]

    train_obj = []
    # print(Xtrain.shape)
    # print(w.shape)

    for iter in range(1, max_iterations + 1):
        A_t = np.floor(np.random.rand(k) * N).astype(int)  # index of the current mini-batch
        X = Xtrain[A_t]
        y = ytrain[A_t]

        y = np.expand_dims(y, axis = -1)
        # print(X.shape,'x')
        # print(y.shape,'y')
        At = y* (np.dot(X,w))

        # print(At.shape,'at')
        Temp_N = At.shape[0]
        At = At.reshape(Temp_N,)
        # print(At.shape)
        ix = (At < 1)
        # print(ix)
        # print(X[ix].shape)
        # print(y[ix].shape)
        nt = 1 / (lamb * iter)
        # print(w.shape,'w')
        tempppp = (1 - nt * lamb) * w
        tempp = (nt / k) * (np.dot(X[ix].T, y[ix]))
        # print(tempppp.shape,'tempppp')
        # print(tempp.shape,'temp')
        w_05 = tempp + tempppp

        # print(w_05.shape,'w_05')

        w_norm = np.linalg.norm(w_05, ord=2)
        ww = (1/np.sqrt(lamb))/w_norm
        w = np.where(ww < 1, ww, 1) * w_05
        # print(w.shape,'w1')
        obj = objective_function(X, y, w, lamb)
        train_obj.append(obj)



        # you need to fill in your solution here
    # w = []
    # train_obj = 0
    # print(train_obj,'train_obj')
    return w, train_obj

    #
    # np.random.seed(0)
    # Xtrain = np.array(Xtrain)
    # ytrain = np.array(ytrain)
    # N = Xtrain.shape[0]
    # D = Xtrain.shape[1]
    #
    # train_obj = []
    # wshape = w.shape
    # w_ = w
    # for iter in range(1, max_iterations + 1):
    #     A_t = np.floor(np.random.rand(k) * N).astype(int)  # index of the current mini-batch
    #
    #     # you need to fill in your solution here
    #     yt = ytrain[A_t]
    #     Xt = Xtrain[A_t]
    #     # train_obj.append(objective_function(Xt,yt,w_,lamb))
    #
    #     A_tp = np.where(yt.reshape(-1, 1) * np.dot(Xt, w_) < 1)
    #     # if A_tp[0].size == 0:
    #     #    continue
    #     Xtp = Xt[A_tp[0]]
    #     ytp = yt.reshape(-1, 1)[A_tp]
    #     nt = 1 / lamb / iter
    #     wt2 = (1 - nt * lamb) * w_ + nt / k * ((ytp.reshape(-1, 1) * Xtp).sum(axis=0).reshape(-1, 1))
    #     _norm = np.dot(wt2.T, wt2)[0, 0] ** 0.5
    #     w_ = np.min([1., 1. / lamb ** 0.5 / _norm]) * wt2
    #     train_obj.append(objective_function(Xtrain, ytrain, w_, lamb))
    #     # import pdb; pdb.set_trace()
    #     # if iter == 1:
    #     #    import pdb; pdb.set_trace()
    # # w = w_.reshape(wshape).tolist()
    # w = w_
    #
    # # import matplotlib.pyplot as plt
    # # plt.plot(np.arange(len(train_obj)),train_obj)
    # # plt.show()
    # return w, train_obj


###### Q1.3 ######
def pegasos_test(Xtest, ytest, w_l):
    """
    Inputs:
    - Xtest: A list of num_test elements, where each element is a list of D-dimensional features.
    - ytest: A list of num_test labels
    - w_l: a numpy array of D elements as a D-dimension vector, which is the weight vector of SVM classifier and learned by pegasos_train()
 
    Returns:
    - test_acc: testing accuracy.
    """
    # you need to fill in your solution here
    Xtest = np.array(Xtest)
    N = Xtest.shape[0]
    ytest = np.expand_dims(np.array(ytest),-1)
    # print(ytest.shape,'ytest')
    b = np.dot(Xtest, w_l)
    # print(b.shape,'b')
    re = ytest*(b)
    # print(re.shape)
    ree = (re > 0).astype(int)
    a = np.sum(ree)
    # print(a,'a')
    # print(N,'N')
    test_acc = a / N

    return test_acc
    # num_cor = len(np.where(np.array(ytest).reshape(-1, 1) * np.dot(Xtest, np.array(w_l).reshape(-1, 1)) > 0)[0])
    # test_acc = num_cor / len(ytest)
    # return test_acc


"""
NO MODIFICATIONS below this line.
You should only write your code in the above functions.
"""

def data_loader_mnist(dataset):

    with open(dataset, 'r') as f:
            data_set = json.load(f)
    train_set, valid_set, test_set = data_set['train'], data_set['valid'], data_set['test']

    Xtrain = train_set[0]
    ytrain = train_set[1]
    Xvalid = valid_set[0]
    yvalid = valid_set[1]
    Xtest = test_set[0]
    ytest = test_set[1]

    ## below we add 'one' to the feature of each sample, such that we include the bias term into parameter w
    Xtrain = np.hstack((np.ones((len(Xtrain), 1)), np.array(Xtrain))).tolist()
    Xvalid = np.hstack((np.ones((len(Xvalid), 1)), np.array(Xvalid))).tolist()
    Xtest = np.hstack((np.ones((len(Xtest), 1)), np.array(Xtest))).tolist()

    for i, v in enumerate(ytrain):
        if v < 5:
            ytrain[i] = -1.
        else:
            ytrain[i] = 1.
    for i, v in enumerate(ytest):
        if v < 5:
            ytest[i] = -1.
        else:
            ytest[i] = 1.

    return Xtrain, ytrain, Xvalid, yvalid, Xtest, ytest


def pegasos_mnist():

    test_acc = {}
    train_obj = {}

    Xtrain, ytrain, Xvalid, yvalid, Xtest, ytest = data_loader_mnist(dataset = 'mnist_subset.json')
    # objective_function(Xtrain, ytrain, [], lamb)
    # print(len(Xtrain[0]), 'Xtrain shape')
    max_iterations = 500
    k = 100
    # k = 5
    for lamb in (0.01, 0.1, 1):
        w = np.zeros((len(Xtrain[0]), 1))
        w_l, train_obj['k=' + str(k) + '_lambda=' + str(lamb)] = pegasos_train(Xtrain, ytrain, w, lamb, k, max_iterations)
        test_acc['k=' + str(k) + '_lambda=' + str(lamb)] = pegasos_test(Xtest, ytest, w_l)

    lamb = 0.1
    for k in (1, 10, 1000):
        w = np.zeros((len(Xtrain[0]), 1))
        w_l, train_obj['k=' + str(k) + '_lambda=' + str(lamb)] = pegasos_train(Xtrain, ytrain, w, lamb, k, max_iterations)
        test_acc['k=' + str(k) + '_lambda=' + str(lamb)] = pegasos_test(Xtest, ytest, w_l)
    # test_acc, train_obj = [],[]
    return test_acc, train_obj


def main():
    test_acc, train_obj = pegasos_mnist() # results on mnist
    print('mnist test acc \n')
    for key, value in test_acc.items():
        print('%s: test acc = %.4f \n' % (key, value))

    with open('pegasos.json', 'w') as f_json:
        json.dump([test_acc, train_obj], f_json)


if __name__ == "__main__":
    main()
