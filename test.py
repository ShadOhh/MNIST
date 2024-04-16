import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# Loads the data from from the directory it is stored in. In this case it's the directory digit-recognizer/train.csv. 
data = pd.read_csv('digit-recognizer/train.csv')

# data head can be ignored it is just a visualizer for when using jupyter lab
data.head()

# saves the data as a numpy array, cause in order to use the data we want it in matrices.
# right now the data is saved as The number as Row and it's 784 pixels as columns for example
#
# obj1| p1 p2 p3 p4 p5 ... p784
# obj2| p1 p2 p3 p4 p5 ... p784
# obj3| p1 p2 p3 p4 p5 ... p784
# obj4| p1 p2 p3 p4 p5 ... p784
# obj5| p1 p2 p3 p4 p5 ... p784
# obj6| p1 p2 p3 p4 p5 ... p784
# obj7| p1 p2 p3 p4 p5 ... p784
# obj8| p1 p2 p3 p4 p5 ... p784
#   .
#   .
#   .
# obj(n)| p1 p2 p3 p4 p5 ... p784


# with p meaning pixel and the number meaning the pixel number. For example p2 means pixel 2
data = np.array(data)

# m, n saves the amount of data in each row and column. m is row and n is column.
m, n = data.shape

# if we print m,n we would get 42000 which is the amount of unique data while 785
# represents the amount of pixels for each of m

# un comment this to see it.
# print(m,n)


# Shuffles the data
np.random.shuffle(data)

# datatest is Transposed of data from the first unit to the 1000th.
# meaning instead of being
# obj1| p1 p2 p3 p4 p5 ... p784
# obj2| p1 p2 p3 p4 p5 ... p784
# obj3| p1 p2 p3 p4 p5 ... p784
# obj4| p1 p2 p3 p4 p5 ... p784
# obj5| p1 p2 p3 p4 p5 ... p784
# obj6| p1 p2 p3 p4 p5 ... p784
# obj7| p1 p2 p3 p4 p5 ... p784
# obj8| p1 p2 p3 p4 p5 ... p784
#   .
#   .
#   .
# obj(n)| p1 p2 p3 p4 p5 ... p784

# It is now
# obj1 obj2 obj3 obj4
#
#  p1   p1   p1   p1
#  p2   p2   p2   p2
#  p3   p3   p3   p3
#   .    .    .    .
#   .    .    .    .
#   .    .    .    .
# p784 p784 p784 p784
datatest = data[0:1000].T

# Ytest is the y axis or all the obj or Objects
# while the Xtest is the x axis or all the p1 ... p784 or pixels
Ytest = datatest[0]
Xtest = datatest[1:n]
Xtest = Xtest / 255.



# Same thing as the above datatest, Ytest, and Xtest.
# but this data is used to train our ai model!

datatrain = data[1000:m].T
Ytrain = datatest[0]
Xtrain = datatest[1:n]
Xtrain = Xtrain / 255.
_, mtrain = Xtrain.shape
def initWnB():

# This generates a rows of weight Ie. 10 rows of 784 columns of weight between -0.5 - 0.5
# How we get the row can column is For example W1 the number of inputs which is denoted by the columns and the numbers of nodes in the hidden layer
# Ie. the number of rows. For example our hidden layer 1 will have 10 nodes but will expect 784 inputs each belonging to a pixel value. Thus generating
# 10 columns of 784, 1 for each node. Bias (b1,b2) is a static number that is just added onto the end result so we only need 1 value instead of 784 
# equation is basically 
                            #(p1*W1[1] + p2*W1[2] ... + p784*W1[784]) + b1

# W2 and b2 is the same thing for the output layer where this time instead of having 784 inputs its only 10 from the previous inputs.
    W1 = np.random.rand(10,784) - 0.5
    b1 = np.random.rand(10,1) - 0.5
    W2 = np.random.rand(10,10) - 0.5
    b2 = np.random.rand(10,1) - 0.5

    return W1,b1,W2,b2

# This is the step function taught in class It takes the maximum value
# For example if con < 0 it will return 0 since 0 is the maximum
# And vice versa Returns con since con is the maximum
# con is short for connection

def function(con):
    return np.maximum(0,con)

# This is the derivative of the function Above this is used for back propagation
# Back Prop is how we update our weight and biases and this is a boolean expression
# Boolean converted to int is True = 1 False = 0
def derFunc(con):
    return con > 0

# This is another function taught in class that i named confidence
# Rather than returning 1 or 0 like the step function, it will return a percentage
# This percentage represent how confident the AI or the algorithm is in it's guess
# Its official name is SoftMax function
def confidence(con):
    sum = np.exp(con) / (np.sum(np.exp(con)))
    return sum



# Foward propogation Connection1 which is the first connection is made
# from the dotproduct of input. Connection one would be passed onto layer1
# using the Step function.
# same thing happens for connection2 where instead of taking the dot product of
# input. But from layer1. Why? IDK
# Just kidding its beacuse for the first connection input serves as... well the input for that connection
# and we then take the result from layer1 and use that as the inputs for the next connection calculation
# which would give us a confidence value for OutLayer2 or OUTPUT
def fProp(W1,b1,W2,b2,input):
    connection1 = W1.dot(input)
    layer1 = function(connection1)
    connection2 = W2.dot(layer1)
    outlayer2 = confidence(connection2)

    return connection1, layer1, connection2, outlayer2

def returnLabel(Y):
    label = np.zeros((Y.size, Y.max()+1))
    label[np.arange(Y.size), Y] = 1
    label = label.T
    return label


def bProp(con1 , l1, con2, out, W2, input, label):
    labels = returnLabel(label)
    dZ2 = out - labels
    dW2 = 1 / m * dZ2.dot(l1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * derFunc(con1)
    dW1 = 1 / m * dZ2.dot(input.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1 , db1, dW2, db2

def newParams(W1, b1, W2, b2, dW1, db1 ,dW2, db2, tRate):
    W1 = W1 - tRate * dW1
    b1 = b1 - tRate * db1
    W2 = W2 - tRate * dW2
    b2 = b2 - tRate * db2
    return W1, b1, W2, b2
def Pred(out):
    return np.argmax(out, 0)

def accur(predictions, label):
    print(predictions, label)
    return np.sum(predictions == label) / label.size

def gDescent(input, label, iter, tRate):
    W1, b1, W2, b2 = initWnB()
    accuracies = []
    for i in range(iter):
        c1, l1, c2, out = fProp(W1, b1, W2, b2, input)
        dW1, db1, dW2, db2 = bProp(c1, l1, c2, out, W2, input, label)
        W1, b1, W2, b2 = newParams(W1, b1, W2, b2, dW1, db1, dW2, db2, tRate)
        if i % 50 == 0:
            accuracy = accur(Pred(out), label)
            accuracies.append(accuracy)
            print("Iter: ", i)
            print("Acc:" , accuracy)
    return accuracies



iterations = 500
learning_rate = 0.01
accuracies = gDescent(Xtrain, Ytrain, iterations, learning_rate)
plt.plot(range(0, iterations, iterations // len(accuracies)), accuracies)
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.title('Gradient Descent Iterations vs Accuracy')
plt.show()

# W1, b1, W2, b2 = gDescent(Xtrain, Ytrain, 500, 0.1)