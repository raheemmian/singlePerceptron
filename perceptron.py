import numpy
import time
def NN(m1, m2, w1, w2, b):
    z = m1*w1 + m2*w2 + b
    return step(z)
def sigmoid(x):
    return 1 /(1 + numpy.exp(-x))
def step(x):
    #for the and gate if x = 0 outputs 1 then the algorithm goes on infinitely
    # but x = 0 outputs 0 then it works fine, and it works good for both or & and gates
    return numpy.heaviside(x, 0)

def perceptron(inp,out,nn):
    w1 = nn[0]
    w2 = nn[1]
    b = nn[2]
    isWeightsCorrect = False
    while isWeightsCorrect == False:
        for i in range(4):
            m1 = inp[i][0]
            m2 = inp[i][1]
            t = out[i]
            y = NN(m1, m2, w1, w2, b)
            print('i : ', i,'target: ', t, 'output : ', y)
            if y != t:
                w1,w2 = update(m1,m2,w1,w2,t,y)
                print('new weights: w1: ',w1, ' w2: ',w2)
                break
            elif i == 3:
                isWeightsCorrect = True  
    return w1,w2

def update(m1,m2,w1,w2,t,y):
    w1 = w1 + (t-y)*m1
    w2 = w2 + (t-y)*m2
    return w1, w2

w1 = -1
w2 = -1
b = -1
nn = [w1,w2,b]

TT = [[0,0], [0,1], [1,0], [1,1]]
orGateTT = [0,1,1,1]
andGateTT = [0,0,0,1]
w1,w2 = perceptron(TT,orGateTT,nn)

print('final weights: w1: ', w1, " w2: ", w2, ' bias: ', b)
