import numpy as np

# activation function (sigmoid)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def compute_input_to_hidden_weight(tk, ak, wkj, aj, ai):
    return (-((tk - ak) * (ak) * (1 - ak) * (wkj))) * ((aj) * (1 - aj) * (ai))

def compute_hidden_to_output_weight(tk, ak, aj):
    return -(tk - ak) * (ak) * (1 - ak) * aj

def calcule_total_error_over_weight(weight):
    if weight == w1:
        x = compute_input_to_hidden_weight(o1, outo1, w5, outh1, i1)
        y = compute_input_to_hidden_weight(o2, outo2, w7, outh1, i1)
        return x+y
    elif weight == w2:
        x = compute_input_to_hidden_weight(o1, outo1, w5, outh1, i2)
        y = compute_input_to_hidden_weight(o2, outo2, w7, outh1, i2)
        return x+y
    elif weight == w3:
        x = compute_input_to_hidden_weight(o1, outo1, w6, outh2, i1)
        y = compute_input_to_hidden_weight(o2, outo2, w8, outh2, i1)
        return x+y
    elif weight == w4:
        x = compute_input_to_hidden_weight(o1, outo1, w6, outh2, i2)
        y = compute_input_to_hidden_weight(o2, outo2, w8, outh2, i2)
        return x+y
    elif(weight == w5):
        return compute_hidden_to_output_weight(o1, outo1, outh1)
    elif(weight == w6):
        # tk, ak, aj
        return compute_hidden_to_output_weight(o1, outo1, outh2)
    elif(weight == w7):
        # tk, ak, aj
        return compute_hidden_to_output_weight(o2, outo2, outh1)
    elif(weight == w8):
        # tk, ak, aj
        return compute_hidden_to_output_weight(o2, outo2, outh2)  

# weights from input layer to hidden layer
w1 = 0.149780716
w2 = 0.19956143
w3 = 0.24975114
w4 = 0.29950229

# weights from hidden layer to output layer
w5 = 0.35891648
w6 = 0.408666186
w7 = 0.511301270
w8 = 0.561370121

# targets and biases
i1 = 0.05
i2 = 0.10

o1 = 0.01
o2 = 0.99

b1 = 0.35
b2 = 0.60

# hidden layer
neth1 = (i1 * w1 + i2 * w2 + b1)
outh1 = sigmoid(neth1)

neth2 = (i2 * w4 + i2 * w3 + b1)
outh2 = sigmoid(neth2)

# output layer
neto1 = (outh1 * w5 + outh2 * w6 + b2)
outo1 = sigmoid(neto1)

neto2 = (outh2 * w8 + outh1 * w7 + b2)
outo2 = sigmoid(neto2)

# weight you want to compute new value

weight_string = input("\n\tEnter the weight you want to change: ")

weight = globals().get(weight_string)

derivative = calcule_total_error_over_weight(weight)

print(f"\n\tTotal Error is : {derivative}")

# new weight value
learningRate = 0.500169758515210 # default

new_weight = weight - (learningRate * derivative)

print(f"\n\tPrevious weight was: {weight}")

print(f"\n\tNew weight is: {new_weight}")


