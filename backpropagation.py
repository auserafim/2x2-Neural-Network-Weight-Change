import numpy as np

# activation function (sigmoid)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def compute_input_to_hidden_weight(tk, ak, wkj, aj, ai):
    return (-((tk - ak) * (ak) * (1 - ak) * (wkj))) * ((aj) * (1 - aj) * (ai))

def compute_hidden_to_output_weight(tk, ak, aj):
    return -(tk - ak) * (ak) * (1 - ak) * aj

def calcule_new_weight(weight):
    if weight == w1:
        x = compute_input_to_hidden_weight(o1, outo1, w5, outh1, 0.05)
        y = compute_input_to_hidden_weight(o2, outo2, w7, outh1, 0.05)
        return x+y
    elif weight == w2:
        x = compute_input_to_hidden_weight(o1, outo1, w5, outh1, 0.1)
        y = compute_input_to_hidden_weight(o2, outo2, w7, outh1, 0.1)
        return x+y
    elif weight == w3:
        x = compute_input_to_hidden_weight(o1, outo1, w6, outh2, 0.5)
        y = compute_input_to_hidden_weight(o2, outo2, w8, outh2, 0.5)
        return x+y
    elif weight == w4:
        x = compute_input_to_hidden_weight(o1, outo1, w6, outh2, 0.1)
        y = compute_input_to_hidden_weight(o2, outo2, w8, outh2, 0.1)
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
w1 = 0.15
w2 = 0.2
w3 = 0.25
w4 = 0.3

# weights from hidden layer to output layer
w5 = 0.4
w6 = 0.45
w7 = 0.5
w8 = 0.55

# targets and biases
i1 = 0.05
i2 = 0.10
o1 = 0.01
o2 = 0.99
b1 = 0.35
b2 = 0.60

# hidden layger
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
weight = w5
derivative = calcule_new_weight(weight)
print(f"O valor do erroTotal eh: {derivative:.10f}")

# new weight value
learningRate = 0.5 # default
new_weight = weight - (learningRate * derivative)
print(f"O valor antigo do peso eh: {weight:.10f}")
print(f"O valor do novo peso eh: {new_weight:.10f}")


