import numpy as np
from layer import Layer
'''
Activation layer:
    used for activation function at each layer

    This inherits the layer class
'''

class Activation(Layer):
    '''
    Initializes activation function and its derivative, which are 
    specified by specific activation functions.
    '''

    def __init__(self, activation, d_activation):
        self.activation = activation
        self.d_activation = d_activation

    '''
    forward propagation is something that this activation class inherits from layer class.
    '''
    def forward_propagate(self, input):
        self.input = input
        return self.activation(input)
    
    '''
    backward propagation is also inherited by this class from parent layer class
    '''
    def backward_propagate(self, grad, learning_rate):
        # grad here represents the gradient of the output produced by relu
        # and we now have to back prop so we need to calculate the derivative of input.
        gradient_of_input = np.multiply(grad, self.d_activation(self.input))
        return gradient_of_input


# in our nn we have used relu and softmax as the activation functions

class ReLU(Activation):

    '''
    relu - rectified linear activation funct, inherits from Activation Class which compute
    piecewise function f(x) = max{0,x}
    '''

    def __init__(self, input):
        self.input = input
        self.relu = lambda x : np.maximum(0, x)
        self.d_relu = lambda x : (x > 0)
        Activation.__init__(self, self.relu, self.d_relu)

'''
lambda allows us to write a func in short.
super lets the child Class call something from the parent Class

'''


def relu():
    return ReLU(0)
    


