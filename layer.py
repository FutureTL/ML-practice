import numpy as np

'''
defines the structure of a layer and the properties it 
can have. Each layer as we know takes an input and gives an output
For calculating the input it uses forward propagation function, 
and for updating the weights- it uses backward prop.
'''
class Layer():
    def __init__(self):
        self.input = None
        self.output = None
    
    def forward_propagate(self, input):
        # forward progagation function that is used by each layer
        pass
    
    def backward_propagate(self, grad, learning_rate):
        # similarly each layer also has a backward propagation function to update the weights/parameters
        pass


# our code uses Dense which is an child class of Layer class
# it will inherit all the properties we have defined for a Layer

class Dense(Layer):
    def __init__(self, input_size, output_size):

        # each dense layer will initialize some weights and bias
        # weight is going to be a 2d vector - which is going to depend on 
        # output size and input -> eg- input for 1st layer is 784x1 and we have defined
        # 40 neurons/units for it so output size is 40 for this layer. so weight vector will be 
        #  40 x 784, so that when we take its dot ptoduct with input (784 x1), we get output as (40x1).
        # so essentially from my understanding we reduce the number if image pixel representation to 40 from 784.

        self.weights = np.random.randn(output_size, input_size)
        # 2d vector of output_size x input_size
        self.biases = np.random.randn(output_size, 1)
        # 2nd vector of output_size x 1 column
        

    '''
    define the forward propagation logic for any layer
    '''
    def forward_propagate(self, input):
        self.input = input 
        return np.dot(self.weights, self.input) + self.biases
             #  [40x784 , 784x1]= 40x1  and biases also has 40x1 dimension so overall 40 output coming from 1st layer



    '''
    backward propagation- 
    imagine this is the 2nd dense layer that has 10 units/neurons. Its input is a and output is z. we have already caluculated
    the derivative/grad of this z in the backward propagate function of softmax. we now need to calculate grad of a, weights and bias
    '''
    def backward_propagate(self, grad, learning_rate):
        gradient_of_input = np.dot(self.weights.T, grad)
        # grad_of_a = dot product of transpose of weights and output gradient
        # weights have 10x40 so its transpose has 40x10, and gard of output, 10x1 so size of a is 40x1. 
        '''
        we also have to update the weights- we must update weights and biases only when gradient fpr both have been 
        found. If we update weight before calcualting the gradient for bias, then it get wrong values.
    
        '''
        gradient_of_weights = np.dot(grad, self.input.T)
        gradient_of_bias = grad
        self.weights = self.weights - learning_rate*gradient_of_weights
        self.biases = self.biases - learning_rate*gradient_of_bias
        return gradient_of_input
        # for 2nd dense layer this derivative will be used by relu layer for calculating its derivative values now.
            

            

'''
Softmax function- applied in the last layer of the neural network to convert the output into probabilities

'''
class Softmax(Layer):
    def forward_propagate(self, input):
        # shifting all values to safer values to avoid overflow situation(max - shifting)
        # remember we have array as an input [2,3,4,5], and softmax will now help us convert these to probabilities

        input = input - np.max(input)
        # modified array - [ -3, -2, -1, 0] as np.max(of the array) is 5.
        self.output = np.exp(input)/np.sum(np.exp(input))
        # numerator gives exponential of all index values- [  $e^-3$, $e^-2$, $e^-1$, $e^0$] (all in powers).
        # np.sum will sum up all those exponential values and then we divide them
        # ouput is an array of those probability conversions
        return self.output

    def backward_propagate(self, grad, learning_rate):
        '''
        lets say z is the input to the softmax layer and y_predict was its output.
        we know the predicted values of y, depend on all input values of z for a softmax function
        y1 = (e^z1)/sum(e^z1 + e^z2....e^zn)
        grad of z = jacobian(dot) grad of y
        grad of y is coming as input in this function, so what about jacobian?
        jacobian = diag(y_predict) - y_predic.y_predict.T      ----- y_predcit is y^

        np.identity(n) - gives diagonal of size nxn
        '''
       
        n = np.size(self.output)
        jacobian = np.identity(n)*self.output - self.output*self.output.T
        return np.dot(jacobian, grad)
        # this gives us gradient of z(which is input to softmax)


def softmax():
    return Softmax()
