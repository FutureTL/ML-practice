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
    
    def forward_propagate():
        # forward progagation function that is used by each layer
        pass
    
    def backward_propagte():
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

        '''

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
