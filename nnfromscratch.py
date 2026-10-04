import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from layer import Dense, Softmax, softmax
from activationfunctions import ReLU, relu
from error import d_mse, mse

# control the randomness and produce the same sequence everytime
seed = 42
np.random.seed(seed)

# data-processing
print("Loading MNIST dataset...")
X, y = fetch_openml(
    "mnist_784",
    version=1,
    as_frame=True,   
    return_X_y=True
)

# purpose of doing as_frame=true- we get data in pandas dataframe format, which is easier to manipulate and analyze.
# for importing data normally dataframe is used.
# good habit is to check the shape of the data we are getting.
print(X.shape)
# shape - (70000, 784) - 70000 images of 28x28 pixels flattened to 784 features
# X.shape[0]=70000
# X.shape[1]=784
print(y.shape)
# shape - (70000,) - 70000 labels for the images - an array of 70000 elements, each representing the digit in the corresponding image.

# now that we know the shape of the data we convert it to numpy arrays as they are better for computation and mathmatical operations.
X = np.asarray(X) 
# convert it to 784X1 column vector- so that we have all pixel arrays of an image column wise now, as it is easy to feed to NN layers
X = X.reshape(X.shape[0],X.shape[1], 1)/255

# similarly we do for y,
y = np.asarray(y, dtype=int)

# uptil now I have done no changes to the data- we have simply converted it to numpy arrays.
# Feature scaling makes sense here as each image pixel can go from 0-255 and we had studied that if there is a higher range in input
# values we must normalize it to make it closer to 0. In this case if we divide by 255, we will get from 0 to 1. 
# but I wont do it initially so that we can compare the difference of using and not.

# visualizing the data:
fig, axes = plt.subplots(5, 8, figsize= (12,12))
# figsize represents size of entire figure which will be shown
# axes individual plots

for ax, image, label in zip(axes.flat, X[:40], y[:40]):
    ax.axis("off")
    ax.imshow(image.reshape(28,28), cmap="gray")
    ax.set_title(str(label), y=-0.35)

fig.suptitle("Examples of MNIST Handwritten Digits", y=0.0, fontsize=20)
# plt.tight_layout()
# plt.show()

# convert lable to one-hot encoded vectors-> this helps us later for easy comparison when our nn will return.
def oneHot(y):
    oneHotVectorOfy = np.zeros((y.size, y.max() + 1))
    # np.zeros((4,8))-> 
    # [
    #     [0,0,0,0,0,0,0,0],
    #     [0,0,0,0,0,0,0,0],
    #     [0,0,0,0,0,0,0,0],
    #     [0,0,0,0,0,0,0,0]
    # ]
    oneHotVectorOfy[np.arange(y.size),y]=1
    # row 0-> column would be y's value-> put 1
    # eg - y= [3,1,7,2]
    # row 0 → column 3 → put 1
    # row 1 → column 1 → put 1
    # row 2 → column 7 → put 1
    # row 3 → column 2 → put 1
    return oneHotVectorOfy

y= oneHot(y)
# y will get converted to 2-d one hot vectors

# now we will split our x,y into training and testing set
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size= 5000,
    random_state=42
)
# by keeping random state it will random select 5000 test examples and not the starting 5000 ones
# we can also use stratification on our examples which can help in reducing the sampling bias - i can read
# about more but here I won't do it because there seems no use of it here.


# all the initial work done with the input and label: Now I will built NN
# I will start at the end where nn is called and then build each function required for that.

print("Starting training...")

class Network():
    def __init__(self,layers):
        self.layers= layers
    
    '''make predictions by propagating
       through each layer forward
    '''
    def predict(self, input):
        output = input

        for layer in self.layers:
           output = layer.forward_propagate(output)

        return output
    
    def train(self, X, y, cost, d_cost, epochs, batch_size, learning_rate, validation_split=0.2, verbose = True):
        
        history = { 'Accuracy': [] , 'Validation Accuracy': [] }
        size = X.shape[0]

        # split into training and validation set 
        split = int(validation_split * size)

        X_train, X_val = X[split:], X[:split]
        y_train, y_val = y[split:], y[:split]

        for epoch in range(epochs):
            count = 0
            val_count = 0

            for x, y in zip(X_train, y_train):
                output = self.predict(x)
                # this will be the output of the softmax layer for our code. This is therefore
                # the probability array containing 10 values
                # next goal would be to compare ths with y to find if it is accurate prediction or we need to calculate the errors for this.
                grad = d_mse(output, y.reshape(-1, 1))
                for layer in reversed(self.layers):
                    grad = layer.backward_propagate(grad, learning_rate)

                # here we will compare the index values of predicted y and true y to see if both match or not- our prediction is correct or not
                if np.argmax(output) == np.argmax(y):
                    count+=1
                    # we are calculating our accuracy 
                
            accuracy = count/(size-split)
            # (size- split ) gives us the no. of values in training. so simply accuracy is how many times we predicted correctly as compared to total times it ran.

            for x, y in zip( X_val, y_val ):
                output = self.predict(x)

                if np.argmax(output) == np.argmax(y):
                    val_count+=1

            val_accuracy = val_count/split

            if verbose:
                print('Epoch: %d/%d' % (epoch+1, epochs))
                print(f'Training Accuracy: {accuracy:.2%} - Validation Accuracy: {val_accuracy:.2%}')

            # Stores accuracy at each epoch
            history['Accuracy'].append(accuracy)
            history['Validation Accuracy'].append(val_accuracy)

        return history



network = Network([
    Dense(784,40),
    relu(),
    Dense(40,10),
    softmax()
])


history = network.train(X_train, y_train, mse, d_mse, epochs=50, batch_size=32, learning_rate=0.1, validation_split=0.2, verbose=True)
