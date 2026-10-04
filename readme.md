commands to run this project:
cd neuralnetworksfromscratch
.venv\Scripts\activate

pip install -r requirements.txt

python nnfromscratch.py

https://gombru.github.io/2018/05/23/cross_entropy_loss/

1. Softmax function: It is generally used in the final output layer to convert the models outputs into probablities that sum up to 1. 
- So all individual outputs coming out of this softmax function lie between 0-1 , and together sum up to 1. 
- Why do we use it? Ans: Because the outputs produced by the model might not be easily interpreted by us, but we are better at understanding probabilities. 
 <!-- in depth  -->
 (https://medium.com/@sue_nlp/what-is-the-softmax-function-used-in-deep-learning-illustrated-in-an-easy-to-understand-way-8b937fe13d49)

2. An observation- sometimes we have to make sure that numericals remain under safe limit- maybe we are doing an exponential or something, so going out of bounds is very likely. So we should control that, by applying some max-shifting tricks/ log-sum-exp trick.
- max shifting -> (z -> z- max(z))
- In our code we have used this concept in softmax layer.

3. 1st obervation the accuracy is very less around 10%. I tried learning rate of 0.1, 0.05, 0.3 and none seemed to create any difference. So I am going to try normalizing the input data to see if it creates any diff. 
- So accuracy jumped to around 60% just by introducing normalization. 
- No I will try increasing the epochs. Right now it was just 10. Lets try 30.
- doing this increased accuracy to 94%. with each epcoh we see it improving to i will increase epochs to 50.
- Increasin to 50 epochs didn't have much of any difference. So lets try changing the learning rate to 0.1 and see if it makes any difference.
- Still no difference.So I will try changing the learning rate to 0.01 and see if it makes any difference.
- This reduced the accuracy to just 60%.Obviously bad. sO lets go towards higher than 0.1 to maybe 0.3










