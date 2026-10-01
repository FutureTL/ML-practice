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
