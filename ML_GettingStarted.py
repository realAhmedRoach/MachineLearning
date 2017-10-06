
# coding: utf-8

# In[ ]:


# import library of functions
from numpy import exp, array, random, dot


# In[ ]:


# the neural network class
class NeuralNetwork():
    def __init__(self):
        # seed the random number generator so the same
        # numbers are generated every time
        random.seed(1)
        
        # initialize the weights as a 3 x 1 matrix
        self.synaptic_weights = 2 * random.random((6, 4)) - 1 # value range from -1 to 1
        
    # the activation function
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))
    
    # the sigmoid derivative
    # shows how much to change the weights by
    def __sigmoid_derivative(self, x):
        return x * (1 - x)
    
    # train the neural network by changing the weights each time
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iterations in range(number_of_training_iterations):
            # pass the training set into our neural network
            output = self.think(training_set_inputs)
            
            # calculate the error
            error = training_set_outputs - output
            
            # multply the error by the input and by the gradient of the sigmoid
            # the less confident weights are adjusted more
            # this means inputs, which are 0, do not adjust the weights
            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))
            
            # adjust the weights with the adjustment
            self.synaptic_weights += adjustment
            
    def think(self, inputs):
        return self.__sigmoid(dot(inputs, self.synaptic_weights))
    




# In[8]:


# declare the possible inputs and outputs
possible_inputs = [[1, 0, 0, 1, 1, 0], [1, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]]
possible_outputs = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

if __name__ == "__main__":
    
    # initialize the neural network
    neural_network = NeuralNetwork()
    
    print ("Random starting synaptic weights: ")
    print (neural_network.synaptic_weights)
    
    # the training set has 3 inputs and 1 output
    training_set_inputs = array(possible_inputs)
    training_set_outputs = array(possible_outputs).T
    
    # Train the neural network with our set
    # Do it 10,100 times
    neural_network.train(training_set_inputs, training_set_outputs, 100000)
    
    print ("New synaptic weights after training: ")
    print (neural_network.synaptic_weights)
    
    # Test the neural network with a new situation
    print ("Considering new situation [0.75, 0.1, 0.1, .86, .93, .14] -> ")
    print (neural_network.think(array([0.75, 0.1, 0.1, .86, .93, .14])))
    
    # This fails on a one-layer neural network...
    print ("Considering new situation [0.854, 0.761, 0.000, 0.736, 0.798, 0.762]] -> ")
    print (neural_network.think(array([0.854, 0.761, 0.000, 0.736, 0.798, 0.762])))
 

