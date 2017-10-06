
# coding: utf-8

# In[23]:


# import library of functions
from numpy import exp, array, random, dot, mean, abs


# In[24]:


input_data = array([[0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]])
output_labels = array([[0],
            [1],
            [1],
            [0]])

print(input_labels)
print(output_labels)


# In[25]:


# the activation function
def __sigmoid(x):
        return 1 / (1 + exp(-x))
    
# the sigmoid derivative
# shows how much to change the weights by
def __sigmoid_derivative(x):
        return x * (1 - x)


# In[26]:


# the weights
synaptic_weight_0 = 2 * random.random((3, 4)) - 1
synaptic_weight_1 = 2 * random.random((4, 1)) - 1

print(synaptic_weight_0)
print(synaptic_weight_1)


# In[27]:


# iterate over the data 
iterations = 60000
for j in range(iterations):

	# Forward propagate through layers 0, 1, and 2
    layer0 = input_data
    layer1 = __sigmoid(dot(layer0, synaptic_weight_0))
    layer2 = __sigmoid(dot(layer1, synaptic_weight_1))

    #calculate error for layer 2
    layer2_error = output_labels - layer2
    
    if (j% 10000) == 0:
        print ("Error: " + str(mean(abs(layer2_error))))
        
    #Use it to compute the gradient
    layer2_adjustment = layer2_error * __sigmoid_derivative(layer2)

    #calculate error for layer 1
    layer1_error = layer2_adjustment.dot(synaptic_weight_1.T)
    
    #Use it to compute its gradient
    layer1_adjustment = layer1_error * __sigmoid_derivative(layer1)
    
    #update the weights using the gradients
    synaptic_weight_1 += layer1.T.dot(layer2_adjustment)
    synaptic_weight_0 += layer0.T.dot(layer1_adjustment)


# In[28]:


# test the neural network
print(__sigmoid(dot(array([0, 1, 1]), synaptic_weight_0)))

