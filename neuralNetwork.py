import numpy
from scipy.special import expit

class neuralNetwork:

    # init the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate

        # setting the weights
        self.weights_input_hidden = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.weights_output_hidden = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        pass

    def query(self, inputs_list):
        inputs = numpy.array(inputs_list)

        hidden_inputs = numpy.dot(inputs, self.weights_input_hidden)
        hidden_outputs = expit(hidden_inputs)

        final_inputs = numpy.dot(hidden_outputs, self.weights_output_hidden)
        final_outputs = expit(hidden_inputs)
        return final_outputs

    def train(self, input_lists, target_lists):
        targets = numpy.array(target_lists)
        inputs = numpy.array(inputs_list)

        hidden_inputs = numpy.dot(inputs, self.weights_input_hidden)
        hidden_outputs = expit(hidden_inputs)

        final_inputs = numpy.dot(hidden_outputs, self.weights_output_hidden)
        final_outputs = expit(hidden_inputs)

        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.weights_output_hidden.T, output_errors)

        delta_woh = self.lr * numpy.dot(( output_errors * final_outputs * (1.0 -final_outputs)), numpy.transpose(hidden_outputs)
        self.weights_output_hidden += delta_woh
        
        delta_wih = self.lr * numpy.dot(( hidden_errors * hidden_outputs * (1.0 -hidden_outputs)), numpy.transpose(inputs)
        self.weights_input_hidden += delta_wih
        pass
    
n = neuralNetwork(3, 3, 3, 0.3)
print n.query([1.0, 0.5, -1.5])