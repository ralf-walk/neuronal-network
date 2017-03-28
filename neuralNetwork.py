import numpy
from scipy.special import expit
import matplotlib.pyplot

class neuralNetwork:

    # init the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate

        # setting the weights
        self.weights_input_hidden = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.weights_output_hidden = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        pass

    def query(self, input, number):
        hidden_input = numpy.dot(self.weights_input_hidden, input)
        hidden_output = expit(hidden_input)

        final_input = numpy.dot(self.weights_output_hidden, hidden_output)
        final_output = expit(final_input)

        match = numpy.argmax(final_output) + 1 == number
        return match

    def train(self, input, target):

        # X = W * I
        hidden_input = numpy.dot(self.weights_input_hidden, input)
        hidden_output = expit(hidden_input)

        final_inputs = numpy.dot(self.weights_output_hidden, hidden_output)
        final_output = expit(final_inputs)

        output_error = target - final_output
        hidden_error = numpy.dot(self.weights_output_hidden.T, output_error)

        delta_woh = self.lr * numpy.dot(( output_error * final_output * (1.0 -final_output)), numpy.transpose(hidden_output))
        self.weights_output_hidden += delta_woh
        
        delta_wih = self.lr * numpy.dot(( hidden_error * hidden_output * (1.0 -hidden_output)), numpy.transpose(input))
        self.weights_input_hidden += delta_wih
        pass
    
    def mnist_train(self):

        with open('mnist_dataset/mnist_train_100.csv', 'r') as mnist_train_file:
            for mnist_train_line in mnist_train_file:
                all_image_values = mnist_train_line.split(',')

                mnist_line_number = int(all_image_values[0:1][0])
                mnist_line_image = numpy.asfarray(all_image_values[1:]).reshape((784,1))

                scaled_input = self.contructScaledInput(mnist_line_image)
                scaled_output = self.contructScaledTarget(mnist_line_number)

                self.train(scaled_input, scaled_output)
        pass

    def mnist_test(self):
        right = 0
        wrong = 0

        with open('mnist_dataset/mnist_test_10.csv', 'r') as mnist_test_file:
            for mnist_test_line in mnist_test_file:
                all_image_values = mnist_test_line.split(',')

                mnist_line_number = int(all_image_values[0:1][0])
                mnist_line_image = numpy.asfarray(all_image_values[1:]).reshape((784,1))

                scaled_input = self.contructScaledInput(mnist_line_image)
 
                match = self.query(scaled_input, mnist_line_number)

                if match:
                    right += 1
                else:
                    wrong += 1

        print "Right " + str(right) + " Wrong " + str(wrong)
        pass

    def mnist(self):
        self.mnist_train()
        self.mnist_test()
        pass

    def contructScaledInput(self, mnist_line_image):

        # so 0.01 to 1.00 instead of 0 to 255
        return numpy.asfarray(mnist_line_image) / 255.0 * 0.99 + 0.01

    def contructScaledTarget(self, number):
        target = numpy.zeros(10) + 0.01
        target[number - 1] = 0.99
        return target.reshape((10,1))
    
    def showImage(self, image_string_from_mnist):
        all_image_values = image_string_from_mnist.split(',')
        image_array = numpy.asfarray(all_image_values[1:]).reshape((28,28))
        matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation='None')
        matplotlib.pyplot.savefig("number_" + all_image_values[0:1][0] + ".png")
        pass

    
n = neuralNetwork(784, 100, 10, 0.3)
n.mnist()