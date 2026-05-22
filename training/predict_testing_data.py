from model import load_data, NeuralNetwork

X, y = load_data('../data/*.npz')

nn = NeuralNetwork()
nn.load('../models/ann.xml')

accuracy = nn.evaluate(X, y)
print('Test accuracy: %.2f%%' % (accuracy * 100))
