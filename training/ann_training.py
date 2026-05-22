from model import load_data, train_val_split, NeuralNetwork

X, y = load_data('../data/*.npz')
X_train, X_val, y_train, y_val = train_val_split(X, y, val_ratio=0.2)

print("Train samples: %d  Val samples: %d" % (len(X_train), len(X_val)))

nn = NeuralNetwork()
nn.create([50400, 81, 9, 3])
nn.train(X_train, y_train, X_val, y_val, max_epochs=50, patience=5)
nn.save('../models/ann.xml')

val_acc = nn.evaluate(X_val, y_val)
print('Final val accuracy: %.2f%%' % (val_acc * 100))
