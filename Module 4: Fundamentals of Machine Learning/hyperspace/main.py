import numpy as np

# computing the predictions
def predict(X, w):
    return np.matmul(X, w)

# calculating the loss
def loss(X, Y, w):
    return np.average((predict(X, w) - Y) ** 2)

# evaluating the gradient
def gradient(X, Y, w):
    return 2 * np.matmul(X.T, (predict(X, w) - Y)) / X.shape[0]

# performing the training phase for our classifier
def train(X, Y, iterations, lr):
    w = np.zeros((X.shape[1], 1))
    for i in range(iterations):
        print("Iteration %4d => Loss: %.20f" % (i, loss(X, Y, w)))
        w -= gradient(X, Y, w) * lr
    return w

# loading the data first and then training the classifier for 50,000 iteration
x1, x2, x3, y = np.loadtxt("pizza_3_vars.txt", skiprows=1, unpack=True)
X = np.column_stack((x1, x2, x3))
Y = y.reshape(-1, 1)
w = train(X, Y, iterations=50000, lr=0.001)
