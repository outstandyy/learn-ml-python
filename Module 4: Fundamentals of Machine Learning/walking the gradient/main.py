import numpy as np

# Basically doing prediction
def predict(X, w, b):
    return X * w + b

# Computing Loss
def loss(X, Y, w, b):
    return np.average((predict(X, w, b) - Y) ** 2)

# calculating gradient
def gradient(X, Y, w, b):
    w_gradient = 2 * np.average(X * (predict(X, w, b) - Y))
    b_gradient = 2 * np.average(predict(X, w, b) - Y)
    return (w_gradient, b_gradient)

# trainaing the classifier
def train(X, Y, iterations, lr):
    w = b = 0
    for i in range(iterations):
        if i%20 == 0 or i==99:
            print("Iteration %4d => Loss: %.10f" % (i, loss(X, Y, w, b)))
        w_gradient, b_gradient = gradient(X, Y, w, b)
        w -= w_gradient * lr
        b -= b_gradient * lr
    return w, b

# Loading the dataset and then training the classifier for 100 iterations
X, Y = np.loadtxt("pizza.txt", skiprows=1, unpack=True)
w, b = train(X, Y, iterations=100, lr=0.005)
print("\nw=%.10f, b=%.10f" % (w, b))
print("Prediction: x=%d => y=%.2f" % (20, predict(20, w, b)))
