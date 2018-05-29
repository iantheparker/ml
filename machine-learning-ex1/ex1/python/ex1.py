import numpy as np
# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def feature_normalize(X):
    X_norm = X
    mu = np.zeros((1, X.shape[1]))
    sigma = np.zeros((1, X.shape[1]))
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = np.subtract(X, mu) / sigma
    return X_norm, mu, sigma


def gradient_descent_multi(X, y, theta, alpha, num_iters):
    m = y.size
    J_history = np.zeros((num_iters, 1))

    for i in range(0, num_iters):
        h = X.dot(theta)
        thetaNew = theta - (alpha * 1 / m * ((h - y).T.dot(X)).T)
        theta = thetaNew

        # Save the cost J in every iteration
        J_history[i] = compute_cost_multi(X, y, theta)
    return theta, J_history


def compute_cost_multi(X, y, theta):
    m = y.size
    J = 0
    h = X.dot(theta)
    J = 1 / (2 * m) * (h - y).T.dot(h - y)
    return J


def normal_equation(X, y):
    theta = np.zeros((X.shape[1], 1))
    theta = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)
    return theta


data = np.loadtxt("../ex1data2.txt", delimiter=',')
X = data[:, :-1]
y = data[:, -1:]
m = y.size

# % Print out some data points
print('First 10 examples from the dataset: \n')
print(f'x = {X[:10, :]}, \n y = {y[:10]}')

# % Scale features and set them to zero mean
print('Normalizing Features ...\n')

X, mu, sigma = feature_normalize(X)

# % Add intercept term to X
X = np.append(X, np.ones((m, 1)), axis=1)


# % % == == == == == == == == Part 2: Gradient Descent == == == == == == == ==
print('Running gradient descent ...\n')

# % Choose some alpha value
alpha = 0.001
num_iters = 40000

# % Init Theta and Run Gradient Descent
theta = np.zeros((X.shape[1], 1))

theta, J_history = gradient_descent_multi(X, y, theta, alpha, num_iters)

# plot the training loss and accuracy
print("[INFO] about to plot...")
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, num_iters), J_history, label="error")
plt.title("Training Loss on Gradient Descent")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig('plot.png')

# % Display gradient descent's result
print('Theta computed from gradient descent: \n')
print(f' {theta} \n')
print('\n')

# % Estimate the price of a 1650 sq - ft, 3 br house
sample_house_X = np.array([1650, 3])
s = (sample_house_X - mu) / sigma
s = np.append(s, np.ones((1, 1)))
price = s.dot(theta)[0]

print(
    f'Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):\n ${price}\n')

# % % == == == == == == == == Part 3: Normal Equations == == == == == == == ==

print('Solving with normal equations...\n')

data = np.loadtxt("../ex1data2.txt", delimiter=',')
X = data[:, :-1]
y = data[:, -1:]
m = y.size

# % Add intercept term to X
X = np.append(X, np.ones((m, 1)), axis=1)

theta = normal_equation(X, y)

# % Display normal equation's result
print('Theta computed from the normal equations: \n')
print(f' {theta} \n')
print('\n')

# % Estimate the price of a 1650 sq - ft, 3 br house
s = np.array([1650, 3, 1])
price = s.dot(theta)[0]

print(
    f'Predicted price of a 1650 sq-ft, 3 br house (using normal equation):\n ${price}\n')
