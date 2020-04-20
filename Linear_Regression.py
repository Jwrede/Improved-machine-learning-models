import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.datasets import make_regression

def feature_normalize(X):
    X_norm = X.copy()
    mu = np.mean(X, axis = 0)
    sigma = np.std(X, axis = 0)
    
    X_norm[:,1:] = (X[:,1:] - mu[1:]) / sigma[1:]
    
    return X_norm, mu, sigma

def compute_cost(X, y, theta, regularization = 0):
    
    m = len(y)
    
    error = X @ theta - y
    J = (1 / m) * (error.T @ error) + regularization / (2 * m) * sum(theta**2)
    
    grad = X.T @ error
    
    return J.item(), grad

def gradient_descent(X, y, theta, alpha, num_iter, regularization = 0):
    m = len(y)
    J_history = []
    for i in range(num_iter):
        J, grad = compute_cost(X, y, theta, regularization)
        J_history.append(J)
        theta = theta * (1 - alpha * (regularization / m)) - (alpha / m) * grad
    return theta, J_history

def mini_batch_gradient_descent(X, y, theta, alpha, num_iter, batch, batch_size = 10, regularization = 0):
    m = len(y)
    J_history = []
    data = np.c_[X,y]
    
    for i in range(num_iter):
        
        J = 0.0
        j = np.random.randint(0, m)
        
        data_mini = data[j:j+batch_size]
        X_mini, y_mini = data_mini[:,:-1], data_mini[:,-1].reshape(-1,1)
        
        J_mini, grad = compute_cost(X_mini, y_mini, theta, regularization)
        J += J_mini
            
        theta = theta * (1 - alpha * (regularization / m)) - (alpha / m) * grad
        J_history.append(J)
        
    
    return theta, J_history

def normal_equation(X, y):
    return np.linalg.inv(X.T @ X) @ (X.T @ y)

def predict(data, theta, mu = None, sigma = None):
    if mu is None and sigma is None:
        return (data @ theta).item()
    else:
        data = data.astype(float)
        data[:,1:] = (data[:,1:] - mu[1:]) / sigma[1:]
        return (data @ theta).item()


X, y = make_regression(10000, 100)
#X = data.iloc[:, :-1].to_numpy()
X = np.c_[np.ones((len(X), 1)), X]
y = np.array([y]).T
init_theta = np.zeros((len(X[0]),1))

X_norm, mu, sigma = feature_normalize(X)

num_iter = 10000
pred = np.array([X[999]])

start = time.time()
theta, J_history_2 = mini_batch_gradient_descent(X_norm,y,init_theta, 0.01, num_iter, batch = 100, batch_size = 1000, regularization = 4)
end = time.time()
print(end - start)

print(predict(pred, theta, mu = mu, sigma = sigma))

start = time.time()
theta, J_history_3 = gradient_descent(X_norm,y,init_theta, 0.01, num_iter, regularization = 4)
end = time.time()
print(end - start)

print(predict(pred, theta, mu = mu, sigma = sigma))


fig, ax = plt.subplots(2,1)

ax[0].plot(range(num_iter), J_history_2)
ax[1].plot(range(num_iter), J_history_3)

plt.show()
