import numpy as np

def sigmoid(X):
    return 1 / (1 + np.exp(X))

def compute_cost(X, y, theta):
    
    m = len(y)
    
    h = sigmoid(X @ theta)
    J = 1 / (2 * m) * error.T @ error
    
    grad = X.T @ error
    
    return J.item(), grad

def gradient_descent(X, y, theta, alpha, num_iter, regularization = 0):
    m = len(y)
    J_history = []
    
    for i in range(num_iter):
        J, grad = compute_cost(X, y, theta)
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
        
        J_mini, grad = compute_cost(X_mini, y_mini, theta)
        J += J_mini
            
        theta = theta * (1 - alpha * (regularization / m)) - (alpha / m) * grad
        J_history.append(J)
        
    
    return theta, J_history

def predict(data, theta, mu = None, sigma = None):
    if mu is None and sigma is None:
        return (data @ theta).item()
    else:
        data = data.astype(float)
        data[:,1:] = (data[:,1:] - mu[1:]) / sigma[1:]
        return (data @ theta).item()

