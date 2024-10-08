import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
data = pd.read_csv('boston_house_prices.csv')  
X = data[['RM']].values
y = data['MEDV'].values
X_b = np.c_[np.ones((X.shape[0], 1)), X] 
def compute_cost(X, y, theta):  
    m = len(y)  
    predictions = X.dot(theta)  
    square_errors = np.power(predictions - y, 2)  
    return 1 / (2 * m) * np.sum(square_errors)  
def gradient_descent(X, y, theta, learning_rate=0.01, n_iterations=1000):  
    m = len(y)  
    cost_history = []  
    for iteration in range(n_iterations):  
        predictions = X.dot(theta)  
        errors = predictions - y  
        gradients = X.T.dot(errors) / m  
        theta -= learning_rate * gradients  
        cost = compute_cost(X, y, theta)  
        cost_history.append(cost)  
    return theta, cost_history  
initial_theta = np.zeros(X_b.shape[1])  
theta_final, costs = gradient_descent(X_b, y, initial_theta)  
plt.figure(figsize=(10, 5))  
plt.subplot(1, 2, 1)  
plt.plot(costs)  
plt.xlabel('Iterations')  
plt.ylabel('Cost')  
plt.title('Cost over Iterations')  
plt.subplot(1, 2, 2)  
X_range = np.array([X_b[:, 1].min(), X_b[:, 1].max()]).reshape(-1, 1)  
X_b_range = np.c_[np.ones((X_range.shape[0], 1)), X_range]  
predictions = X_b_range.dot(theta_final)  
plt.scatter(X, y, color='red', label='Actual data')  
plt.plot(X_range, predictions, color='blue', linewidth=3, label='Predicted regression line')  
plt.xlabel('Average number of rooms per dwelling (RM)')  
plt.ylabel('Median value of owner-occupied homes in $1000s (MEDV)')  
plt.title('Boston House Prices')  
plt.legend()  
plt.tight_layout()  
plt.show()
