import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.preprocessing import StandardScaler  
def plot_curve(x, y, y_pred, filename='picture_sin.png'):  
    plt.figure(figsize=(8, 6))  
    plt.plot(x, y, label='sin(x)', color='blue', linewidth=2)  
    plt.plot(x, y_pred, label='Fitted curve', color='red', linestyle='--')  
    plt.xlabel('x')  
    plt.ylabel('y')  
    plt.legend()  
    plt.grid(True)  
    plt.show()  
    plt.savefig(filename)  
class Our_Polynomial_Regression(object):  
    def __init__(self, degree=5):  
        self.degree = degree  
        self.w = None  
    def fit(self, X, y, learning_rate=0.01, epochs=10000, verbose=False):  
        n = len(X)  
        self.w = np.random.randn(self.degree + 1)  # Include bias term  
        loss_history = []  
        for i in range(epochs):  
            y_pred = np.dot(X, self.w)  
            loss = np.mean((y - y_pred) ** 2)  
            loss_history.append(loss)  
            gradient = -2 / n * np.dot(X.T, (y - y_pred))  
            self.w -= learning_rate * gradient  
            if verbose and i % (epochs // 10) == 0:  
                print(f"Epoch {i}, Loss: {loss}")  
        return loss_history  
    def predict(self, X):  
        return np.dot(X, self.w)  
def main():  
    # Data loaded:  
    x = np.linspace(0, 2 * np.pi, 100)  
    y = np.sin(x)  
    # Polynomial feature expansion and standardization  
    X = np.vstack([x ** i for i in range(6)]).T  
    scaler = StandardScaler()  
    X_scaled = scaler.fit_transform(X)  
    # Our regression model  
    model = Our_Polynomial_Regression(degree=5)  
    loss_history = model.fit(X_scaled, y, verbose=True)  
    # Predictions  
    y_pred = model.predict(X_scaled)  
    # Plotting  
    plot_curve(x, y, y_pred)  
if __name__ == "__main__":  
    main()
