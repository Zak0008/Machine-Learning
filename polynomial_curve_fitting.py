import numpy as np
import matplotlib.pyplot as plt

class PolyRegressor:
    def __init__(self, degree=1, regularization=0.0):
        self.degree = degree
        self.regularization = regularization
        self.w = None
    
    def design_matrix(self, x):
        return np.vander(x, self.degree + 1, increasing=True)
    
    def fit(self, x, t, method='direct', **kwargs):
        phi = self.design_matrix(x)
        I = np.eye(phi.shape[1])
        if method == 'direct':
            self.fit_direct(x, t)
        elif method == 'gd':
            self.fit_gd(x, t, **kwargs)
        elif method == 'sgd':
            self.fit_sgd(x, t, **kwargs)
        elif method == 'adam':
            self.fit_adam(x, t, **kwargs)
        else:
            raise NotImplementedError(f"Method '{method}' not implemented")
    
    def fit_direct(self, x, t):
        phi = self.design_matrix(x)
        I = np.eye(Phi.shape[1])
        self.w = np.linalg.inv(Phi.T @ Phi + self.regularization* I) @ Phi.T @ t
    
    def fit_gd(self, x, t, lr=0.01, epochs=1000, use_numerical=False, decay=None, verbose=False):
        phi = self.design_matrix(x)
        N, M = phi.shape
        self.w = np.zeros(M)
        for epoch in range(epochs):
            if decay:
                learning_rate = decay(lr, epoch)
            else: 
                learning_rate = lr
            
            if use_numerical:
                grad = self.numerical_gradiant(Phi, t)
            else:
                error = phi@self.w - t 
                grad = (phi.T @ error + self.regularization * self.w) / N
            
            self.w -= learning_rate*grad

            if verbose and epoch % (epochs // 10) == 0:
                loss = np.mean((phi @ self.w - t) ** 2)
                print(f"[{epoch}] Loss: {loss:.6f}")
    
    def fit_sgd(self, x, t, lr=0.01, epochs=1000, beta=0.9, decay=0.0, verbose=False, **kwargs):
        X = self.design_matrix(x)
        N, D = X.shape
        self.w = np.zeros(D)
        self.v = np.zeros(D)

        for epoch in range(epochs):
            for i in range(N):
                xi = X[i]
                ti = t[i]
                y = np.dot(self.w, xi)
                error = y - ti
                grad = 2 * error * xi + 2 * self.regularization * self.w
                self.v = beta*self.v + (1-beta)*grad
                self.w -= lr * self.v

            if decay:
                lr *= (1.0 / (1.0 + decay * epoch))

            if verbose and epoch % (epochs // 10) == 0:
                predictions = X @ self.w
                loss = np.mean((predictions - t) ** 2)
                print(f"[{epoch}] Loss: {loss:.6f}")

    def fit_adam(self, x, t, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, epochs=1000, decay=0.0, verbose=False):
        phi = self.design_matrix(x)
        N, D = phi.shape
        self.w = np.zeros(D)
        self.m = np.zeros(D)
        self.v = np.zeros(D)
        for epoch in range(epochs):
            for i in range(N):
                xi = phi[i]
                ti = t[i]
                y = np.dot(self.w, xi)
                error = y - ti
                grad = 2 * error * xi + 2 * self.regularization * self.w

                self.m = beta1 * self.m + (1 - beta1) * grad
                self.v = beta2 * self.v + (1 - beta2) * grad**2

                m_hat = self.m / (1 - beta1**(epoch + 1))
                v_hat = self.v / (1 - beta2**(epoch + 1))

                self.w -= lr * m_hat / (np.sqrt(v_hat) + epsilon)

            if decay:
                lr *= (1.0 / (1.0 + decay * epoch))

            if verbose and epoch % (epochs // 10) == 0:
                predictions = phi @ self.w
                loss = np.mean((predictions - t) ** 2)
                print(f"[{epoch}] Loss: {loss:.6f}")

    def numerical_gradiant(self, phi, t, eps=1e-5):
        grad=np.zeros_like(self.w)
        for i in range(len(self.w)):
            w_orig = self.w[i]

            self.w[i] = w_orig + eps
            loss_plus = np.mean((Phi @ self.w - t) ** 2) + self.regularization * np.sum(self.w ** 2)

            self.w[i] = w_orig - eps
            loss_minus = np.mean((Phi @ self.w - t) ** 2) + self.regularization* np.sum(self.w ** 2)

            grad[i] = (loss_plus - loss_minus) / (2 * eps)
            self.w[i] = w_orig  # reset weight
        return grad

    def predict(self, x):
        phi = self.design_matrix(x)
        return phi@self.w
    
    def __repr__(self):
        return f"PolyRegressor(degree={self.degree}, lambda={self.regularization}, fitted={self.w is not None})"

def decay_schedule(lr0, epoch):
    return lr0 / (1 + 0.01*epoch)

x = np.linspace(0, 1, 10)
t = np.sin(2*np.pi*x) + 0.1*np.random.randn(10)

model = PolyRegressor(degree=3)
model.fit(x, t, method='adam', lr=0.5, beta1=0.9, beta2=0.999, epochs=1000, verbose=True)

y = model.predict(x)
plt.scatter(x, t, color='red', label='Data')
plt.plot(x, y, label='Adam Fit')
plt.legend()
plt.show()