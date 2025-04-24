import numpy as np
import matplotlib.pyplot as plt

class BayesRegressor:
    def __init__(self, degree=1, alpha=1.0, beta=1.0):
        self.degree = degree
        self.alpha = alpha
        self.beta = beta
        self.m_N = None
        self.S_N = None
        self.regularization = alpha/beta
    
    def design_matrix(self, x):
        return np.vander(x, N=self.degree + 1, increasing=True)
    
    def fit_robbins_monro(self, x, t, eta=0.1, epochs=1, verbose=False):
        x = np.asarray(x)
        t = np.asarray(t)
        phi_all = self.design_matrix(x)
        N, M = phi_all.shape
        self.w = np.random.randn(M) * 0.1

        n = 1  
        for epoch in range(epochs):
            for i in range(N):
                phi = phi_all[i]
                y = np.dot(self.w, phi)
                error = y - t[i]
                grad = error * phi
                a_n = eta / n
                self.w -= a_n * grad
                n += 1

            if verbose:
                predictions = phi_all @ self.w
                loss = np.mean((predictions - t) ** 2)
                print(f"[{epoch}] Loss: {loss:.6f}")

    def fit_robbins_monro_map(self, x, t, eta=0.1, epochs=1, verbose=False):
        x = np.asarray(x)
        t = np.asarray(t)
        phi_all = self.design_matrix(x)
        N, M = phi_all.shape
        self.w = np.random.randn(M) * 0.1

        n = 1
        for epoch in range(epochs):
            for i in range(N):
                phi = phi_all[i]
                y = np.dot(self.w, phi)
                error = y - t[i]
                grad = error * phi + self.regularization * self.w
                a_n = eta / n
                self.w -= a_n * grad
                n += 1

            if verbose:
                predictions = phi_all @ self.w
                loss = np.mean((predictions - t) ** 2)
                print(f"[{epoch}] MAP Loss: {loss:.6f}")

    def fit(self, x, t):
        phi = self.design_matrix(x)
        I = np.eye(phi.shape[1])

        S_inv = self.alpha * I + self.beta * (phi.T @ phi)
        self.S_N = np.linalg.inv(S_inv)
        self.m_N = self.beta * self.S_N @ phi.T @ t 
    
    def predict(self, x, return_std=False):
        phi = self.design_matrix(x)
        if hasattr(self, 'w'):  # For Robbins-Monro methods
            mean = phi @ self.w
        else:  # For exact Bayesian fitting
            mean = phi @ self.m_N
        
        if return_std:
            if hasattr(self, 'w'):  # For Robbins-Monro methods
                std = np.zeros_like(mean)
            else:  # For exact Bayesian fitting
                var = np.sum(phi @ self.S_N * phi, axis=1) + 1 / self.beta
                std = np.sqrt(var)
            return mean, std
        
        return mean 
    
    def sample_weight(self, n_samples=1):
        return np.random.multivariate_normal(self.m_N, self.S_N, size=n_samples)
    
    def __repr__(self):
        return f"BayesRegressor(degree={self.degree}, alpha={self.alpha}, beta={self.beta}, fitted={self.m_N is not None})"

    def plot_predictive(self, x_train, t_train, x_test, std_scale=1.0):
        mean, std = self.predict(x_test, return_std=True)
    
        plt.figure(figsize=(8, 5))
        plt.scatter(x_train, t_train, color='red', label='Training Data')
        plt.plot(x_test, mean, label='Predictive Mean', color='blue')
        plt.fill_between(x_test, mean - std_scale * std, mean + std_scale * std,
                         alpha=0.3, color='blue', label=f'±{std_scale} std dev')
        plt.title("Bayesian Predictive Distribution")
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_sample_fits(self, x_train, t_train, x_test, n_samples=5):
        Phi_test = self.design_matrix(x_test)
    
        sampled_ws = np.random.multivariate_normal(self.m_N, self.S_N, n_samples)
    
        plt.figure(figsize=(8, 5))
        plt.scatter(x_train, t_train, color='red', label='Training Data')
    
        for w in sampled_ws:
            y_sample = Phi_test @ w
            plt.plot(x_test, y_sample, lw=1, alpha=0.7)

        plt.title(f"{n_samples} Posterior Sample Fits")
        plt.grid(True)
        plt.show()


# Small synthetic dataset with noise
x_train = np.linspace(0, 1, 100)  # Only 5 points
t_train = np.sin(2 * np.pi * x_train) + 0.1 * np.random.randn(100)  # Add some noise

# Test points for prediction
x_test = np.linspace(0, 1, 500)

# Initialize the BayesRegressor model
model = BayesRegressor(degree=3, alpha=0.01, beta=1.0)

# Fit using Robbins-Monro
model.fit_robbins_monro(x_train, t_train, eta=1, epochs=1000, verbose=True)
y_rm = model.predict(x_test)

# Debugging step: check if y_rm is None or valid
print(f"y_rm: {y_rm[:5]}...")  # Print first 5 values for inspection

# Reset model and fit using Robbins-Monro MAP
model = BayesRegressor(degree=10, alpha=0.01, beta=1)
model.fit_robbins_monro_map(x_train, t_train, eta=1, epochs=1000, verbose=True)
y_rm_map = model.predict(x_test)

# Reset model and fit using Exact Bayesian Fitting
model = BayesRegressor(degree=10, alpha=0.01, beta=1)
model.fit(x_train, t_train)
y_exact = model.predict(x_test)

# Check if y_rm_map and y_exact are valid
print(f"y_rm_map: {y_rm_map[:5]}...")  # Print first 5 values for inspection
print(f"y_exact: {y_exact[:5]}...")    # Print first 5 values for inspection

# Plotting all three results
plt.figure(figsize=(10, 6))
plt.scatter(x_train, t_train, color='red', label='Training Data')

plt.plot(x_test, y_rm, label='Robbins-Monro', color='blue', linestyle='--')
plt.plot(x_test, y_rm_map, label='Robbins-Monro MAP', color='green', linestyle=':')
plt.plot(x_test, y_exact, label='Exact Bayesian', color='orange', linestyle='-.')

plt.title('Comparison of Robbins-Monro, Robbins-Monro MAP, and Exact Bayesian')
plt.legend()
plt.grid(True)
plt.show()
