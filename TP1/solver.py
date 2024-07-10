from scipy.optimize import minimize

# Define the objective function f(x)
def objective(x):
    x1, x2 = x
    c1, cm1 = 2.3, 0.47
    c2, cm2 = 4.27, 0.26
    return c1 * (x1 + 10)**2 + cm1 * (x1 + 5) + c2 * (x2 + 10)**2 + cm2 * (x2 + 5)


# Initial guess (starting point for the optimizer)
x0 = [25, 25]

# Use SLSQP (Sequential Least Squares Programming) method
sol = minimize(objective, x0, method='SLSQP')

print('Optimal solution:', sol.x)
print('Minimum value of the objective function:', sol.fun)
