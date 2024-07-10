import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Definindo a função objetivo para aceitar um vetor x
def objective(x):
    x1, x2 = x
    c1, cm1 = 2.3, 0.47
    c2, cm2 = 4.27, 0.26
    return c1 * (x1 + 10)**2 + cm1 * (x1 + 5) + c2 * (x2 + 10)**2 + cm2 * (x2 + 5)

# Definindo a restrição para aceitar um vetor x
def constraints(x):
    x1, x2 = x
    return x1 + x2 - 50

# Criando o grid para o plot
x_range = np.linspace(0, 100, 400)
y_range = np.linspace(0, 100, 400)
X, Y = np.meshgrid(x_range, y_range)
Z = objective([X, Y])  # Usar a função com desempacotamento de vetor
G = constraints([X, Y])  # Usar a função com desempacotamento de vetor

# Descobrindo ponto ótimo
# Constraint dictionary
con = {'type': 'ineq', 'fun': constraints}

# Initial guess (starting point for the optimizer)
x0 = [25, 25]

# Use SLSQP (Sequential Least Squares Programming) method
sol = minimize(objective, x0, method='SLSQP', constraints=con)

print('Optimal solution:', sol.x)
print('Minimum value of the objective function:', sol.fun)

# Plotando a superfície e as curvas de nível
fig = plt.figure(figsize=(14, 6))

# 3D surface plot of the objective function
ax = fig.add_subplot(121, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
ax.contour(X, Y, G, levels=[0], colors='red')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x)')
ax.set_title('3D Surface Plot of the Objective Function')

# Contour plot with the constraint
ax2 = fig.add_subplot(122)
contour = ax2.contourf(X, Y, Z, levels=50, cmap='viridis')
ax2.contour(X, Y, G, levels=[0], colors='red')
ax2.plot(sol.x[0], sol.x[1], 'ro')
ax2.set_xlabel('x1')
ax2.set_ylabel('x2')
ax2.set_title('Contour Plot of the Objective Function with Constraint')
plt.colorbar(contour, ax=ax2)

plt.tight_layout()
plt.show()