import jax.numpy as np
from jax import grad


def f(x):
	return np.cos(x)

grad_f = grad(f)

x0 = np.pi / 6

print("The value of cos(x) at x={} is {}".format(x0, f(x0)))
print("The derivative of cos(x) at x={} is {}".format(x0, grad_f(x0)))
