import jax.numpy as np
from jax import jit, value_and_grad, vmap
from jax.config import config
import tables as tb
import time
from functools import partial
config.update("jax_enable_x64", True)
config.update('jax_disable_jit', False)

NS = 32
PI = np.pi


def biot_savart(r, I, dl, l):
	mu_0 = 1.0
	mu_0I = I * mu_0
	mu_0Idl = (
		mu_0I[:, np.newaxis, np.newaxis] * dl
	)
	r_minus_l = (
		r[np.newaxis, np.newaxis, :]
		- l
	)
	top = np.cross(
		mu_0Idl, r_minus_l
	)
	bottom = (
		np.linalg.norm(r_minus_l, axis=-1) ** 3
	)
	B = np.sum(
		top / bottom[:, :, np.newaxis], axis=(0, 1)
	)
	return B

biot_savart_surface = vmap(vmap(biot_savart, (0, None, None, None), 0), (1, None, None, None), 1)

def quadratic_flux(r, I, dl, l, nn, sg):
	B = biot_savart_surface(r, I, dl, l)
	return (0.5 * np.sum(np.sum(nn * B) ** 2 * sg))

def unpack_fourier(fc):
	xc = fc[0]
	yc = fc[1]
	zc = fc[2]
	xs = fc[3]
	ys = fc[4]
	zs = fc[5]
	return xc, yc, zc, xs, ys, zs

def r(fc, theta):
	NC = fc.shape[1]
	NF = fc.shape[2]
	xc, yc, zc, xs, ys, zs = unpack_fourier(fc)
	x = np.zeros((NC, NS + 1))
	y = np.zeros((NC, NS + 1))
	z = np.zeros((NC, NS + 1))
	for m in range(NF):
		arg = m * theta
		carg = np.cos(arg)
		sarg = np.sin(arg)
		x += (
			xc[:, np.newaxis, m] * carg[np.newaxis, :]
			+ xs[:, np.newaxis, m] * sarg[np.newaxis, :]
		)
		y += (
			yc[:, np.newaxis, m] * carg[np.newaxis, :]
			+ ys[:, np.newaxis, m] * sarg[np.newaxis, :]
		)
		z += (
			zc[:, np.newaxis, m] * carg[np.newaxis, :]
			+ zs[:, np.newaxis, m] * sarg[np.newaxis, :]
		)
	return np.concatenate(
		(x[:, :, np.newaxis], y[:, :, np.newaxis], z[:, :, np.newaxis]), axis=2
	)

def r1(fc, theta):
	NC = fc.shape[1]
	NF = fc.shape[2]
	xc, yc, zc, xs, ys, zs = unpack_fourier(fc)
	x1 = np.zeros((NC, NS + 1))
	y1 = np.zeros((NC, NS + 1))
	z1 = np.zeros((NC, NS + 1))
	for m in range(NF):
		arg = m * theta
		carg = np.cos(arg)
		sarg = np.sin(arg)
		x1 += (
			-m * xc[:, np.newaxis, m] * sarg[np.newaxis, :]
			+ m * xs[:, np.newaxis, m] * carg[np.newaxis, :]
		)
		y1 += (
			-m * yc[:, np.newaxis, m] * sarg[np.newaxis, :]
			+ m * ys[:, np.newaxis, m] * carg[np.newaxis, :]
		)
		z1 += (
			-m * zc[:, np.newaxis, m] * sarg[np.newaxis, :]
			+ m * zs[:, np.newaxis, m] * carg[np.newaxis, :]
		)
	return np.concatenate(
		(x1[:, :, np.newaxis], y1[:, :, np.newaxis], z1[:, :, np.newaxis]), axis=2
	)

def loss(r_surf, nn, sg, fc):
	NC = fc.shape[1]
	theta = np.linspace(0, 2 * PI, NS + 1)
	l = r(fc, theta)[:,:-1,:]
	dl = r1(fc, theta)[:,:-1,:] * (2 * PI / NS)
	I = np.ones(NC)
	return quadratic_flux(r_surf, I, l, dl, nn, sg)

nn = np.load("nn.npy")
sg = np.load("sg.npy")
r_surf = np.load("r_surf.npy")

loss_partial = partial(loss, r_surf, nn, sg)

loss_and_grad_func = jit(value_and_grad(loss_partial))

def main():

	with tb.open_file("coils.hdf5", "r") as f:
		fc = np.asarray(f.root.coilSeries[:, :, :])

	N = 100
	lr = 0.0000001

	t_init = time.time()

	for n in range(N):
		loss, grad = loss_and_grad_func(fc)
		fc = fc - grad * lr
		print("{}: loss is {}".format(n,loss))

	t_fin = time.time()
	print("Time to run is {}".format(t_fin - t_init))







if __name__ =="__main__":
	main()