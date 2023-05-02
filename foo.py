import matplotlib as plt
import sympy as sp

seq_len, mem, N = sp.symbols('seq_len mem N')

f = seq_len ** 2 / N + seq_len * mem + mem ** 2 * N

ff = (seq_len / N + mem) ** 2 * N 

sp.Subs(f, N, 1)

sp.plotting.plot3d(f, (seq_len, 1, 256), (mem, 1, 256), show=True)



