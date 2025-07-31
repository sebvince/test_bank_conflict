import numpy as np

m = 16384
n = 16384
k = 16384

a = np.random.randint(low=0, high=255, size=(m, k//2), dtype=np.uint8)
a_scale = np.random.randint(low=0, high=255, size=(m, k//32), dtype=np.uint8)
b = np.random.randint(low=0, high=255, size=(n, k//2), dtype=np.uint8)
b_scale = np.random.randint(low=0, high=255, size=(m, k//32), dtype=np.uint8)
# out = np.random.randint(low=0, high=1<<16,size=(m, n), dtype=np.uint16)
out = np.random.randint(low=0, high=1<<32,size=(m, n), dtype=np.uint32)

a.tofile("a.bin")
a_scale.tofile("a_scale.bin")
b.tofile("b.bin")
b_scale.tofile("b_scale.bin")
out.tofile("out.bin")
