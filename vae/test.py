import numpy as np


def permute_dims(z):
    z_cols = np.unstack(z, axis=1)
    store = []
    for z_col in z_cols:
        rand = np.random.randint(3)  # denna kanske inte borde vara hårdkodad
        z_perm = np.roll(z_col, shift=rand, axis=0)
        store.append(z_perm)
    z_permuted = np.stack(store, axis=1)
    return z_permuted

z = np.array(([3,1,2], [14, 15, 17], [100, 101, 102]))
z_perm = permute_dims(z)
print(z_perm)