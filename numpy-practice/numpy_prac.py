import numpy as np

def zero_out_border(a) :
    assert a.ndim == 2
    a[[0, a.shape[0]-1]] = 0
    a[:, 0][:] = 0
    a[:, a.shape[1]-1] = 0
    return a


def checkerboard(n,m) :
    a = np.zeros((n, m), dtype=int)
    for i in range(n):
        for j in range(m):
            a[i, j] = ((i + j) % 2)
    return a


def row_col_sorted(a) :
    assert a.ndim == 2
    return all(sorted(a[:, i]) for i in range(a.shape[1])) and all(sorted(a[i]) for i in range(a.shape[0]))

def sorted(a) :
    assert a.ndim == 1
    return all(a[i] <= a[i + 1] for i in range(len(a) - 1))

def spiral(n, m):
    totalLayers = (min(m, n) // 2) + 1
    value = 1
    arr = np.zeros((n, m), dtype=int)
    for layer in range(totalLayers):
        value = makeLayer(arr, value, layer)
    return arr

def makeLayer(arr, value, layer):
    for i in range(arr.shape[0] - 1 - layer - layer):
        arr[i + layer][0 + layer] = value
        value += 1
    for i in range(arr.shape[1] - 1 - layer - layer):
        arr[arr.shape[0] - 1 - layer][i + layer] = value
        value += 1
    for i in range(arr.shape[0] - 1 - layer - layer, 0, -1):
        arr[i + layer][arr.shape[1] - 1 - layer] = value
        value += 1
    for i in range(arr.shape[1] - 1 - layer - layer, 0, -1):
        arr[0 + layer][i + layer] = value
        value += 1
    return value
