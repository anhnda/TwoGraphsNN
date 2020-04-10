import numpy as np

N = 3
k = np.linspace(-10 , 10, num=10000 )


def K(x):
    return x * x * 1.0 / 2


def gen_single_matrix(kv):
    matrix = np.ndarray((N, N))
    matrix.fill(0.001)
    for i in range(N):
        matrix[i, i] = K(kv - (-(N + 1.0) / 2.0 + i)) * 2
    eigvalues = np.linalg.eigvalsh(matrix)
    return eigvalues


def gen_all_matrix(k):
    mat_list = []
    for kv in k:
        matrix = gen_single_matrix(kv)
        mat_list.append(matrix)
    return mat_list


def lam_tu_tu(mat_list):
    a1 = []
    a2 = []
    a3 = []
    for v in mat_list:
        a1.append(v[0])
        a2.append(v[1])
        a3.append(v[2])
    return a1, a2, a3


def lam_nhanh_len_ti(mat_list):
    ars = [[], [], []]
    for v in mat_list:
        for j in range(3):
            ars[j].append(v[j])
    return ars[0], ars[1], ars[2]


def lam_nhanh_nua(mat_list):
    mat = np.vstack(mat_list)
    print(type(mat), mat.shape)
    return mat[:, 0], mat[:, 1], mat[:, 2]


def plotAll():

    import scipy.interpolate
    mat_list = gen_all_matrix(k)
    a1, a2, a3 = lam_nhanh_nua(mat_list)
    import matplotlib.pyplot as plt
    print (a1)
    print(a2)
    print(a3)


    fig, ax = plt.subplots()

    x = k

    # x, a1 = scipy.interpolate.spalde(x, a1)
    plt.plot(x,a1, )
    plt.plot(x,a2, )
    plt.plot(x,a3, )
    plt.ylim(0,6)
    plt.show()

plotAll()
