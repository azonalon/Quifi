#%%
from shutil import copyfile
copyfile("../target/debug/quifi.dll", "./quifi.pyd")
#%%
import quifi
import os, sys
import numpy as np
print(quifi.say_hello())
print(quifi.optional(None))
print(quifi.optional(42, name='hahah'))
X = np.arange(200)*1.0
print(quifi.array_fuck(X, X))
res = quifi.find_solution(X, np.exp(-0.1*X**2), n_basefunctions=1)
#%%
import matplotlib.pyplot as plt
plt.plot(res[0]['y_est'])
plt.plot(X, np.exp(-0.1*X**2))
plt.semilogx()
#%%
import matplotlib.pyplot as plt
import numpy as np
import h5py
import quifi
cmap = plt.get_cmap("twilight")
try:
    f = h5py.File("../data/pytest_exponentials.h5")
except:
    f = h5py.File("data/pytest_exponentials.h5")
for j in range(0,max([int(x) for x in f.keys()])+1):
    grp = f[str(j)]
    wav = grp['wave'][:]
    (aux0offset, field, od, aux0limit) = (
        grp["current_mod_offset[V]"][()],
        grp["magnetic_field[T]"][()],
        grp["pump_filter[OD]"][()],
        grp["current_mod_lim[V]"][()])
    if j != 0:
        continue
    dt = grp['dt'][()]
    ss = grp['supersample'][()]
    X   = np.arange(0, len(wav[0])) * dt
    Xh   = np.arange(0, len(wav[0][0])) / 2e6 * ss
    # Ysub  = grp['subtracted'][:]
    # Xsub  = grp['subtracted_df'][()] * np.arange(len(Ysub))
    dF   = (grp['f1'][()] - grp['f0'][()])/202*2
    F = np.arange(202) * dF
    od  = grp["pump_filter[OD]"][()]
    par = grp['current_mod_offset[V]'][()]
    field = grp['magnetic_field[T]'][()]
    im = wav[0]
    i0, i1 = 204, 350
    Y = im[:,i0:i1]
    X = np.array([Xh[:i1-i0]*1e3]*Y.shape[0])
    Ys = np.copy(Y)
    # print(X.shape, Y.shape)
    # plt.imshow(X)
    # plt.figure()
    ps = []
    for i in range(len(Y)):
        res = quifi.find_solution(X[i], Y[i], n_basefunctions=1)[0]
        # print(res['alpha'])
        print(res)
        exit(0)
        ps.append(np.concatenate([
            np.sort(res['p_opt']),
            res['c_opt'],
            [np.sum(res['wresid']**2)]
        ]))
        Ys[i] -= res['c_opt'][-1]
        if i % 20 == 0:
            a = np.max(Y[i]) - np.min(Y[i])
            b = np.min(Y[i])
            plt.plot(X[i], (Y[i]-b)/a)
            plt.plot(X[i], (res['y_est']-b)/a)
    plt.xlabel("time[ms]")
    plt.ylabel("Demod normalized")
    plt.figure()
    plt.imshow(Ys)
#%%

    plt.xlabel("frequency[arb.u.]")
    plt.plot(np.transpose(ps)[2], 's')
    plt.plot(np.transpose(ps)[3], 's')
    plt.plot(np.transpose(ps)[4], 's')
    plt.ylabel("Linear coefficents")
    plt.ylim(-1, 1)
    plt.figure()
    plt.xlabel("frequency[arb.u.]")
    plt.ylabel("Decay constants [ms]")
    plt.plot(np.transpose(ps)[0], 'd')
    plt.plot(np.transpose(ps)[1], 'd')
    plt.figure()
    plt.ylabel("SSR")
    plt.semilogy()
    plt.xlabel("frequency[arb.u.]")
    # plt.ylim(0, 1e-7)
    plt.plot(np.transpose(ps)[5], 'd')

#%%
plt.figure(figsize=(8,8))
plt.imshow(np.transpose(Ys), aspect='auto')