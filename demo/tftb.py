import numpy as np
import scipy.io as io
import random
from matplotlib import pyplot as plt
import math

EPS = 1e-8
"""
WIGNER-VILLE DISTRIBUTION
"""

def tfrwv(x):
    x_row = x.shape[0]
    t = np.arange(0, x_row)

    N = x_row

    t_row = t.shape[0]

    
    tfr = np.zeros([N, t_row], dtype = complex)
    for irow in range(0, t_row):
        ti = t[irow]
        taumax = min(ti, x_row - ti - 1, int(np.round(N / 2)) - 2)
        tau = np.arange(- taumax, taumax + 1)
        
        indices = np.ones([tau.shape[0], 1])
        for jj in range(0, tau.shape[0]):
            if tau[jj] >= 0:
                indices[jj, 0] =  np.mod(N - 1 + tau[jj], N - 1)
            else:
                indices[jj, 0] = np.mod(N - 1 + tau[jj], N - 1) + 1

        a1 = x[ti + tau]
        a2 = x[ti - tau]
        bb = a1 * np.conj(a2)
        for i in range(0, indices.shape[0]):
            indi = int(indices[i, 0])
            tfr[indi, irow] = bb[i]

    tfr = np.fft.fft(tfr, axis = 0)
    tfr = tfr.real

    return tfr


"""
PLOT
"""
def imageTF(M, dyn):
    M = M / np.max(M)
    M = np.maximum(M, pow(10, -dyn / 10))
    M = 10 * np.log10(M)
    M = np.mat(M)

    plt.imshow(np.array(np.flipud(M)))


"""
IDEAR TIME-FREQUENCY DISTRIBUTION
FS.SHAPE    : (NUM, LENGTH) , EACH fs.shape : (length, )
A.SHAPE     : (NUM, LENGTH) , EACH a.shape  : (length, )
"""
def ideal_tf(n, fs, a):
    num = len(fs)
    tfs = []
    tf = np.zeros([n ,n])

    fsd = fs
    for i in range(0, num):
        temp = np.round(fsd[i],4)
        tfs.append(np.zeros([n ,n]))
        temp1 = tfs[i]
        temp2 = a[i]

        for k in range(0, len(temp)):
            if temp[k]<0:
                temp[k] = min(n-1, n- 1 -(np.round(-2*(n-1)*temp[k])))
                aa = int(temp[k])
                temp1[aa, k] = pow(temp2[k], 2)
            else:
                temp[k] = min(n, 1 + np.round(2 * n * temp[k]))
                aa = int(temp[k]) - 1
                temp1[aa, k] = pow(temp2[k], 2)

        fsd[i] = temp
        tfs[i] = temp1
        tf = tf + tfs[i]

    return tf


"""
GENERATE LINEAR FM SIGNAL
"""
def fmlin(*params):

    if (len(params) == 1):
        n = params[0]
        fnormi = 0.0
        fnormf = 0.5
        t0 = round(n/2)
    elif (len(params) == 2):
        n = params[0]
        fnormi = params[1]
        fnormf = 0.5
        t0 = round(n/2)
    elif (len(params) == 3):
        n = params[0]
        fnormi = params[1]
        fnormf = params[2]
        t0 = round(n/2)
    else:
        n = params[0]
        fnormi = params[1]
        fnormf = params[2]
        t0 = params[3]

   
    y = np.array(range(1, n + 1))
    y = fnormi * (y - t0) +((fnormf - fnormi)/(2.0 * (n-1))) * (pow((y -1), 2) - pow((t0 - 1), 2))
    aa = 1j * 2.0 * np.pi * y
    y = np.exp(1j * 2.0 * np.pi * y)
    y = y / y[t0 - 1]

    iflaw = np.linspace(fnormi, fnormf, n)

    return y, iflaw


"""
GENERATE SINE FM SIGNAL
"""
def fmsin(*params):
    if (len(params) == 1):
        n = params[0]
        fnormin = 0.05
        fnormax = 0.45
        period = n
        t0 = round(n / 2)
        fnorm0 = 0.25
        pm1 = +1
    elif (len(params) == 2):
        n = params[0]
        fnormin = params[1]
        fnormax = 0.45
        period = n
        t0 = round(n / 2)
        fnorm0 = 0.25
        pm1 = +1
    elif (len(params) == 3):
        n = params[0]
        fnormin = params[1]
        fnormax = params[2]
        period = n
        t0 = round(n / 2)
        fnorm0 = 0.5 * (fnormin + fnormax)
        pm1 = +1
    elif (len(params) == 4):
        n = params[0]
        fnormin = params[1]
        fnormax = params[2]
        period = params[3]
        t0 = round(n / 2)
        fnorm0 = 0.25
        pm1 = +1
    elif (len(params) == 5):
        n = params[0]
        fnormin = params[1]
        fnormax = params[2]
        period = params[3]
        t0 = params[4]
        fnorm0 = 0.25
        pm1 = +1
    elif (len(params) == 6):
        n = params[0]
        fnormin = params[1]
        fnormax = params[2]
        period =  params[3]
        t0 = params[4]
        fnorm0 = params[5]
        pm1 = +1
    else:
        n = params[0]
        fnormin = params[1]
        fnormax = params[2]
        period = params[3]
        t0 = params[4]
        fnorm0 = params[5]
        pm1 = params[6]

    fnormid = 0.5 * (fnormax + fnormin)
    delta_f = 0.5 * (fnormax - fnormin)
    phi = -pm1 * np.arccos((fnorm0 - fnormid) / delta_f)
    time_t = np.arange(1, n + 1) - t0
    phase = 2 * np.pi *fnormid * time_t + delta_f * period *(np.sin(2 * np.pi * time_t / period + phi) - np.sin(phi))
    y = np.exp(1j * phase)
    iflaw = fnormid + delta_f * np.cos(2 * np.pi * time_t / period + phi)

    return y,iflaw


def fm_multi(*params):
    if (len(params) == 1):
        n = params[0]
        fnormin = 0.05
        fnormax = 0.45
        period = n
        t0 = round(n / 2)
        fnorm0 = 0.25
        pm1 = +1
    elif (len(params) == 2):
        n = params[0]
        fnormin = params[1]
        fnormax = 0.45
        period = n
        t0 = round(n / 2)
        fnorm0 = 0.25
        pm1 = +1
    elif (len(params) == 3):
        n = params[0]
        fnormin = params[1]
        fnormax = params[2]
        period = n
        t0 = round(n / 2)
        fnorm0 = 0.5 * (fnormin + fnormax)
        pm1 = +1
    elif (len(params) == 4):
        n = params[0]
        fnormin = params[1]
        fnormax = params[2]
        period = params[3]
        t0 = round(n / 2)
        fnorm0 = 0.25
        pm1 = +1
    elif (len(params) == 5):
        n = params[0]
        fnormin = params[1]
        fnormax = params[2]
        period = params[3]
        t0 = params[4]
        fnorm0 = 0.25
        pm1 = +1
    elif (len(params) == 6):
        n = params[0]
        fnormin = params[1]
        fnormax = params[2]
        period =  params[3]
        t0 = params[4]
        fnorm0 = params[5]
        pm1 = +1
    else:
        n = params[0]
        fnormin = params[1]
        fnormax = params[2]
        period = params[3]
        t0 = params[4]
        fnorm0 = params[5]
        pm1 = params[6]

    # print('n >= 0 \n fnormin, fnormax and fnorm0 must be between -0.5 and 0.5 \n fnormin must be lower than fnormax \n fnorm0 must be between fnormin and fnormax' )

    fnormid = 0.25 * (fnormax + 3*fnormin)
    delta_f = 0.25 * (fnormax - fnormin)
    fm1 = random.uniform(0,9)
    fm2 = random.uniform(0, 9)
    fm3 = random.uniform(0, 9)

    # phi = -pm1 * np.arccos((fnorm0 - fnormid) / delta_f)
    time_t = np.arange(1, n + 1) - t0

    phase1 = delta_f * fm1 * period * (np.sin(2 * np.pi * time_t /(fm1 * period)))
    phase2 = delta_f * fm2 * period * (np.cos(2 * np.pi * time_t /(fm2 * period)))
    phase3 = delta_f * fm3 * period * (np.sin(2 * np.pi * time_t /(fm3 * period)))

    f1 = delta_f*np.cos(2 * np.pi * time_t /(fm1 * period))
    f2 = - delta_f*np.sin(2 * np.pi * time_t /(fm2 * period))
    f3 = delta_f*np.cos(2 * np.pi * time_t /(fm3 * period))

    phase = 2 * np.pi *fnormid * time_t + phase1 + phase2 + phase3
    y = np.exp(1j * phase)
    iflaw = fnormid + f1 + f2 + f3

    return y,iflaw


"""
ADD NOISE
"""
def add_noise(x, snr):
    if x.imag.any():
        iscomplex = 1
    else:
        iscomplex = 0

    # print(x.shape)
    ps = 1 / x.shape[0]  * sum(pow(abs(x), 2))

    pn = ps / pow(10, snr / 10)

    if iscomplex:
        real_part = np.random.randn(x.shape[0], 1).flatten()
        image_part = np.random.randn(x.shape[0], 1).flatten()
        w_0 = np.sqrt(pn / 2) * (real_part + 1j * image_part)
    else:
        w_0 = np.sqrt(pn) * np.random.randn(x.shape[0])

    # print(x.shape)
    # print(w_0.shape)
    y = np.add(x, w_0.T)
    # print(y.shape)

    return y


"""
GENERATE GAUSSIAN AMPLITUDE MODULATION
"""
def amgauss(N = 64, t0 = 32, T = 16):
    tmt0 = np.arange(1, N + 1) - t0
    y = np.exp(- pow((tmt0 / T), 2) * np.pi)

    return y


"""
CALL NMSE
"""
def calnmse(orgsig, recsig):
    mse = pow(np.linalg.norm(orgsig.flatten() - recsig.flatten()), 2)
    sigEner = pow(np.linalg.norm(orgsig.flatten()), 2)
    nmse = (mse / sigEner)

    return nmse


def call1(orgsig, recsig):
    mse = (np.linalg.norm(orgsig.flatten() - recsig.flatten(),1))
    sigEner = (np.linalg.norm(orgsig.flatten(),1))
    nmse = (mse / sigEner)

    return nmse


def analytic_x(sig):
    n = len(sig)
    analyze_X = np.zeros(n,dtype=complex)
    analyze_x = np.zeros(n,dtype=complex)
    if n % 2 == 0:
        true_X = np.fft.fft(sig)
        analyze_X[0] = true_X[0]
        analyze_X[1:(int(n/2))] = 2*true_X[1:(int(n/2))]
        analyze_X[int(n/2)] = true_X[int(n/2)]
        analyze_x = np.fft.ifft(analyze_X)
    else:
        true_X = np.fft.fft(sig)
        analyze_X[0] = true_X[0]
        analyze_X[1:(math.ceil(n / 2))] = 2 * true_X[1:(math.floor(n / 2))]
        analyze_X[int(n / 2)] = true_X[int(n / 2)]
        analyze_x = np.fft.ifft(analyze_X)

    return analyze_x


def integ2d(mat):
    sz = mat.shape
    x = np.array(range(0, sz[0]))
    y = np.array(range(0, sz[1]))

    xshape = x.shape
    yshape = y.shape

    mat = (np.sum(mat.T, 1).T - mat[:, 0] / 2 - mat[:, sz[0] - 1] / 2) * (x[1] - x[0])
    dmat = mat[0: sz[1] - 2] + mat[1 : sz[1] - 1]
    dy = (y[1 : sz[1] - 1] - y[0 : sz[1] - 2]) / 2
    som = np.sum(dmat * dy)

    return som


def renyi(tfr):
    sz = tfr.shape
    alpha = 3
    
    tfr = tfr / integ2d(tfr)
    r = np.log2(integ2d(tfr ** alpha) + EPS) / (1 - alpha)

    return r


