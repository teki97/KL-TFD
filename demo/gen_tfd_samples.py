from tftb import *

from scipy import io as sio

"""
produce : 2 lines(50%), 1 line + 1 sine(50%)
"""

def A_fhp(x, omega):
    # batch_size = x.shape[0]
    n = int(round(np.sqrt(x.shape[0])))

    yy = []
    xx = np.reshape(x, [n, n], order='F')
    yc = 1 / n * np.fft.fft2(xx).flatten(order='F')
    yy.append(yc[0].real)
    for inde in omega:
        yy.append(np.sqrt(2) * yc[inde].real)
    for inde in omega:
        yy.append(np.sqrt(2) * yc[inde].imag)
    yy = np.array(yy)


    y = yy.astype(np.float32)

    return y


def At_fhp(y, omega,n):
    k = y.shape[0]

    fx = np.zeros([n, n], dtype=complex).flatten()

    fx[0] = y[0]
    for inde in range(0, omega.shape[0]):
        # pytorch没有complex的数据类型...
        fx[int(omega[inde])] = np.sqrt(2) * (y[1 + int(inde)] + 1j * y[int((k + 3) / 2 - 1 + inde)])
    fx = np.reshape(fx, [n, n], order='F')
    x = n * np.fft.ifft2(fx).flatten(order='F')

    x = x.astype(np.float32)

    return x





def gen_tfd_samples_paper_conv(num = 1000, larg = 8, length = 64, snr=10):
    tf_all = []
    line = []
    sin = []
    phi = []

    # 随机选取两组参数
    # a = amgauss(length, length / 2, length / 2)
    a = amgauss(length, length / 2, 3 * length / 4)
    # s_a_1 = amgauss(length, length / 2, length / 2)
    s_a= amgauss(length, length / 2, 3 * length / 4)

    fnor1 = np.array([0.15, 0.45])
    fnor2 = np.array([0.45, 0.15])

    [line1, iflaw1] = fmlin(length, fnor1[0], fnor1[1])
    [line2, iflaw2] = fmlin(length, fnor2[0], fnor2[1])
    linee = a * line1 + a * line2

    # add noise
    linee = add_noise(linee, snr)

    line_wvd = tfrwv(linee)

    line_wvd = line_wvd.flatten(order = 'F')

    line.append(line_wvd)

    fs = [iflaw1, iflaw2]
    a1 = [a, a]

    tf = ideal_tf(length, fs, a1)
    tf_all.append(tf.flatten(order='F'))

    # sin
    s_fnor1 = np.array([0.1, 0.15, 0.315])
    s_fnor2 = np.array([0.35, 0.45])
    l_fnor666 = np.array([0.3, 0.15])

    s_fnorm0_1 = s_fnor1[1]

    [sin1, s_iflaw1] = fmsin(length, s_fnor1[0], s_fnor1[2], length, length / 2, s_fnorm0_1)
    [line3, s_iflaw2] = fmlin(length, s_fnor2[0], s_fnor2[1])
    [line4, s_iflaw3] = fmlin(length, l_fnor666[0], l_fnor666[1])
    # line4[0:100] = 0
    # line4[length-100:length] =0

    sin_line = s_a * sin1 + s_a * line3 + s_a * line4

    # add noise
    sin_line = add_noise(sin_line, snr)

    sin_line_wvd = tfrwv(sin_line)

    sin_line_wvd = sin_line_wvd.flatten(order='F')

    sin.append(sin_line_wvd)

    s_fs = [s_iflaw1, s_iflaw2, s_iflaw3]
    s_a2 = [s_a, s_a, s_a]

    s_tf = ideal_tf(length, s_fs, s_a2)
    tf_all.append(s_tf.flatten(order='F'))

    tf_all = np.array(tf_all)

    for l in line:
        y_samples.append(l)
        
    for s in sin:
        y_samples.append(s)

    y_samples = np.array(y_samples)
    
    return tf_all.astype(np.float32), y_samples.astype(np.float32)



