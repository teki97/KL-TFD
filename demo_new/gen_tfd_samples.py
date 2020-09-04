from tftb import *
from scipy import io as sio


def gen_tfd_samples_paper_conv(num = 1000, larg = 8, length = 64, snr=10):
    tf_all = []
    line = []
    sin = []
    phi = []

    a = amgauss(length, length / 2,  3*length / 4)
    s_a= amgauss(length, length / 2, 3*length / 4)

    fnor1 = np.array([0.3, 0.15])
    fnor2 = np.array([0.1, 0.2, 0.35])
    fnorm0_1 = fnor2[1]

    [line1, iflaw1] = fmlin_x(length, fnor1[0], fnor1[1], length / 2)
    [sin22, iflaw22] = fmsin_x(length, fnor2[0], fnor2[2], length, length / 2, fnorm0_1)
    linee = a * line1 + a * sin22

    # add noise
    linee = add_noise(linee, snr)

    line_wvd = wvd1(linee)

    line_wvd = line_wvd.flatten(order = 'F')

    line.append(line_wvd)

    fs = [iflaw1, iflaw22]
    a1 = [a, a]

    tf = ideal_tf(length, fs, a1)
    tf_all.append(tf.flatten(order='F'))

    # sin
    s_fnor1 = np.array([0.1, 0.2, 0.35])
    s_fnor2 = np.array([0.33, 0.45])
    l_fnor666 = np.array([0.28, 0.18])

    s_fnorm0_1 = s_fnor1[1]

    [sin1, s_iflaw1] = fmsin_x(length, s_fnor1[0], s_fnor1[2], length, length / 2, s_fnorm0_1)
    [line3, s_iflaw2] = fmlin_x(length, s_fnor2[0], s_fnor2[1], length / 2)
    [line4, s_iflaw3] = fmlin_x(length, l_fnor666[0], l_fnor666[1], length / 2)

    sin_line = s_a * sin1 + s_a * line3 + s_a * line4

    # add noise
    sin_line = add_noise(sin_line, snr)

    sin_line_wvd = wvd1(sin_line)

    sin_line_wvd = sin_line_wvd.flatten(order='F')

    sin.append(sin_line_wvd)

    s_fs = [s_iflaw1, s_iflaw2, s_iflaw3]
    s_a2 = [s_a, s_a, s_a]

    s_tf = ideal_tf(length, s_fs, s_a2)
    tf_all.append(s_tf.flatten(order='F'))

    tf_all = np.array(tf_all)
    io.savemat('data/synthetic_data1.mat', {'sig': np.mat(linee)})
    io.savemat('data/synthetic_data2.mat', {'sig': np.mat(sin_line)})

    y_samples = []
    for l in line:
        y_samples.append(l)
        
    for s in sin:
        y_samples.append(s)

    y_samples = np.array(y_samples)
    
    return tf_all.astype(np.float32), y_samples.astype(np.float32)

