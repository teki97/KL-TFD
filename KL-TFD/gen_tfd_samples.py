from tftb import *


def gen_tfd_samples(num = 1000, length = 64, snr=10):
    tf_all = []
    line = []
    sin = []
    phi = []

    # 随机选取两组参数
    for i in range(0, int(num / 2)):
        a = amgauss(length, length / 2, 4*length / 5)
        s_a = amgauss(length, length / 2, 4*length / 5)
        a_1 = amgauss(length, length / 2, 3 * length / 4)
        s_a_1 = amgauss(length, length / 2, 3 * length / 4)

        fnor1 = np.divide(np.random.random(2), 2)
        fnor2 = np.divide(np.random.random(2), 2)
        # print('l_fnor:' + str(fnor1))
        # print('l_fnor:' + str(fnor2))

        [line1, iflaw1] = fmlin_x(length, fnor1[0], fnor1[1], length / 2)
        [line2, iflaw2] = fmlin_x(length, fnor2[0], fnor2[1], length / 2)
        linee = a * line1 + a * line2

        # add noise
        linee = add_noise(linee, snr)

        line_wvd = wvd1(linee)
        # line1_wvd = tfrwv(linee1)

        # imageTF(line_wvd, 20)
        # plt.show()
        line_wvd = line_wvd.flatten(order = 'F')
        # line1_wvd = line1_wvd.flatten(order='F')



        # print(line_wvd)
        line.append(line_wvd)
        # line.append(line1_wvd)

        fs = [iflaw1, iflaw2]
        a = [a, a]
        a_1 = [a_1, a_1]

        # tf = ideal_tf(length, fs, a_1)
        # tf = ideal_tf_1(length, fs)
        tf = ideal_tf(length, fs, a)
        # imageTF(tf, 20)
        # plt.show()


        # sin
        s_fnor1 = np.sort(np.divide(np.random.random(3), 2))
        s_fnor2 =np.divide(np.random.random(2), 2)
        # print('s_fnor:' + str(s_fnor1))
        # print('s_fnor:' + str(s_fnor2))
        s_fnorm0_1 = s_fnor1[1]

        [sin1, s_iflaw1] = fmsin_x(length, s_fnor1[0], s_fnor1[2], length, length / 2, s_fnorm0_1)
        [line3, s_iflaw2] = fmlin_x(length, s_fnor2[0], s_fnor2[1], length / 2)
        # [line3, s_iflaw2] = fmlin(length, s_fnor1[0], s_fnor1[1])

        sin_line = s_a * sin1 + s_a *line3

        # add noise
        sin_line = add_noise(sin_line, snr)

        sin_line_wvd = wvd1(sin_line)
        # sin_line_wvd1 = tfrwv(sin_line)

        # imageTF(sin_line_wvd, 20)
        # plt.show()
        sin_line_wvd = sin_line_wvd.flatten(order = 'F')
        # sin_line_wvd1 = sin_line_wvd1.flatten(order='F')


        sin.append(sin_line_wvd)
        # sin.append(sin_line_wvd1)

        s_fs = [s_iflaw1, s_iflaw2]
        s_a = [s_a, s_a]
        s_a_1 = [s_a_1, s_a_1]

        # s_tf = ideal_tf(length, s_fs, s_a_1)
        # s_tf = ideal_tf_1(length, s_fs)
        s_tf = ideal_tf(length, s_fs, s_a)

        # imageTF(tf, 20)
        # imageTF(s_tf, 20)
        # plt.show()

        tf_all.append(tf.flatten(order = 'F'))
        # tf_all.append(tf.flatten(order='F'))
        tf_all.append(s_tf.flatten(order = 'F'))
        # tf_all.append(s_tf.flatten(order='F'))


    tf_all = np.array(tf_all)

    ###-----------y samples-------------###
    # measure
    # 我服了，无论什么范围右边都是开区间！！！！！
    # 我服了，我发现我之前构造矩阵构造错了
    # 我服了，之前构造的没错


    y_samples = []
    # kkk = 0
    for (l, s) in zip(line, sin):

        # 此处注意，矩阵相乘不能直接用×，而是需要用np.dot()
        # print(phi.shape)
        # print(l.shape)
        # print(s.shape)
        # imageTF(np.reshape(l, (64, 64), order='F'), 20)
        # plt.show()
        # print(l.shape)
        # print(phi.shape)
        # y_test = np.dot(phi, l)

        # y_samples.append(np.dot(phi, l))
        y_samples.append(l)
        # y1 = np.dot(phi[1, :], l)

        # print(y_test.shape)
        # y_test = y_samples[kkk]
        # kkk = kkk+1
        # x_test = At_fhp(y_test, omega, 128)

        # imageTF(np.reshape(x_test,(128, 128), order='F'), 20)
        # plt.show()

        # y_samples.append(np.dot(phi, s))
        y_samples.append(s)
    y_samples = np.array(y_samples)
    # np.random.shuffle(y_samples)


    # np.save('data/tf_recovery_tf.npy', tf_all)
    # np.save('data/tf_recovery_y.npy', y_samples)
    # print(tf_all)
    return tf_all.astype(np.float32), y_samples.astype(np.float32)


def gen_tfd_samples_sin(num = 1000, length = 64, snr=10):
    tf_all = []
    line = []
    sin = []
    phi = []

    # 随机选取两组参数
    for i in range(0, int(num / 2)):
        s_a = amgauss(length, length / 2, 4 * length / 5)
        s_a_1 = amgauss(int(length/2), length / 4, 2 * length / 5)

        # sin
        s_fnor1 = np.sort(np.divide(np.random.random(9), 2))
        # s_fnor2 = np.divide(np.random.random(2), 2)

        s_fnorm0_1 = s_fnor1[2]

        [sin1, s_iflaw1] = fmsin_x(length, s_fnor1[0], s_fnor1[4], length, length / 2, s_fnorm0_1)
        [line1, iflaw1] = fmlin_x(length, s_fnor1[3], s_fnor1[1], length / 2)
        # [line11, iflaw11] = fmlin_x(length, s_fnor1[7], s_fnor1[5], length / 2)

        sin_line = s_a * sin1 + s_a *line1 #+ s_a * line11

        s_fnor3 = np.sort(np.divide(np.random.random(7), 2))

        s_fnorm0_2 = s_fnor3[3]

        [sin2, s_iflaw2] = fmsin_x(length, s_fnor3[2], s_fnor3[4], length, length / 2, s_fnorm0_2)
        [line2, iflaw2] = fmlin_x(length, s_fnor3[0], s_fnor3[1], length / 2)
        [line22, iflaw22] = fmlin_x(length, s_fnor3[6], s_fnor3[5], length / 2)
        sin_line_cross = s_a * sin2 + s_a * line2  + s_a * line22

        # two_fs_1 = (np.divide(np.random.random(2), 2))
        # two_fs_2 = (np.divide(np.random.random(2), 2))
        #
        # [linee1, l_iflaw1] = fmlin_x(length, two_fs_1[0], two_fs_1[1], length / 2)
        # [linee2, l_iflaw2] = fmlin_x(length, two_fs_2[1], two_fs_2[0], length / 2)
        # #
        # line_two = s_a * linee1 + s_a * linee2 #+s_a * line22
        #
        # two_fs_3 = (np.divide(np.random.random(2), 2))
        # two_fs_4 = np.sort(np.divide(np.random.random(3), 2))
        #
        # [linee3, l_iflaw3] = fmlin_x(length, two_fs_3[0], two_fs_3[1], length / 2)
        # [linee4, l_iflaw4] = fmsin_x(length, two_fs_4[0], two_fs_4[2], length, length / 2, two_fs_4[1])
        # #
        # sin_two = s_a * linee3 + s_a * linee4  # +s_a * line22

        # add noise
        sin_line = add_noise(sin_line, snr)
        sin_line_wvd = wvd1(sin_line)

        sin_line_cross = add_noise(sin_line_cross, snr)
        sin_line_cross_wvd = wvd1(sin_line_cross)

        # line_two = add_noise(line_two, snr)
        # line_two_wvd = wvd1(line_two)
        #
        # sin_two = add_noise(sin_two, snr)
        # sin_two_wvd = wvd1(sin_two)

        # normalization
        # sin_line_wvd = sin_line_wvd / np.max(sin_line_wvd)
        # sin_line_cross_wvd = sin_line_cross_wvd / (np.max(sin_line_cross_wvd))

        # line_cross = add_noise(line_cross, snr)
        # line_cross_wvd = wvd1(line_cross)

        sin.append(sin_line_wvd.flatten(order = 'F'))
        sin.append(sin_line_cross_wvd.flatten(order='F'))
        # sin.append(line_two_wvd.flatten(order='F'))
        # sin.append(sin_two_wvd.flatten(order='F'))
        # sin.append(line_cross_wvd.flatten(order='F'))

        s_fs = [s_iflaw1, iflaw1]
        s_fs_cross = [s_iflaw2, iflaw2, iflaw22]
        # line_two_fs = [l_iflaw1, l_iflaw2]
        # sin_two_fs = [l_iflaw3, l_iflaw4]

        # l_fs = [l_iflaw1, l_iflaw2]
        a1 = [s_a, s_a]
        a2 = [s_a, s_a, s_a]


        s_tf = ideal_tf(length, s_fs, a1)
        s_tf_cross = ideal_tf(length, s_fs_cross, a2)
        # line_two_tf = ideal_tf(length, line_two_fs, a2)
        # sin_two_tf = ideal_tf(length, sin_two_fs, a2)

        # normalization
        # s_tf = s_tf / (np.max(s_tf))
        # s_tf_cross = s_tf_cross / (np.max(s_tf_cross))

        # l_tf = ideal_tf(length, l_fs, a2)
        # s_tf = ideal_tf_1(length, s_fs)
        # s_tf_cross = ideal_tf_1(length, s_fs_cross)


        tf_all.append(s_tf.flatten(order = 'F'))
        tf_all.append(s_tf_cross.flatten(order='F'))
        # tf_all.append(line_two_tf.flatten(order='F'))
        # tf_all.append(sin_two_tf.flatten(order='F'))
        # tf_all.append(l_tf.flatten(order='F'))

        # T
        # sin_line_wvd_t = sin_line_wvd.flatten(order = 'C')
        # sin.append(sin_line_wvd.flatten(order = 'C'))
        # sin.append(sin_line_cross_wvd.flatten(order = 'C'))
        # tf_all.append(s_tf.flatten(order = 'C'))
        # tf_all.append(s_tf_cross.flatten(order = 'C'))


    tf_all = np.array(tf_all)

    y_samples = []

    for s in sin:
        y_samples.append(s)
    y_samples = np.array(y_samples)

    return tf_all.astype(np.float32), y_samples.astype(np.float32)


def gen_tfd_samples_paper_conv(num = 1000, larg = 8, length = 64, snr=10):
    tf_all = []
    line = []
    sin = []
    phi = []

    # 随机选取两组参数
    # a = amgauss(length, length / 2, length / 2)
    a = amgauss(length, length / 2,  3*length / 4)
    # s_a = amgauss(length, length / 2, length / 2)
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
    # line_wvd = line_wvd / np.max(line_wvd)

    line.append(line_wvd)

    fs = [iflaw1, iflaw22]
    a1 = [a, a]

    tf = ideal_tf(length, fs, a1)
    tf_all.append(tf.flatten(order='F'))

    # sin
    s_fnor1 = np.array([0.1, 0.2, 0.35])
    s_fnor2 = np.array([0.4, 0.45])
    l_fnor666 = np.array([0.28, 0.18])

    s_fnorm0_1 = s_fnor1[1]

    [sin1, s_iflaw1] = fmsin_x(length, s_fnor1[0], s_fnor1[2], length, length / 2, s_fnorm0_1)
    [line3, s_iflaw2] = fmlin_x(length, s_fnor2[0], s_fnor2[1], length / 2)
    [line4, s_iflaw3] = fmlin_x(length, l_fnor666[0], l_fnor666[1], length / 2)
    # line4[0:100] = 0
    # line4[length-100:length] =0

    sin_line = s_a * sin1 + s_a * line3 + s_a * line4

    # add noise
    # print (sin_line)
    sin_line = add_noise(sin_line, snr)

    sin_line_wvd = wvd1(sin_line)

    sin_line_wvd = sin_line_wvd.flatten(order='F')
    # sin_line_wvd = sin_line_wvd / np.max(sin_line_wvd)

    sin.append(sin_line_wvd)

    s_fs = [s_iflaw1, s_iflaw2, s_iflaw3]
    s_a2 = [s_a, s_a, s_a]

    s_tf = ideal_tf(length, s_fs, s_a2)
    # s_tf1 = ideal_tf(length, s_fs, s_a2)
    # s_tf2 = ideal_tf(length, [s_iflaw3], [s_a])
    # s_tf2[:, 0:100] = 0
    # s_tf2[:, length-100:length] = 0
    # s_tf = s_tf1 + s_tf2
    # imageTF(s_tf,20)
    tf_all.append(s_tf.flatten(order='F'))



    # tf_all.append(tf.flatten(order='F'))

    # tf_all.append(s_tf.flatten(order='F'))


    tf_all = np.array(tf_all)
    io.savemat('data/synthetic_data1.mat', {'sig': np.mat(linee)})
    io.savemat('data/synthetic_data2.mat', {'sig': np.mat(sin_line)})

    ###-----------y samples-------------###
    # measure
    # 我服了，无论什么范围右边都是开区间！！！！！
    # 我服了，我发现我之前构造矩阵构造错了
    # 我服了，之前构造的没错

    #
    # # print(list1[0].shape)
    #
    # dftmtx = np.fft.fft(np.eye(n))
    # phi_init = np.kron(dftmtx, dftmtx)
    # phi.append(phi_init[0, :])
    # real_part = phi_init.real
    # imag_part = phi_init.imag
    #
    # for inde in list1[0]:
    #     phi.append(np.sqrt(2) * real_part[inde, :])
    # for inde in list1[0]:
    #     phi.append(np.sqrt(2) * imag_part[inde, :])
    #
    # test_phi = np.array(phi)

    # 归一化
    # 此处把我坑惨了，应该是larg+1的平方，之前写成larg的平方了。。
    # for ii in range(0, pow(larg + 1, 2)):
    #     phi[ii] = phi[ii] / np.linalg.norm(phi[ii])
    #
    # phi = np.array(phi)
    # # print(phi.shape)
    # np.save('data/dic' + str(phi.shape[0]) + '_' + str(phi.shape[1]) + '.npy', phi.T)

    # # y = phi * tfrwv(s)
    y_samples = []
    # kkk = 0
    for l in line:

        # 此处注意，矩阵相乘不能直接用×，而是需要用np.dot()
        # print(phi.shape)
        # print(l.shape)
        # print(s.shape)
        # imageTF(np.reshape(l, (64, 64), order='F'), 20)
        # plt.show()
        # print(l.shape)
        # print(phi.shape)
        # y_test = np.dot(phi, l)

        # y_samples.append(np.dot(phi, l))
        y_samples.append(l)
        # y1 = np.dot(phi[1, :], l)

        # print(y_test.shape)
        # y_test = y_samples[kkk]
        # kkk = kkk+1
        # x_test = At_fhp(y_test, omega, 128)

        # imageTF(np.reshape(x_test,(128, 128), order='F'), 20)
        # plt.show()

        # y_samples.append(np.dot(phi, s))
    for s in sin:
        y_samples.append(s)

    y_samples = np.array(y_samples)
    # np.random.shuffle(y_samples)


    # np.save('data/tf_recovery_tf.npy', tf_all)
    # np.save('data/tf_recovery_y.npy', y_samples)
    # print(tf_all)
    return tf_all.astype(np.float32), y_samples.astype(np.float32)


def gen_tfd_samples_sin_cross(num = 1000, length = 64, snr=10):
    tf_all = []
    line = []
    sin = []
    phi = []

    # 随机选取两组参数
    for i in range(0, num):
        # a = amgauss(length, length / 2, length / 2)
        # s_a = amgauss(length, length / 2, length / 2)
        a = amgauss(length, length / 2, 3 * length / 4)
        s_a = amgauss(length, length / 2, 3 * length / 4)

        # sin
        s_fnor1 = np.sort(np.divide(np.random.random(7), 2))
        # s_fnor2 =np.divide(np.random.random(2), 2)

        s_fnorm0_1 = s_fnor1[1]

        [sin1, s_iflaw1] = fmsin_x(length, s_fnor1[0], s_fnor1[4], length, length / 2, s_fnorm0_1)
        [line1, iflaw1] = fmlin_x(length, s_fnor1[3], s_fnor1[2], length / 2)
        [line2, iflaw2] = fmlin_x(length, s_fnor1[5], s_fnor1[6], length / 2)

        sin_line = s_a * sin1 + s_a *line1 + s_a * line2

        # add noise
        sin_line = add_noise(sin_line, snr)
        sin_line_wvd = wvd1(sin_line)

        sin_line_wvd = sin_line_wvd.flatten(order = 'F')
        # sin_line_wvd = sin_line_wvd / np.max(sin_line_wvd)
        sin.append(sin_line_wvd)

        s_fs = [s_iflaw1, iflaw1, iflaw2]
        s_a = [s_a, s_a, s_a]


        s_tf = ideal_tf(length, s_fs, s_a)
        # s_tf = ideal_tf_1(length, s_fs)

        tf_all.append(s_tf.flatten(order = 'F'))


    tf_all = np.array(tf_all)

    y_samples = []

    for s in sin:
        y_samples.append(s)
    y_samples = np.array(y_samples)

    return tf_all.astype(np.float32), y_samples.astype(np.float32)






