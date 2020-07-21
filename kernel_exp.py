import torch
from torch import nn
import numpy as np
import time
from scipy import io as sio
from matplotlib import pyplot as plt

import  models
import os

from tftb import *
from gen_tfd_samples import gen_tfd_samples_paper_conv


def bat(filename,model,alpha,snr):
    bat = io.loadmat('data/bat1.mat')
    sig1 = bat['bat1']
    sig1 = sig1.flatten()
    length = 400
    alpha = alpha
    sig = sig1[0:length]


    sig = (sig1[0:length])/(max(sig1))
    sig = add_noise(sig, snr)




    ana_sig = analytic_x(sig)
    wvd = tfrwv(ana_sig)
    wvd = wvd.flatten(order='F')
    wvd = wvd.astype(np.float32)
    # wvd = wvd*(max(wvd_t[0,:]-min(wvd_t[0,:]))/(max(wvd)-min(wvd)))
    wvd = wvd*alpha
    wvd = torch.tensor(wvd).to('cuda:0')
    wvd = wvd.view(1, 1, length, length).transpose(3, 2)

    # model = models.Encoder_2D_dilation(kernel_size=5,dilation=4)

    model.train(False)
    model.eval()

    smooth_wvd = model(wvd)
    smooth_wvd = smooth_wvd.cpu()
    smooth_wvd = smooth_wvd.data.numpy()
    wvd = wvd.cpu()
    wvd = wvd.data.numpy()

    dyn = 20
    plt.figure(figsize=(20, 40))
    plt.subplot(1, 2, 1)
    imageTF(smooth_wvd, dyn)
    plt.title('smooth wvd')
    plt.subplot(1, 2, 2)
    imageTF(wvd, dyn)
    plt.title('wvd')

    path = 'results/' + filename
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(path + '/bat_' + str(snr) + '.eps', bbox_inches='tight')

    plt.figure()
    plt.imshow(np.array(np.flipud(np.mat(abs(wvd)))))
    # plt.axis('off')
    plt.savefig(path + '/fig1.png', bbox_inches='tight')

    plt.figure()
    plt.imshow(np.array(np.flipud(np.mat(abs(smooth_wvd)))))
    # plt.axis('off')
    plt.savefig(path + '/fig2.png', bbox_inches='tight')




    io.savemat(path + '/bat' + str(snr) + '.mat', {'bat' + str(snr): np.mat(abs(smooth_wvd))})


def eeg(filename,model,alpha):
    eeg = io.loadmat('data/EEG_examples.mat')
    sig1 = eeg['sig_seiz']
    length = 256
    sig = sig1.flatten()
    alpha = alpha
    sig = (sig[0:length])/(max(sig))
    # ideal, wvd_t = gen_tfd_samples(num=2, length=length)


    ana_sig = analytic_x(sig)
    io.savemat('results/' + filename + '/eeg_example.mat', {'sig': np.mat(ana_sig)})
    wvd = tfrwv(ana_sig)
    wvd = wvd.flatten(order='F')
    wvd = wvd.astype(np.float32)
    # wvd = wvd*(max(wvd_t[0,:]-min(wvd_t[0,:]))/(max(wvd)-min(wvd)))
    wvd = wvd*alpha
    wvd = torch.tensor(wvd).to('cuda:0')
    wvd = wvd.view(1, 1, length, length).transpose(3, 2)

    # model = models.Encoder_2D_dilation(kernel_size=5,dilation=4)
    # model = models.Encoder_2D_dilation(kernel_size=1, dilation=1)
    # model.load_state_dict(torch.load('trained/' + filename + '.pth'))
    model = model.cuda()

    model.train(False)
    model.eval()

    smooth_wvd = model(wvd)
    smooth_wvd = smooth_wvd.cpu()
    smooth_wvd = smooth_wvd.data.numpy()
    wvd = wvd.cpu()
    wvd = wvd.data.numpy()

    dyn = 15
    plt.subplot(1, 2, 1)
    imageTF(smooth_wvd, dyn)
    plt.title('smooth wvd')
    plt.subplot(1, 2, 2)
    imageTF(wvd, dyn)
    plt.title('wvd')

    plt.savefig('results/' + filename + '/eeg_1.eps', bbox_inches='tight')

    plt.figure()
    plt.imshow(np.array(np.flipud(np.mat(abs(wvd)))))
    # plt.axis('off')
    plt.savefig('results/' + filename + '/eeg_fig1.png', bbox_inches='tight')

    plt.figure()
    plt.imshow(np.array(np.flipud(np.mat(abs(smooth_wvd)))))
    # plt.axis('off')
    plt.savefig('results/' + filename + '/eeg_fig2.png', bbox_inches='tight')

    io.savemat('results/' + filename + '/eeg.mat', {'eeg': np.mat(abs(smooth_wvd))})


def test_synthetic_paper(filename,model,snr):
    length=256
    num = 2
    ideal, wvd = gen_tfd_samples_paper_conv(length=length, snr=snr)

    ideal = torch.tensor(ideal).to('cuda:0')
    ideal = ideal.view(num, 1, length, length).transpose(3, 2)
    wvd = torch.tensor(wvd).to('cuda:0')
    wvd = wvd.view(num, 1, length, length).transpose(3, 2)

    model.train(False)
    model.eval()

    with torch.no_grad():
        smooth_wvd = model(wvd)
    smooth_wvd = smooth_wvd.cpu()
    smooth_wvd = smooth_wvd.data.numpy()
    ideal = ideal.cpu()
    ideal = ideal.data.numpy()
    wvd = wvd.cpu()
    wvd = wvd.data.numpy()

    dyn = 20
    for img in range(num):
        plt.figure(figsize=(20,60))
        plt.subplot(1, 3, 1)
        imageTF(smooth_wvd[img,0,:,:], dyn)
        plt.title('smooth wvd')
        plt.subplot(1, 3, 2)
        imageTF(wvd[img,0,:,:], dyn)
        plt.title('wvd')
        plt.subplot(1, 3, 3)
        imageTF(ideal[img, 0, :, :], dyn)
        plt.title('ideal')

        path = 'results/' + filename
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(path + '/' + str(img) + '.eps', bbox_inches='tight')

        io.savemat(path + '/' + str(img) + '.mat', {'x' + str(img): np.mat(abs(smooth_wvd[img,0,:,:]))})

    for img in range(num):
        plt.figure(figsize=(20,60))
        plt.subplot(1, 3, 1)
        plt.imshow(np.array(np.flipud(smooth_wvd[img,0,:,:])))
        plt.title('smooth wvd')
        plt.subplot(1, 3, 2)
        plt.imshow(np.array(np.flipud(wvd[img,0,:,:])))
        plt.title('wvd')
        plt.subplot(1, 3, 3)
        plt.imshow(np.array(np.flipud(ideal[img,0,:,:])))
        plt.title('ideal')

        path = 'results/' + filename
        plt.savefig(path + '/' + str(img) + '_1.eps', bbox_inches='tight')


def m_run(model, filename,noise):
    length = 256
    num = 2
    mc_max = 100
    mc_res = []
    mc_l1 = np.zeros((1,len(noise)))
    for s in range(len(noise)):
        for mm in range(mc_max):
            ideal, wvd = gen_tfd_samples_paper_conv(length=length, snr=noise[s])

            ideal = torch.tensor(ideal).to('cuda:0')
            ideal = ideal.view(num, 1, length, length).transpose(3, 2)
            wvd = torch.tensor(wvd).to('cuda:0')
            wvd = wvd.view(num, 1, length, length).transpose(3, 2)

            model.train(False)
            model.eval()

            with torch.no_grad():
                smooth_wvd = model(wvd)
            smooth_wvd = smooth_wvd.cpu()
            smooth_wvd = smooth_wvd.data.numpy()
            ideal = ideal.cpu()
            ideal = ideal.data.numpy()
            wvd = wvd.cpu()
            wvd = wvd.data.numpy()

            ideal[0, 0, :, :] = abs(ideal[0, 0, :, :]) / np.max(abs(ideal[0, 0, :, :]))
            smooth_wvd[0, 0, :, :] = abs(smooth_wvd[0, 0, :, :]) / np.max(abs(smooth_wvd[0, 0, :, :]))
            mc_l1[0,s] = mc_l1[0,s] + call1(ideal[0, 0, :, :], smooth_wvd[0, 0, :, :])
        mc_l1[0,s] = mc_l1[0,s] / 100
    io.savemat('results/' + filename +'/kld.mat',{'TFD_KLD' : np.mat(abs(mc_l1))})
    


def m_renyi(model, filename, noise):
    mc_max = 100
    mc_renyi = np.zeros((1,len(noise)))

    bat = io.loadmat('data/bat1.mat')
    sig1 = bat['bat1']
    sig1 = sig1.flatten()
    length = 400
    alpha = 1
    # sig = sig1[0:length]



    for s in range(len(noise)):
        for mm in range(mc_max):

            sig = (sig1[0:length]) / (max(sig1))
            sig = add_noise(sig, noise[s])

            ana_sig = analytic_x(sig)
            wvd = tfrwv(ana_sig)
            wvd = wvd.flatten(order='F')
            wvd = wvd.astype(np.float32)
            
            wvd = wvd * alpha
            wvd = torch.tensor(wvd).to('cuda:0')
            wvd = wvd.view(1, 1, length, length).transpose(3, 2)

            model.train(False)
            model.eval()

            with torch.no_grad():
                smooth_wvd = model(wvd)
            smooth_wvd = smooth_wvd.cpu()
            smooth_wvd = smooth_wvd.data.numpy()
            smooth_wvd = abs(smooth_wvd) / np.max(abs(smooth_wvd))
            
            test = renyi(smooth_wvd[0, 0, :, :])
            mc_renyi[0,s] = mc_renyi[0,s] + test
            
        mc_renyi[0,s] = mc_renyi[0,s] / 100
    io.savemat('results/' + filename + '/ldk_r.mat',{'r_KLD' : np.mat(abs(mc_renyi))})
    

def m_renyi_EEG(model, filename, noise):
    mc_max = 100
    mc_renyi = np.zeros((1,len(noise)))

    bat = io.loadmat('data/bat1.mat')
    sig1 = bat['bat1']
    sig1 = sig1.flatten()
    length = 400
    alpha = 1
    # sig = sig1[0:length]



    for s in range(len(noise)):
        for mm in range(mc_max):

            sig = (sig1[0:length]) / (max(sig1))
            sig = add_noise(sig, noise[s])

            ana_sig = analytic_x(sig)
            wvd = tfrwv(ana_sig)
            wvd = wvd.flatten(order='F')
            wvd = wvd.astype(np.float32)
            
            wvd = wvd * alpha
            wvd = torch.tensor(wvd).to('cuda:0')
            wvd = wvd.view(1, 1, length, length).transpose(3, 2)

            model.train(False)
            model.eval()

            with torch.no_grad():
                smooth_wvd = model(wvd)
            smooth_wvd = smooth_wvd.cpu()
            smooth_wvd = smooth_wvd.data.numpy()
            smooth_wvd = abs(smooth_wvd) / np.max(abs(smooth_wvd))
            
            test = renyi(smooth_wvd[0, 0, :, :])
            mc_renyi[0,s] = mc_renyi[0,s] + test
            
        mc_renyi[0,s] = mc_renyi[0,s] / 100
    io.savemat('results/' + filename + '/ldk_r.mat',{'r_KLD' : np.mat(abs(mc_renyi))})


if __name__ == '__main__':
    filename1 = 'kernel_learning_1'
    filename2 = 'kernel_learning_2'
    filename3 = 'kernel_learning_3'
    
    model1 = models.Encoder_2D_dilation_se_res_new(kernel_size=3, dilation=1)
    model2 = models.Encoder_2D_dilation_se_res_new(kernel_size=3, dilation=1)
    model3 = models.Encoder_2D_dilation_se_res_new(kernel_size=3, dilation=1)
    
    model1.load_state_dict(torch.load('trained/' + filename1 + '.pth'))
    model1 = model1.cuda()

    model2.load_state_dict(torch.load('trained/' + filename2 + '.pth'))
    model2 = model2.cuda()

    model3.load_state_dict(torch.load('trained/' + filename3 + '.pth'))
    model3 = model3.cuda()

    bat(filename1, model1, 1, 20)
    test_synthetic_paper(filename2, model2, 20)
    eeg(filename3, model3, 1)
    
