import torch
from torch import nn
import numpy as np
import time
from scipy import io as sio
from matplotlib import pyplot as plt

import  train,models
import os

from tftb import *
from gen_tfd_samples import gen_tfd_samples_paper_conv
from dit.other import renyi_entropy


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
    wvd = wvd1(ana_sig)
    wvd = wvd.flatten(order='F')
    wvd = wvd.astype(np.float32)
   
    wvd = wvd*alpha
    wvd = torch.tensor(wvd).to('cuda:0')
    wvd = wvd.view(1, 1, length, length).transpose(3, 2)

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

    path = 'results/test/' + filename
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(path + '/bat_' + str(snr) + '.eps', bbox_inches='tight')


    io.savemat(path + '/bat' + str(snr) + '.mat', {'bat' + str(snr): np.mat(abs(smooth_wvd))})


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
    smooth_wvd = torch.sum(smooth_wvd,1,keepdim=True)
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

        path = 'results/test/' + filename
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

        path = 'results/test/' + filename
        plt.savefig(path + '/' + str(img) + '_1.png', bbox_inches='tight')


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
            
            ideal[1, 0, :, :] = abs(ideal[1, 0, :, :]) / np.max(abs(ideal[1, 0, :, :]))
            smooth_wvd[1, 0, :, :] = abs(smooth_wvd[1, 0, :, :]) / np.max(abs(smooth_wvd[1, 0, :, :]))
            mc_l1[0,s] = mc_l1[0,s] + call1(ideal[1, 0, :, :], smooth_wvd[1, 0, :, :])
        mc_l1[0,s] = mc_l1[0,s] / 100
    io.savemat('results/test/' + filename +'/paper/data2_dist_kld.mat',{'TFD_KLD2' : np.mat(abs(mc_l1))})
   

def m_renyi(model, filename, noise):
    mc_max = 100
    mc_renyi = np.zeros((1,len(noise)))

    bat = io.loadmat('data/bat1.mat')
    sig1 = bat['bat1']
    sig1 = sig1.flatten()
    length = 400
    alpha = 1

    for s in range(len(noise)):
        for mm in range(mc_max):

            sig = (sig1[0:length]) / (max(sig1))
            sig = add_noise(sig, noise[s])

            ana_sig = analytic_x(sig)
            wvd = wvd1(ana_sig)
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
    io.savemat('results/test/' + filename + '/paper/r_kld.mat',{'r_KLD' : np.mat(abs(mc_renyi))})
    

if __name__ == '__main__':
    filename = 'conv_3kernel_8layersnew_dilate_b8_3se64_res_l11_snr10_256_low11_old_sin_cross_54_newwvd_3line'
    filename = filename + 'snr' + str(10)
    model = models.Encoder_2D_dilation_se_res_new88(kernel_size=3, dilation=1)
    
    model.load_state_dict(torch.load('trained/' + filename + '.pth'))
    model = model.cuda()
    
    bat(filename, model, 1, 10)
    test_synthetic_paper(filename, model, 10)
    noise = [0, 5, 15, 25, 35, 45]
    m_run(model, filename, noise)
    m_renyi(model, filename, noise)
    
#
