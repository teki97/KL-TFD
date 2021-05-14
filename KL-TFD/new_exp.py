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
from gen_tfd_samples import gen_tfd_samples_sin_cross


def bat(filename,model,snr):
    device = torch.device("cuda:0" if torch.cuda.is_available() is True else 'cpu')
    bat = io.loadmat('data/bat1.mat')
    sig1 = bat['bat1']
    sig1 = sig1.flatten()
    length = 400
    sig = sig1/(max(sig1))

    ana_sig = analytic_x(sig)
    wvd = wvd1(ana_sig)
    wvd = wvd.flatten(order='F')
    wvd = wvd.astype(np.float32)
    wvd = torch.tensor(wvd).to(device)
    wvd = wvd.view(1, 1, length, length).transpose(3, 2)

    model.train(False)
    model.eval()

    smooth_wvd = model(wvd)
    smooth_wvd = smooth_wvd.cpu()
    smooth_wvd = smooth_wvd.data.numpy()
    wvd = wvd.cpu()
    wvd = wvd.data.numpy()
    test = renyi(smooth_wvd[0, 0, :, :])

    dyn = 20
    plt.figure(figsize=(20, 40))
    plt.subplot(1, 2, 1)
    imageTF(smooth_wvd, dyn)
    plt.title('smooth wvd(' + str(test)+')')
    plt.subplot(1, 2, 2)
    imageTF(wvd, dyn)
    plt.title('wvd')

    path = 'results/test/' + filename
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(path + '/bat_' + str(snr) + '.eps', bbox_inches='tight')
    io.savemat(path + '/bat' + str(snr) + '.mat', {'bat' + str(snr): np.mat(abs(smooth_wvd))})

def synthetic(filename,model,snr):
    device = torch.device("cuda:0" if torch.cuda.is_available() is True else 'cpu')
    length=256
    num = 20
    ideal, wvd = gen_tfd_samples_sin_cross(num=num,length=length,snr=snr)
    ideal = torch.tensor(ideal).to(device)
    ideal = ideal.view(num, 1, length, length).transpose(3, 2)
    wvd = torch.tensor(wvd).to(device)
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

        path = 'results/test/' + filename
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(path + '/' + str(img) + '.png', bbox_inches='tight')

        io.savemat(path + '/' + str(img) + '.mat', {'x' + str(img): np.mat(abs(smooth_wvd[img, 0, :, :]))})

def synthetic_paper(filename,model,snr):
    device = torch.device("cuda:0" if torch.cuda.is_available() is True else 'cpu')
    length=256
    num = 2
    ideal, wvd = gen_tfd_samples_paper_conv(length=length, snr=snr)
    ideal = torch.tensor(ideal).to(device)
    ideal = ideal.view(num, 1, length, length).transpose(3, 2)
    wvd = torch.tensor(wvd).to(device)
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
        path = 'results/test/' + filename
        if not os.path.exists(path):
            os.makedirs(path)
        io.savemat(path + '/' + str(img) + '.mat', {'x' + str(img): np.mat(abs(smooth_wvd[img, 0, :, :]))})

        ideal[img, 0, :, :] = abs(ideal[img, 0, :, :]) / np.max(abs(ideal[img, 0, :, :]))
        smooth_wvd[img, 0, :, :] = abs(smooth_wvd[img, 0, :, :]) / np.max(abs(smooth_wvd[img, 0, :, :]))
        nmse = call1(ideal[img, 0, :, :], smooth_wvd[img, 0, :, :])
        plt.figure(figsize=(20,60))
        plt.subplot(1, 3, 1)
        imageTF(smooth_wvd[img,0,:,:], dyn)
        plt.title('smooth wvd (' + str(nmse) + ')')
        plt.subplot(1, 3, 2)
        imageTF(wvd[img,0,:,:], dyn)
        plt.title('wvd')
        plt.subplot(1, 3, 3)
        imageTF(ideal[img, 0, :, :], dyn)
        plt.title('ideal')

        plt.savefig(path + '/' + str(img) + '.eps', bbox_inches='tight')

def m_run_3(model, filename,noise):
    device = torch.device("cuda:0" if torch.cuda.is_available() is True else 'cpu')
    length = 256
    num = 2
    mc_max = 100
    mc_l1 = np.zeros((1,len(noise)))
    path = 'results/test/' + filename
    if not os.path.exists(path):
        os.makedirs(path)
    for s in range(len(noise)):
        for mm in range(mc_max):
            ideal, wvd = gen_tfd_samples_paper_conv(length=length, snr=noise[s])
            ideal = torch.tensor(ideal).to(device)
            ideal = ideal.view(num, 1, length, length).transpose(3, 2)
            wvd = torch.tensor(wvd).to(device)
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
    io.savemat(path +'/data3_dist_kld.mat',{'TFD_KLD3' : np.mat(abs(mc_l1))})
    print(mc_l1)

def m_run_1(model, filename,noise):
    device = torch.device("cuda:0" if torch.cuda.is_available() is True else 'cpu')
    length = 256
    num = 2
    mc_max = 100
    mc_l1 = np.zeros((1,len(noise)))
    path = 'results/test/' + filename
    if not os.path.exists(path):
        os.makedirs(path)
    for s in range(len(noise)):
        for mm in range(mc_max):
            ideal, wvd = gen_tfd_samples_paper_conv(length=length, snr=noise[s])
            ideal = torch.tensor(ideal).to(device)
            ideal = ideal.view(num, 1, length, length).transpose(3, 2)
            wvd = torch.tensor(wvd).to(device)
            wvd = wvd.view(num, 1, length, length).transpose(3, 2)

            model.train(False)
            model.eval()

            with torch.no_grad():
                smooth_wvd = model(wvd)
            smooth_wvd = smooth_wvd.cpu()
            smooth_wvd = smooth_wvd.data.numpy()
            ideal = ideal.cpu()
            ideal = ideal.data.numpy()
            
            ideal[0, 0, :, :] = abs(ideal[0, 0, :, :]) / np.max(abs(ideal[0, 0, :, :]))
            smooth_wvd[0, 0, :, :] = abs(smooth_wvd[0, 0, :, :]) / np.max(abs(smooth_wvd[0, 0, :, :]))
            mc_l1[0,s] = mc_l1[0,s] + call1(ideal[0, 0, :, :], smooth_wvd[0, 0, :, :])
            
        mc_l1[0,s] = mc_l1[0,s] / 100
    io.savemat(path +'/data1_dist_kld.mat',{'TFD_KLD1' : np.mat(abs(mc_l1))})
    
def m_renyi(model, filename, noise):
    device = torch.device("cuda:0" if torch.cuda.is_available() is True else 'cpu')
    mc_max = 100
    mc_renyi = np.zeros((1,len(noise)))

    bat = io.loadmat('data/bat1.mat')
    sig1 = bat['bat1']
    sig1 = sig1.flatten()
    length = 400

    path = 'results/test/' + filename
    if not os.path.exists(path):
        os.makedirs(path)

    for s in range(len(noise)):
        for mm in range(mc_max):
            sig = (sig1[0:length]) / (max(sig1))
            sig = add_noise(sig, noise[s])

            ana_sig = analytic_x(sig)
            wvd = wvd1(ana_sig)
            wvd = wvd.flatten(order='F')
            wvd = wvd.astype(np.float32)

            wvd = torch.tensor(wvd).to(device)
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
    io.savemat(path + '/r_kld.mat',{'r_KLD' : np.mat(abs(mc_renyi))})
    print(mc_renyi)
   

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() is True else 'cpu')
 
    # model1 = models.Encoder_conv_bam_skip(nlayers=8)
    # model2 = models.Encoder_conv_bam_skip(nlayers=12)
    # model3 = models.Encoder_conv_bam_skip(nlayers=16)
    # model4 = models.Encoder_conv_bam_skip(nlayers=20)
    # model5 = models.Encoder_conv_bam_skip(nlayers=24)
    model6 = models.Encoder_conv_bam_skip(nlayers=28)
    model7 = models.Encoder_conv_bam_skip(nlayers=32)
    # model1.load_state_dict(torch.load('trained/k8.pth'))
    # model2.load_state_dict(torch.load('trained/k12.pth'))
    # model3.load_state_dict(torch.load('trained/k16.pth'))
    # model4.load_state_dict(torch.load('trained/k20.pth'))
    # model5.load_state_dict(torch.load('trained/k24.pth'))
    model6.load_state_dict(torch.load('trained/k28.pth'))
    model7.load_state_dict(torch.load('trained/k32.pth'))
    # model1.to(device)
    # model2.to(device)
    # model3.to(device)
    # model4.to(device)
    # model5.to(device)
    model6.to(device)
    model7.to(device)

    noise = range(0, 50, 5)
    # m_renyi(model1, 'k8', noise)
    # m_renyi(model2, 'k12', noise)
    # m_renyi(model3, 'k16', noise)
    # m_renyi(model4, 'k20', noise)
    # m_renyi(model5, 'k24', noise)
    m_renyi(model6, 'k28', noise)
    m_renyi(model7, 'k32', noise)
    #
    # m_run_1(model1, 'k8', noise)
    # m_run_1(model2, 'k12', noise)
    # m_run_1(model3, 'k16', noise)
    # m_run_1(model4, 'k20', noise)
    # m_run_1(model5, 'k24', noise)
    # m_run_1(model6, 'k28', noise)
    # m_run_1(model7, 'k32', noise)
    # #
    # m_run_3(model1, 'k8', noise)
    # m_run_3(model2, 'k12', noise)
    # m_run_3(model3, 'k16', noise)
    # m_run_3(model4, 'k20', noise)
    # m_run_3(model5, 'k24', noise)
    # m_run_3(model6, 'k28', noise)
    # m_run_3(model7, 'k32', noise)

    # synthetic_paper('k8', model1, 10)
    # synthetic_paper('k12', model2, 10)
    # synthetic_paper('k16', model3, 10)
    # synthetic_paper('k20', model4, 10)
    # synthetic_paper('k24', model5, 10)
    # synthetic_paper('k28', model6, 10)
    # synthetic_paper('k32', model7, 10)

    # bat('k8', model1, 50)
    # bat('k12', model2, 1, 50)
    # bat('k16', model3, 1, 50)
    # bat('k20', model4, 1, 50)
    # bat('k24', model5, 1, 50)
    # bat('k28', model6, 1, 50)
    # bat('k32', model7, 1, 50)









