# Test the choice of \gamma in QNP
# we test on the TV+wavelet regularization.

SampleTrj = 'Radial' # 'Spiral' #

if SampleTrj == 'Radial':
    trj_file = 'data/radial/trj.npy'
    mps_file = 'data/radial/mps.npy' 
elif SampleTrj == 'Spiral':
    trj_file = "data/spiral/trj.npy"
    mps_file = "data/spiral/mps.npy"

import os
import scipy.io
import numpy as np
import sigpy as sp
import optalgTao as opt

image_type = 'brain' #'knee' #  'cardiac'  #
if image_type == 'knee':
    data = scipy.io.loadmat('data/knee_GT.mat')
elif image_type == 'brain':
    data = scipy.io.loadmat('data/brain_GT.mat')
im_real = data['im_real']
im_imag = data['im_imag']

trj = np.load(trj_file)
mps = np.load(mps_file)
(nc, sy, sx) = mps.shape
S = sp.linop.Multiply(mps.shape[1:], mps)
F = sp.linop.NUFFT(mps.shape, coord=trj)#, toeplitz=True
W = sp.linop.Wavelet(S.ishape)
A = F * S
LL = sp.app.MaxEig(A.H*A,dtype=np.complex_).run()# * 1.01#A.N
# normalize A
A = np.sqrt(1/LL) * A 

im = im_real+1j*im_imag
im = im/np.max(np.abs(im))
im_original = im
b = A*im
noise_level = 1e-2
b_m,b_n,b_k = b.shape
np.random.seed(2)
noise_real = np.random.randn(b_m,b_n,b_k)
np.random.seed(5)
noise_imag = np.random.randn(b_m,b_n,b_k)
b_noise  = b+noise_level*(noise_real+1j*noise_imag)
snr = 10*np.log10(np.linalg.norm(b)/np.linalg.norm(b_noise-b))
print('The measurements SNR is {0}'.format(snr))


MaxIter_QN = 20
TV_type = 'l1'
TV_bound = 'Dirchlet'
a_k = 1


MaxIter_inner_TV_set = np.array([5,10,20,50,100])
MaxIter_inner_wavTV_set = np.array([5,10,20,50,100])
gamma = 1.7
verbose = True

if SampleTrj == 'Radial':
    if image_type == 'knee':
        folderName = 'results/Radial/Knee/Inner'
    elif image_type == 'brain':
        folderName = 'results/Radial/Brain/Inner'
elif SampleTrj == 'Spiral':
    if image_type == 'knee':
        folderName = 'results/Spiral/Knee/Inner'
    elif image_type == 'brain':
        folderName = 'results/Spiral/Brain/Inner'

if not os.path.exists(folderName):
    os.mkdir(folderName)


if SampleTrj == 'Radial':
    if image_type == 'knee':
        lamda = 5e-4
        beta = 5e-4
    elif image_type == 'brain':
        lamda = 1e-4
        beta = 5e-4
elif SampleTrj == 'Spiral':
    if image_type == 'knee':
        lamda = 5e-4
        beta = 5e-4
    elif image_type == 'brain':
        lamda = 5e-4
        beta = 5e-4

# wavelet+TV part
for iter in range(MaxIter_inner_wavTV_set.size):
    algName = '/QNPDual_TV_Wavelet'
    loc = folderName+algName
    if not os.path.exists(loc):
        os.mkdir(loc)
    loc = loc+'/'+str(MaxIter_inner_wavTV_set[iter])
    if not os.path.exists(loc):
        os.mkdir(loc)
    opt.QNP_WaveletTV(MaxIter_QN,A,b_noise,lamda = lamda,beta = beta,gamma = gamma,\
                      a_k = a_k,Maxsub_Iter = MaxIter_inner_wavTV_set[iter],W=W,\
                      save=loc,SaveIter= True,verbose = verbose,original=im_original,\
                      TV_bound=TV_bound,TV_type = TV_type)
