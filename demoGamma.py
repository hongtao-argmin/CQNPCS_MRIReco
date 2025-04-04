# Test the choice of \gamma in QNP
# we test on the wavelet+TV regularization.

SampleTrj = 'Spiral' #'Radial' # 

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
# nortmalize A
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

MaxIter_QN = 30
TV_type = 'l1'
TV_bound = 'Dirchlet'

a_k = 1
gamma_wavTV_set = np.array([1.25,1.35,1.5,1.55,1.6,1.65,1.7,2,2.5,3])

MaxIter_inner_QN = 20
verbose = True



if SampleTrj == 'Radial':
    if image_type == 'knee':
        folderName = 'results/Radial/Knee/Gamma'
    elif image_type == 'brain':
        folderName = 'results/Radial/Brain/Gamma'
elif SampleTrj == 'Spiral':
    if image_type == 'knee':
        folderName = 'results/Spiral/Knee/Gamma'
    elif image_type == 'brain':
        folderName = 'results/Spiral/Brain/Gamma'

if not os.path.exists(folderName):
    os.mkdir(folderName)
        
psnr_set = []
ssim_set = []
    

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
for iter in range(gamma_wavTV_set.size):
    algName = '/QNPDual_TV_Wavelet'
    loc = folderName+algName
    if not os.path.exists(loc):
        os.mkdir(loc)
    loc = loc+'/'+str(gamma_wavTV_set[iter])
    if not os.path.exists(loc):
        os.mkdir(loc)
    opt.QNP_WaveletTV(MaxIter_QN,A,b_noise,lamda = lamda,beta = beta,gamma = gamma_wavTV_set[iter],\
                  a_k = a_k,Maxsub_Iter = MaxIter_inner_QN,W=W,\
       save=loc,SaveIter= True,verbose = verbose,original=im_original,\
       TV_bound=TV_bound,TV_type = TV_type)
