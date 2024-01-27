# demo for the spiral and radial trj. Reco.
import scipy.io
import numpy as np
import sigpy as sp
import optalgTao as opt #import optimization algorithms
import matplotlib.pyplot as plt

# choose trajectory type
SampleTrj = 'Spiral' #'Radial'

if SampleTrj == 'Spiral':
    trj_file = "data/spiral/trj.npy" # trajectory
    mps_file = "data/spiral/mps.npy" # sensitivity map
elif SampleTrj == 'Radial':
    trj_file = 'data/radial/trj.npy'
    mps_file = 'data/radial/mps.npy'
    
# load data
image_type = 'brain' # 'knee'
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

A_w = F * S * W.H #  includes wavelet transform for only wavelet regularization
A = F * S         # exclude the wavelet transform
LL_w = sp.app.MaxEig(A_w.H*A_w,dtype=np.complex_).run()
LL = sp.app.MaxEig(A.H*A,dtype=np.complex_).run()# * 1.01#A.N
    
# normalize A and A_w
A_w = np.sqrt(1/LL_w)*A_w
A = np.sqrt(1/LL)*A
im = im_real+1j*im_imag
im = im/np.max(np.abs(im))

im_original = im
#folder to save the results -- you may need to creat the folder first before you run the code
if image_type == 'knee':
    folderName = 'results/Spiral/Knee'
elif image_type == 'brain':
    folderName = 'results/Spiral/Brain'
# save the GT image.
np.save("%s/GTReal.npy" % folderName, np.real(im_original))
np.save("%s/GTImag.npy" % folderName, np.imag(im_original))

# formulate k-space and add complex Gaussian noise
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

# define the maximal number of iterations.
MaxIter_QN = 20
MaxIter_FISTA = 20
MaxIter_PD = 40
MaxIter_QN_Smooth = 40
MaxIter_FISTA_Smooth = 40

# define the type of TV and the boundary condition.
# The brain and knee image in smaller than the FOV 
# that the use of different boundary conditions will not affect the results.
TV_type = 'l1'
TV_bound = 'Dirchlet'
a_k = 1

# define gamma for QN
gamma_wav = 1.7
gamma_wavTV = 1.7
gamma_wavTVSmooth = 1.7 

MaxIter_inner_FISTA = 20 
MaxIter_inner_QN = 20
verbose = True


rho = 1 # parameter for ADMM
MaxCG_Iter = 20 # for ADMM

eta = 1.1 # parameter for line search
mu = 1e-5 # smoothing parmeter


# the trade-off parameters
if image_type == 'knee':
    lamda = 2e-3
elif image_type == 'brain':
    lamda = 1e-3

# only wavelet regularization
# run primal dual
algName = '/PD_Wavelet'
loc = folderName+algName
x_PD_Wav,cost_set_PD_Wav,psnr_set_PD_Wav,ssim_set_PD_Wav,CPUTime_set_PD_Wav = \
opt.PrimalDual_RegL1(MaxIter_PD, A_w, b_noise,save=loc,isOrth = True,\
                         W=W,TR_off = lamda,SaveIter=True,\
                         original=im_original,verbose=verbose)   
 
# run FISTA 
algName = '/fista_Wavelet'
loc = folderName+algName
proxg = sp.prox.L1Reg(A_w.ishape,lamda)
x_FISTA_Wav,cost_set_FISTA_Wav,psnr_set_FISTA_Wav,ssim_set_FISTA_Wav,CPUTime_set_FISTA_Wav = \
opt.FISTA_RegL1(MaxIter_FISTA, A_w, b_noise, proxg,save=loc,isOrth = True,\
                         W=W,TR_off = lamda,SaveIter=True,\
                         original=im_original,verbose=verbose)

# run QNP
algName = '/QNP_Wavelet'
loc = folderName+algName
x_QNP_Wav,cost_set_QNP_Wav,psnr_set_QNP_Wav,ssim_set_QNP_Wav,CPUTime_set_QNP_Wav = \
opt.QNP_RegL1(MaxIter_QN,A_w,b_noise,W=W,TR_off=lamda,\
               gamma = gamma_wav,a_k=a_k,verbose=verbose,\
               save=loc,original=im_original)


if image_type == 'knee':
    lamda = 5e-4
    beta = 5e-4
elif image_type == 'brain':
    if SampleTrj == 'Spiral':
        lamda = 5e-4
    elif SampleTrj == 'Radial':
        lamda = 1e-4
    beta = 5e-4

    
# wavelet+TV part
# run primal dual 
algName = '/PD_TV_Wavelet'
loc = folderName+algName
x_PD_Wav_TV,cost_set_PD_Wav_TV,psnr_set_PD_Wav_TV,ssim_set_FISTA_PD_TV,CPUTime_set_PD_Wav_TV= \
opt.PrimalDual_WaveletTV(MaxIter_PD,A,b_noise,lamda=lamda,beta = beta,
                    TV_bound = TV_bound,verbose = verbose,
    save=loc,Maxsub_Iter = MaxIter_inner_FISTA,W=W,SaveIter=True,\
    original=im_original,TV_type = TV_type)
# run FISTA
algName = '/FISTADual_TV_Wavelet'
loc = folderName+algName
x_FISTA_Wav_TV,cost_set_FISTA_Wav_TV,psnr_set_FISTA_Wav_TV,ssim_set_FISTA_Wav_TV,CPUTime_set_FISTA_Wav_TV= \
opt.FISTA_WaveletTV(MaxIter_FISTA,A,b_noise,lamda=lamda,beta = beta,
                    TV_bound = TV_bound,verbose = verbose,
    save=loc,Maxsub_Iter = MaxIter_inner_FISTA,W=W,SaveIter=True,\
    original=im_original,TV_type = TV_type)
    
# run QNP
algName = '/QNPDual_TV_Wavelet'
loc = folderName+algName
x_set_QNP_Wav_TV,cost_set_QNP_Wav_TV,psnr_set_QNP_Wav_TV,ssim_set_QNP_Wav_TV,CPUTime_set_QNP_Wav_TV = \
opt.QNP_WaveletTV(MaxIter_QN,A,b_noise,lamda = lamda,beta = beta,gamma = gamma_wavTV,\
                  a_k = a_k,Maxsub_Iter = MaxIter_inner_QN,W=W,\
       save=loc,SaveIter= True,verbose = verbose,original=im_original,\
       TV_bound=TV_bound,TV_type = TV_type)
# run ADMM for wavelet and TV based reconstruction
algName = '/ADMM_TV_Wavelet'
loc = folderName+algName
x_set_ADMM_Wav_TV,cost_set_ADMM_Wav_TV,psnr_set_ADMM_Wav_TV,ssim_set_ADMM_Wav_TV,CPUTime_set_ADMM_Wav_TV = \
opt.ADMM_WaveletTV(MaxIter_FISTA,A,b_noise,W,lamda=lamda,beta=beta,rho=rho,\
                   TV_bound=TV_bound,MaxCG_Iter = MaxCG_Iter,verbose = verbose,
    save=loc,SaveIter=True,original=im_original)

# smooth version
# run FISTA for smooth-wavelet and TV vased reconstruction
algName = '/FISTASmooth_TV_Wavelet'
loc = folderName+algName
x_set_FISTA_Wav_TV_Smooth,cost_set_FISTA_Wav_TV_Smooth,\
    psnr_set_FISTA_Wav_TV_Smooth,ssim_set_FISTA_Wav_TV_Smooth,\
        CPUTime_set_FISTA_Wav_TV_Smooth,cost_set_FISTA_Wav_TV_Smooth_true = \
opt.FISTA_WaveletTV_Smooth(MaxIter_FISTA_Smooth, A, b_noise,W=W,L_f = 1,eta = eta,mu = mu,Num_iter = MaxIter_inner_FISTA,\
                       TV_bound = TV_bound,TV_type = TV_type,lamda = lamda,beta=beta,\
         save=loc, verbose = verbose,\
         original=im_original,SaveIter=True)

# run QNP for smooth-wavelet and TV vased reconstruction
algName = '/QNPSmooth_TV_Wavelet'
loc = folderName+algName
x_set_QNP_Wav_TV_Smooth,cost_set_QNP_Wav_TV_Smooth,psnr_set_QNP_Wav_TV_Smooth,ssim_set_QNP_Wav_TV_Smooth,CPUTime_psnr_set_QNP_Wav_TV_Smooth,cost_set_QNP_Wav_TV_Smooth_true= \
opt.QNP_WaveletTV_Smooth(MaxIter_QN_Smooth, A, b_noise,gamma=gamma_wavTVSmooth,W=W,L_f=1,eta=eta,mu=mu,Num_iter = MaxIter_inner_QN,\
                       TV_bound = TV_bound,TV_type = TV_type,lamda = lamda,beta=beta,\
         save=loc, verbose = verbose,\
         original=im_original,SaveIter=True)    
    

# -------------------------------------------------------
# plot results here
# -------------------------------------------------------
