# Implementation of the used optimization algorithms.
import numpy as np
import sigpy as sp
import time
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
from scipy.optimize import fsolve

import TVutilities as TVfun

def SSIM(original,compressed):
    return ssim(original,compressed,\
                data_range=compressed.max() - compressed.min())

def PSNR(original, compressed):
    mse = np.mean((np.abs(original - compressed)) ** 2)
    if(mse == 0):  
        return 100
    # decide the scale of the image
    if np.max(np.abs(original))<1.01:
        max_pixel = 1
    else:
        max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel/np.sqrt(mse))
    return psnr

def Power_Iter_handle(A,m,n,tol = 1e-6):
    ''' 
    Power iteration to compute the maximal eigenvalue of A. 
    A is represented as a handle function.
    '''
    b_k = np.random.randn(m,n)
    Ab_k = A(b_k)
    norm_b_k = np.linalg.norm(Ab_k)
    while True:
        b_k_1 = Ab_k/norm_b_k
        if np.linalg.norm(b_k-b_k_1)<=tol:
            break
        else:
            b_k = b_k_1
        Ab_k = A(b_k_1)
        norm_b_k = np.linalg.norm(Ab_k)
    b = b_k_1
    L = np.vdot(b,Ab_k/np.vdot(b,b))
    return np.real(L)

def Reg_Transform(z,v,lamda,T,MaxIter = 100,isOrth = True,L_T=None,tol=1e-6):
    # Solve 0.5*\|x-v\|_2^2+\lambda \|TX\|_1 iteratively 
    # where T represents some transform.
    # we should know the adjoint of T if isOrth is false.
    # We solve this problem on its dual formulation.
    # z: the initial value
    if isOrth:
        L_T = 1
    else:
        if L_T is None: 
            # if L_T is unknown, we run power iteration to compute the maximal eigenvalue
            L_T = sp.app.MaxEig(T*T.H).run()
    x_old = z.copy() 
    t_k_1 = 1
    if isOrth:
        grad = lamda*z-T*v
    else:
        grad = lamda*T*(T.H*z)-T*v
    for k in range(MaxIter):
        temp = z-grad/L_T
        x_new = temp/np.maximum(np.abs(temp),1)
        re = np.linalg.norm(np.abs(x_old-x_new))/np.linalg.norm(np.abs(x_new))
        if re<tol:
            break
        t_k = t_k_1
        t_k_1 = (1+np.sqrt(1+4*t_k**2))/2
        z = x_new + ((t_k-1)/t_k_1)*(x_new - x_old)
        x_old = x_new.copy()
        if isOrth:
            grad = lamda*z-T*v
        else:
            grad = lamda*T*(T.H*z)-T*v
    x = v-lamda*T.H*x_old
    return x,x_old,L_T

def WPML1Reg(D,u,v,lamda,u_sign,size_m,size_n,tol=1e-4,alpha_0=None):
    # solve the weighted proximal mapping
    # min_x 0.5*\|x-v\|_W+lamda*\|x\|_1 where W = D + u_sign*uu'
    # size_m,size_n: the number of rows and columns of 
    if alpha_0 is None:
        alpha_0 = np.zeros(2) 
    # In our setting, D is a scalar
    u_temp = u/D
    lamda_temp = lamda/D
    # solve the nonlinear equation problem.
    def L_alpha(alpha):
        alpha_real,alpha_imag = alpha
        temp  = u.conj().T@(v-sp.thresh.soft_thresh(lamda_temp,\
                                                    v-(u_sign*[alpha_real+1j*alpha_imag]\
                                                       )*u_temp))+[alpha_real+1j*alpha_imag]
        return np.squeeze([temp.real,temp.imag])
    alpha_star = fsolve(L_alpha,alpha_0,xtol=tol)    
    return np.reshape(sp.thresh.soft_thresh(lamda_temp,v-u_sign*[alpha_star[0]+1j*alpha_star[1]]*u_temp),(size_m,size_n),order = 'F'),alpha_star

def WPMTVReg_Complex(v,TR_off,Binv=None,P_1=None,P_2=None,\
                     Num_iter=100,tol=1e-6,TV_bound='Dirchlet',TV_type = 'l1',\
                     sigma_max=None):
    # Compute the WPM iteratively.
    # Note that this function can also be used for computing
    # the classical proximal operator.
    m,n = v.shape
    if P_1 is None:
        if TV_bound == 'Neumann':
            P_1 = np.zeros((m-1,n),dtype=np.complex_)
            P_2 = np.zeros((m,n-1),dtype=np.complex_)
            R_1 = np.zeros((m-1,n),dtype=np.complex_)
            R_2 = np.zeros((m,n-1),dtype=np.complex_)   
        elif TV_bound == 'Dirchlet' or TV_bound == 'Periodic':
            P_1 = np.zeros((m,n),dtype=np.complex_)
            P_2 = np.zeros((m,n),dtype=np.complex_)
            R_1 = np.zeros((m,n),dtype=np.complex_)
            R_2 = np.zeros((m,n),dtype=np.complex_)
    else:
        R_1 = P_1
        R_2 = P_2
    t_k_1 = 1
    x_out = np.zeros((m,n),dtype=np.complex_)
    if Binv is None:
        Binv = lambda x: x
        sigma_max = 1
    elif sigma_max is None:
        sigma_max = Power_Iter_handle(Binv,m,n,tol=1e-2)
    #sigma_max = sigma_max/1.3 # ref our Nesterov paper Hong and Yavneh NLAA2022 
    for iter in range(Num_iter):
        x_old = x_out
        P_1_old = P_1
        P_2_old = P_2
        temp = TVfun.Lforward(R_1,R_2,TV_bound,isComplex=True)
        x_out = v-TR_off*Binv(temp)
        re = np.linalg.norm(x_old-x_out)/np.linalg.norm(x_out)
        if re<tol or iter == Num_iter-1:
            break 
        Q_1,Q_2 = TVfun.Ltrans(x_out,m,n,TV_bound)
        P_1 = R_1+1/(8*TR_off*sigma_max)*Q_1
        P_2 = R_2+1/(8*TR_off*sigma_max)*Q_2
        #perform project step
        if TV_type == 'iso':
            if TV_bound == 'Neumann':
                temp = np.abs(np.vstack((P_1,np.zeros((1,n),dtype=np.complex_))))**2+np.abs(np.column_stack((P_2,np.zeros((m,1),dtype=np.complex_))))**2
                temp = np.sqrt(np.maximum(temp,1))
                P_1 = P_1/temp[0:m-1,:]
                P_2 = P_2/temp[:,0:n-1]
            elif TV_bound == 'Dirchlet' or TV_bound == 'Periodic':
                temp = np.abs(P_1)**2+np.abs(P_2)**2
                temp = np.sqrt(np.maximum(temp,1))
                P_1 = P_1/temp
                P_2 = P_2/temp                            
        elif TV_type == 'l1':
            P_1 = P_1/np.maximum(np.abs(P_1),1)
            P_2 = P_2/np.maximum(np.abs(P_2),1)
        t_k = t_k_1
        t_k_1 = (1+np.sqrt(1+4*t_k**2))/2
        R_1 = P_1 + ((t_k-1)/t_k_1)*(P_1-P_1_old)
        R_2 = P_2 + ((t_k-1)/t_k_1)*(P_2-P_2_old)
    return x_out,P_1,P_2

def obj_TV(X,TV_bound,TV_type):
    # Cost value of TV 
    m,n = X.shape
    P_1,P_2 = TVfun.Ltrans(X,m,n,TV_bound) 
    if TV_type == 'iso':
        if TV_bound=='Neumann':
            D = np.zeros((m,n))
            D[0:m-1,:] = np.abs(P_1)**2
            D[:,0:n-1] = D[:,0:n-1]+np.abs(P_2)**2
            f_TV = np.sum(np.sqrt(D))
        elif TV_bound=='Dirchlet' or TV_bound=='Periodic':
            f_TV = np.sum(np.sqrt(np.abs(P_1)**2+np.abs(P_2)**2))
    elif TV_type == 'l1':
        f_TV = np.sum(np.abs(P_1))+np.sum(np.abs(P_2))
    return f_TV

#---------------------------------------------------------------------------
# functions for ADMM
def mulAX(X,A,W,rho,TV_bound):
    # mainly for solving the least square problem in ADMM
    Y = A.H*(A*X)+rho*W.H*(W*X)
    Dx = TVfun.GetGradSingle(X,TV_bound=TV_bound,Dir='x-axis',isAdjoint = False)
    Dy = TVfun.GetGradSingle(X,TV_bound=TV_bound,Dir='y-axis',isAdjoint = False)
    Y = Y+rho*(TVfun.GetGradSingle(Dx,TV_bound=TV_bound,Dir='x-axis',isAdjoint = True)+\
    TVfun.GetGradSingle(Dy,TV_bound=TV_bound,Dir='y-axis',isAdjoint = True))
    return Y

def CG_Alg(x_k,RHS,A,W,rho,TV_bound,MaxCG_Iter,tol=1e-6):
    # implement the CG algorithm to compute the least square problem in ADMM
    r_k = RHS - mulAX(x_k,A,W,rho,TV_bound)
    p_k = r_k
    for iter in range(MaxCG_Iter):
        Ap_k = mulAX(p_k,A,W,rho,TV_bound)
        alpha_k = np.vdot(r_k,r_k)/np.vdot(p_k,Ap_k)
        x_k_1 = x_k+alpha_k*p_k
        if iter<MaxCG_Iter:
            r_k_1 = r_k - alpha_k*mulAX(p_k,A,W,rho,TV_bound)
            if np.linalg.norm(r_k_1)<tol:
                break
            beta_k = np.vdot(r_k_1,r_k_1)/np.vdot(r_k,r_k)
            p_k_1 = r_k_1+beta_k*p_k
            p_k = p_k_1
            r_k = r_k_1
            x_k = x_k_1
    return x_k_1

def CG_Alg_Handle(x_k,RHS,A,MaxCG_Iter,tol=1e-6):
    # implement the CG algorithm to compute the least square problem
    # solve Ax = b
    r_k = RHS - A(x_k)
    p_k = r_k
    for iter in range(MaxCG_Iter):
        Ap_k = A(p_k)
        alpha_k = np.vdot(r_k,r_k)/np.vdot(p_k,Ap_k)
        x_k_1 = x_k+alpha_k*p_k
        if iter<MaxCG_Iter:
            r_k_1 = r_k - alpha_k*A(p_k)
            if np.linalg.norm(r_k_1)<tol:
                break
            beta_k = np.vdot(r_k_1,r_k_1)/np.vdot(r_k,r_k)
            p_k_1 = r_k_1+beta_k*p_k
            p_k = p_k_1
            r_k = r_k_1
            x_k = x_k_1
    return x_k_1
#---------------------------------------------------------------------------

def Reg_WaveletTV(v,W,z_1,z_2,z_3,lamda,beta,L_T,Binv = None,\
                  MaxIter = 100,TV_type = 'l1',TV_bound='Dirchlet',tol=1e-6):
    # solve 0.5*\|x-v\|_2^2+lamda*\|WX\|_1 + beta*\|DX\|_{TV}
    # where W and D represents the wavelet and differential opertor.
    # z_1, z_2, z_3 are the dual variables
    m,n = v.shape
    # zero initial values
    x_1_old = np.zeros((z_1.shape),dtype=np.complex_)
    x_2_old = np.zeros((m,n),dtype=np.complex_)
    x_3_old = np.zeros((m,n),dtype=np.complex_)
    
    if Binv is None:
        Binv = lambda x: x
    t_k_1 = 1
    temp_old = np.zeros((m,n),dtype=np.complex_)
    for k in range(MaxIter):
        temp = Binv(lamda*W.H*z_1+beta*TVfun.Lforward(z_2,z_3,TV_bound,isComplex = True))
        re = np.linalg.norm(temp_old-temp)/(np.linalg.norm(temp)+1e-10)
        if k>0:
            if re<tol or k == MaxIter-1:
                break
        temp_w = temp-v
        temp_old = temp
        grad_1 = W*(lamda*temp_w)
        grad_2,grad_3 = TVfun.Ltrans(beta*temp_w,m,n,TV_bound)
        
        temp_1 = z_1-grad_1/L_T
        temp_2 = z_2-grad_2/L_T
        temp_3 = z_3-grad_3/L_T
        # projection
        x_1_new = temp_1/np.maximum(np.abs(temp_1),1)
        if TV_type == 'l1':
            x_2_new = temp_2/np.maximum(np.abs(temp_2),1)
            x_3_new = temp_3/np.maximum(np.abs(temp_3),1)
        elif TV_type == 'iso':
            if TV_bound == 'Neumann':
                temp_proj = np.abs(np.vstack((temp_2,np.zeros((1,n),dtype=np.complex_))))**2+np.abs(np.column_stack((temp_3,np.zeros((m,1),dtype=np.complex_))))**2
                temp_proj = np.sqrt(np.maximum(temp_proj,1))
                x_2_new = temp_2/temp_proj[0:m-1,:]
                x_3_new = temp_3/temp_proj[:,0:n-1]
            elif TV_bound == 'Dirchlet' or TV_bound == 'Periodic':
                temp_proj = np.abs(temp_2)**2+np.abs(temp_3)**2
                temp_proj = np.sqrt(np.maximum(temp_proj,1))
                x_2_new = temp_2/temp_proj
                x_3_new = temp_3/temp_proj 
        t_k = t_k_1
        t_k_1 = (1+np.sqrt(1+4*t_k**2))/2
        z_1 = x_1_new + ((t_k-1)/t_k_1) * (x_1_new - x_1_old)
        z_2 = x_2_new + ((t_k-1)/t_k_1) * (x_2_new - x_2_old)
        z_3 = x_3_new + ((t_k-1)/t_k_1) * (x_3_new - x_3_old)
        x_1_old = x_1_new.copy()
        x_2_old = x_2_new.copy()
        x_3_old = x_3_new.copy()
    
    x = v-Binv(lamda*W.H*x_1_old+beta*TVfun.Lforward(x_2_old,x_3_old,TV_bound,isComplex = True))        
    return x,x_1_old,x_3_old,x_3_old

def FISTA_RegL1(num_iters, A, b, proxg,save=None,isOrth = False,\
                 L_T = None,Maxsub_Iter = 50,W = None,TR_off = None,\
                 original=None,SaveIter=True,verbose = True):
  """
  Solve the following optimization problem with FISTA:
  .. math:
    \min_x \frac{1}{2} \| A x - b \|_2^2 + TR_off *\|W x\|_1 
    if W is Orth. or invertable we solve the following problem instead
    \min_x \frac{1}{2} \| A W^{-1} x - b \|_2^2 + TR_off *\|x\|_1
    Then the image is W^{-1}x^*. 
    
  Inputs:
    num_iters: Maximum number of iterations.
    
    A (Linop): Forward model.
    b (Array): Measurements.
    TR_off: the trade off parameter.
    proxg (Prox): Proximal operator of g.
    verbose: print the process of the running.
    save (None or String): If specified, path to save iterations and
                           timings.
    isOrth: true if W is orthogonal that W is included in A.
    Maxsub_Iter: the number of iterations for computing the proximal operator if isOrth is fase.
    W: the chosen wavelet transform.
    L_T: the maximal eiganvalue for W'W for solving the proximal operator iteratively.
  Returns:
    x (Array): Reconstruction (we present the image style).
    and lst_cost,lst_psnr,lst_ssim,lst_time
  """
  AHb = A.H*b
  x = AHb.copy()
  z = x.copy()
  
  lst_time  = []
  lst_cost = []
  lst_psnr = []
  lst_ssim = []
  if verbose:
      pbar = tqdm(total=num_iters, desc="FISTA Sparse L1", \
                    leave=True)
  if isOrth:
      if W is None:
          print('Error: We set W is Orth. but its value not given.\n')
          pass
      lst_cost.append(0.5*np.linalg.norm(A*x-b)**2+TR_off*np.sum(np.abs(x)))
  else:
      lst_cost.append(0.5*np.linalg.norm(A*x-b)**2+TR_off*np.sum(np.abs(W*x)))
  lst_time.append(0)
  if original is not None:
      if isOrth:
          lst_psnr.append(PSNR(np.abs(original),np.abs(W.H*x)))
          lst_ssim.append(SSIM(np.abs(original),np.abs(W.H*x)))
      else:
          lst_psnr.append(PSNR(np.abs(original),np.abs(x)))
          lst_ssim.append(SSIM(np.abs(original),np.abs(x)))
  t_k_1 = 1
  if not isOrth:
      m_w,n_w = W.ishape
      zz_old = np.zeros((m_w,n_w),dtype=np.complex_)
  for k in range(num_iters):
      start_time = time.perf_counter()
      x_old = x.copy()
      x = z.copy()
      gr = A.H*A*x-AHb
      if isOrth:
          x = proxg(1, x - gr)
      else:
          x,zz_old,L_T = Reg_Transform(zz_old,x - gr,TR_off,W,MaxIter = Maxsub_Iter,isOrth = isOrth,L_T=L_T)
      t_k = t_k_1
      t_k_1 = (1+np.sqrt(1+4*t_k**2))/2
      z = x + ((t_k-1)/t_k_1)*(x - x_old)
      end_time = time.perf_counter()
      lst_time.append(end_time - start_time)
      if original is not None:
          if isOrth:
              lst_psnr.append(PSNR(np.abs(original),np.abs(W.H*x)))
              lst_ssim.append(SSIM(np.abs(original),np.abs(W.H*x)))
          else:
              lst_psnr.append(PSNR(np.abs(original),np.abs(x)))
              lst_ssim.append(SSIM(np.abs(original),np.abs(x)))
      if isOrth:
          lst_cost.append(0.5*np.linalg.norm(A*x-b)**2+TR_off*np.sum(np.abs(x)))
      else:
          lst_cost.append(0.5*np.linalg.norm(A*x-b)**2+TR_off*np.sum(np.abs(W*x)))
      if save != None:
        tp = sp.get_array_module(x)
        np.save("%s/time.npy" % save, np.cumsum(lst_time))
        np.save("%s/cost.npy" % save, lst_cost)
        if original is not None:
            np.save("%s/psnr.npy" % save, lst_psnr)
            np.save("%s/ssim.npy" % save, lst_ssim)
        if SaveIter:
            if isOrth:
                tp.save("%s/iter_%03d.npy" % (save, k), W.H*x)
            else:
                tp.save("%s/iter_%03d.npy" % (save, k), x)
      if verbose:
          pbar.set_postfix(cost="%0.5f%%" % lst_cost[-1])
          pbar.update()
          pbar.refresh()
  if isOrth:
      x = W.H*x
  if verbose:
      pbar.set_postfix(cost="%0.5f%%" % lst_cost[-1])
      pbar.close()
      print(np.cumsum(lst_time)[-1])
      print(lst_cost[-1])
  return x,lst_cost,lst_psnr,lst_ssim,np.cumsum(lst_time)

def PrimalDual_RegL1(num_iters, A, b,save=None,isOrth = False,\
                 L_T = None,Maxsub_Iter = 50,W = None,TR_off = None,\
                 original=None,SaveIter=True,verbose = True):
  """
  Solve the optimization problem with primal dual:
  .. math:
    \min_x \frac{1}{2} \| A x - b \|_2^2 + TR_off *\|W x\|_1 
    if W is following Orth. or invertable we solve the following problem instead
    \min_x \frac{1}{2} \| A W^{-1} x - b \|_2^2 + TR_off *\|x\|_1
    Then the image is W^{-1}x^*. 
    
  Inputs:
    num_iters: Maximum number of iterations.
    
    A (Linop): Forward model.
    b (Array): Measurements.
    TR_off: the trade off parameter.
    proxg (Prox): Proximal operator of g.
    verbose: print the process of the running.
    save (None or String): If specified, path to save iterations and
                           timings.
    isOrth: true if W is orthogonal that W is included in A.
    Maxsub_Iter: the number of iterations for computing the proximal operator if isOrth is fase.
    W: the chosen wavelet transform.
    L_T: the maximal eiganvalue for W'W for solving the proximal operator iteratively.
  Returns:
    x (Array): Reconstruction (we present the image style).
    and lst_cost,lst_psnr,lst_ssim,lst_time
  """
  AHb = A.H*b
  x_k = AHb.copy()
  x_k_temp = x_k
  
  lst_time  = []
  lst_cost = []
  lst_psnr = []
  lst_ssim = []
  if verbose:
      pbar = tqdm(total=num_iters, desc="Primal Dual Sparse L1", \
                    leave=True)
  if isOrth:
      if W is None:
          print('Error: We set W is Orth. but its value not given.\n')
          pass
      lst_cost.append(0.5*np.linalg.norm(A*x_k-b)**2+TR_off*np.sum(np.abs(x_k)))
  else:
      lst_cost.append(0.5*np.linalg.norm(A*x_k-b)**2+TR_off*np.sum(np.abs(W*x_k)))
  lst_time.append(0)
  if original is not None:
      if isOrth:
          lst_psnr.append(PSNR(np.abs(original),np.abs(W.H*x_k)))
          lst_ssim.append(SSIM(np.abs(original),np.abs(W.H*x_k)))
      else:
          lst_psnr.append(PSNR(np.abs(original),np.abs(x_k)))
          lst_ssim.append(SSIM(np.abs(original),np.abs(x_k)))
  if isOrth:
      L = 1
  else:
      L = np.sqrt(1+TR_off**2)
  tau = 1/L
  sigma = 1/L
  theta = 1
  m,n = x_k.shape
  if isOrth:
      p_k = A*x_k
      p_k = np.zeros(p_k.shape,dtype=np.complex_)
  else:
      p_k = A*x_k
      q_k = W*x_k
      p_k,q_k = np.zeros(p_k.shape,dtype=np.complex_),np.zeros(q_k.shape,dtype=np.complex_)
  for k in range(num_iters):
      start_time = time.perf_counter()
      if isOrth:
          p_k = (p_k+sigma*(A*x_k_temp-b))/(1+sigma)
          x_k_1 = sp.thresh.soft_thresh(TR_off*tau,x_k-tau*A.H*p_k)
          x_k_temp = x_k_1+theta*(x_k_1-x_k)
          x_k = x_k_1
      else:
          p_k = (p_k+sigma*(A*x_k_temp-b))/(1+sigma)
          q_k = q_k+sigma*(W*(TR_off*x_k_temp))
          # projection
          q_k = q_k/np.maximum(np.abs(q_k),1)
          x_k_1 = x_k-tau*(A.H*p_k+W.H*(TR_off*q_k))
          x_k_temp = x_k_1+theta*(x_k_1-x_k)
          x_k = x_k_1
      end_time = time.perf_counter()
      lst_time.append(end_time - start_time)
      if original is not None:
          if isOrth:
              lst_psnr.append(PSNR(np.abs(original),np.abs(W.H*x_k)))
              lst_ssim.append(SSIM(np.abs(original),np.abs(W.H*x_k)))
          else:
              lst_psnr.append(PSNR(np.abs(original),np.abs(x_k)))
              lst_ssim.append(SSIM(np.abs(original),np.abs(x_k)))
      if isOrth:
          lst_cost.append(0.5*np.linalg.norm(A*x_k-b)**2+TR_off*np.sum(np.abs(x_k)))
      else:
          lst_cost.append(0.5*np.linalg.norm(A*x_k-b)**2+TR_off*np.sum(np.abs(W*x_k)))
      if save != None:
        tp = sp.get_array_module(x_k)
        np.save("%s/time.npy" % save, np.cumsum(lst_time))
        np.save("%s/cost.npy" % save, lst_cost)
        if original is not None:
            np.save("%s/psnr.npy" % save, lst_psnr)
            np.save("%s/ssim.npy" % save, lst_ssim)
        if SaveIter:
            if isOrth:
                tp.save("%s/iter_%03d.npy" % (save, k), W.H*x_k)
            else:
                tp.save("%s/iter_%03d.npy" % (save, k), x_k)
      if verbose:
          pbar.set_postfix(cost="%0.5f%%" % lst_cost[-1])
          pbar.update()
          pbar.refresh()
  if isOrth:
      x = W.H*x_k
  else:
      x = x_k
  if verbose:
      pbar.set_postfix(cost="%0.5f%%" % lst_cost[-1])
      pbar.close()
      print(np.cumsum(lst_time)[-1])
      print(lst_cost[-1])
  return x,lst_cost,lst_psnr,lst_ssim,np.cumsum(lst_time)

def QNP_RegL1(num_iters,A,b,TR_off,W=None,original=None,gamma=1.35,a_k=1,save=None,\
               SaveIter=True,verbose=True):
  """
  Solve the following optimization problem using quasi-Newton proximal 
  method:
  .. math::
    \min_x \frac{1}{2} \| A x - b \|_2^2 + TR_off*\|x\|_1
    
  Assumes MaxEig(A.H * A) = 1.
  
  Inputs:
    num_iters: Maximum number of iterations.
    A (Linop): Forward model.
    b (Array): Measurements.
    TR_off: trade-off parameter.
    W: the choice of wavelet transform, if original is given, W must be given.
    gamma,a_k: scaling and step-size in QNP algorithms.
    save (None or String): If specified, path to save iterations and
                           timings. If original is given, we also save psnr and ssim.
    SaveIter: if it is true, we save all iterates. 
    verbose (Bool): Print information.
    
  Returns:
    x (Array): Reconstruction (Image itself).
    lst_cost: cost.
    lst_psnr: psnr.
    lst_ssim: ssim.
    lst_time: running time
  """
  AHb = A.H(b)
  x = AHb.copy()
  size_m,size_n = x.shape
  lst_time  = []
  lst_cost = []
  lst_psnr = []
  lst_ssim = []
  if verbose:
      pbar = tqdm(total=num_iters, desc="QNP Sparse", \
                  leave=True)
  lst_cost.append(0.5*np.linalg.norm(A*x-b)**2+TR_off*np.sum(np.abs(x)))
  lst_time.append(0)
  alpha_star = np.zeros(2)
  x_old = x.copy()
  if original is not None:
      lst_psnr.append(PSNR(np.abs(original),np.abs(W.H*x)))
      lst_ssim.append(SSIM(np.abs(original),np.abs(W.H*x)))
  for k in range(num_iters):
      start_time = time.perf_counter()
      gr = A.H*A*x-AHb
      if k==0:
          x = sp.thresh.soft_thresh(TR_off,x_old-gr) 
          gr_old = gr.copy()
      else:
          y_k = gr - gr_old
          s_k = x-x_old
          x_old = x.copy()
          gr_old = gr.copy()
          # estimate the Hessian with SR1
          y_k_dot = np.real(np.vdot(y_k,y_k))
          s_k_dot = np.real(np.vdot(s_k,s_k))
          tau_BB = y_k_dot/np.real(np.vdot(s_k,y_k))
          if tau_BB<0:
              x = x_old - gr #run ISTA
              D = 1
              Bx_inv = lambda xx: a_k*xx
          else:
              H_0 = gamma*tau_BB
              D = H_0
              temp_1 = y_k-D*s_k;
              temp_2 = np.real(np.vdot(temp_1,s_k))
              if np.abs(temp_2)<=1e-8*np.sqrt(s_k_dot)*np.linalg.norm(temp_1):
                  u = 0
                  u_sign = 0
                  Bx_inv = lambda xx: (a_k/D)*xx
              else:
                  u = temp_1.reshape((size_m*size_n,1),order='F')/np.sqrt(np.abs(temp_2))
                  u_inv = u/D
                  u_sign = np.array((np.sign(temp_2),))
                  u_u_dot = np.vdot(u_inv,u)
                  u_sign_scale = a_k*u_sign/(1+u_sign*u_u_dot)
                  Bx_inv = lambda xx: ((a_k/D)*xx-np.reshape((u_sign_scale*(u_inv.conj().T@\
                                                                  xx.reshape((size_m*size_n,1),order='F')))*u_inv,(size_m,size_n),order='F'))
          x,alpha_star = WPML1Reg(D,u,np.reshape(x_old-Bx_inv(gr_old),(size_m*size_n,1),order = 'F'),\
                                  TR_off,u_sign,size_m,size_n,alpha_0 = alpha_star)                               
      end_time = time.perf_counter()
      lst_time.append(end_time - start_time)
      lst_cost.append(0.5*np.linalg.norm(A*x-b)**2+TR_off*np.sum(np.abs(x)))
      if original is not None:
          lst_psnr.append(PSNR(np.abs(original),np.abs(W.H*x)))
          lst_ssim.append(SSIM(np.abs(original),np.abs(W.H*x)))
      if save != None:
          tp = sp.get_array_module(x)
          np.save("%s/time.npy" % save, np.cumsum(lst_time))
          np.save("%s/cost.npy" % save, lst_cost)    
          if original is not None:
                  np.save("%s/psnr.npy" % save, lst_psnr)
                  np.save("%s/ssim.npy" % save, lst_ssim)
          if SaveIter:
              tp.save("%s/iter_%03d.npy" % (save, k), W.H*x)
      if verbose:
          pbar.set_postfix(cost="%0.2f%%" % lst_cost[-1])
          pbar.update()
          pbar.refresh()
  
  x = W.H*x
  if verbose:
      pbar.set_postfix(cost="%0.2f%%" % lst_cost[-1])
      pbar.close()
      print(np.cumsum(lst_time)[-1])
      print(lst_cost[-1])
  return x,lst_cost,lst_psnr,lst_ssim,np.cumsum(lst_time)

def ADMM_WaveletTV(num_iters, A, b,W,lamda,beta,rho=1,TV_bound='Dirchlet',\
                   MaxCG_Iter = 50,verbose = True,\
                   save=None,original=None,SaveIter=True):
    
  """Unconstrained Optimization.

  Solve the following optimization problem using ADMM:
  We only implement the l1 TV.
  .. math::
    \min_x \frac{1}{2} \| A x - b \|_2^2 + \lambda \|Wx\|_1+\beta\|Dx\|_1
W: wavelet, D, differential operator 

  Inputs:
    num_iters : Maximum number of iterations.
    
    A (Linop): Forward model.
    b (Array): Measurements.
    proxg (Prox): Proximal operator of g.
    verbose: print the process of the running.
    save (None or String): If specified, path to save iterations and
                           timings.
    isIter: use iterative method to solve the proximal operator.
  Returns:
    x (Array): Reconstruction.
  """
  AHb = A.H(b)
  x = AHb.copy()
  m,n = x.shape
  U_1 = TVfun.GetGradSingle(x,TV_bound=TV_bound,Dir='x-axis',isAdjoint = False)
  U_2 = TVfun.GetGradSingle(x,TV_bound=TV_bound,Dir='y-axis',isAdjoint = False)
  U_3 = W*x
  m_1,n_1 = U_3.shape
  Z_1 = np.zeros((m,n),dtype=np.complex_)
  Z_2 = np.zeros((m,n),dtype=np.complex_)
  Z_3 = np.zeros((m_1,n_1),dtype=np.complex_)
  lst_time  = []
  lst_cost = []
  lst_psnr = []
  lst_ssim = []
  if verbose:
      pbar = tqdm(total=num_iters, desc="ADMM Wavelet+l1TV", \
                  leave=True)
  lst_cost.append(0.5*np.linalg.norm(A*x-b)**2+lamda*np.sum(np.abs(W*x))+\
                      beta*obj_TV(x,TV_bound=TV_bound,TV_type='l1'))
  lst_time.append(0)
  if original is not None:
      lst_psnr.append(PSNR(np.abs(original),np.abs(x)))
      lst_ssim.append(SSIM(np.abs(original),np.abs(x)))
  for k in range(num_iters):
      start_time = time.perf_counter()
      RHS = AHb+rho*(TVfun.GetGradSingle(U_1-Z_1,TV_bound=TV_bound,Dir='x-axis',isAdjoint = True)+
                     TVfun.GetGradSingle(U_2-Z_2,TV_bound=TV_bound,Dir='y-axis',isAdjoint = True)+
                     W.H*(U_3-Z_3))
      x = CG_Alg(x,RHS,A,W,rho,TV_bound,MaxCG_Iter)
      grad_x = TVfun.GetGradSingle(x,TV_bound=TV_bound,Dir='x-axis',isAdjoint = False)
      grad_y = TVfun.GetGradSingle(x,TV_bound=TV_bound,Dir='y-axis',isAdjoint = False)
      Wx = W*x
      U_1 = sp.thresh.soft_thresh(beta/rho,grad_x+Z_1)
      U_2 = sp.thresh.soft_thresh(beta/rho,grad_y+Z_2)
      U_3 = sp.thresh.soft_thresh(lamda/rho,Wx+Z_3)
      Z_1 = Z_1+(grad_x-U_1)
      Z_2 = Z_2+(grad_y-U_2)
      Z_3 = Z_3+(Wx-U_3)
      end_time = time.perf_counter()
      lst_time.append(end_time - start_time)
      if original is not None:
          lst_psnr.append(PSNR(np.abs(original),np.abs(x)))
          lst_ssim.append(SSIM(np.abs(original),np.abs(x)))
      lst_cost.append(0.5*np.linalg.norm(A*x-b)**2+lamda*np.sum(np.abs(Wx))+\
                      beta*obj_TV(x,TV_bound=TV_bound,TV_type='l1'))
      
      if save != None:
        tp = sp.get_array_module(x)
        np.save("%s/time.npy" % save, np.cumsum(lst_time))
        np.save("%s/cost.npy" % save, lst_cost)
        if original is not None:
            np.save("%s/psnr.npy" % save, lst_psnr)
            np.save("%s/ssim.npy" % save, lst_ssim)
        if SaveIter:
            tp.save("%s/iter_%03d.npy" % (save, k), x)
      if verbose:
          pbar.set_postfix(cost="%0.5f%%" % lst_cost[-1])
          pbar.update()
          pbar.refresh()
  if verbose:
      pbar.set_postfix(cost="%0.5f%%" % lst_cost[-1])
      pbar.close()
      print(np.cumsum(lst_time)[-1])
      print(lst_cost[-1])
  return x,lst_cost,lst_psnr,lst_ssim,np.cumsum(lst_time)

def FISTA_WaveletTV(num_iters,A, b,Pred=None,PredInv=None,lamda = 0,beta = 0,TV_bound = 'Dirchlet',TV_type = 'l1',verbose = True,
        save=None,Maxsub_Iter = 50,W=None,original=None,SaveIter=True):
    """
  Solve the following optimization problem using fast proximal gradient
  descent:
  .. math:
for \min_x \frac{1}{2} \| A x - b \|_2^2 + \lamda \|Wx\|_1 + \beta TV(x)

  Assumes MaxEig(A.H * A) = 1.
  Inputs:
    num_iters (Int): Maximum number of iterations.
    A (Linop): Forward model.
    b (Array): Measurements.
    proxg (Prox): Proximal operator of g.
    verbose: print the process of the running.
    save (None or String): If specified, path to save iterations and
                           timings.
  Returns:
    x (Array): Reconstruction.
  """
    AHb = A.H(b)
    x = AHb.copy()
    m,n = x.shape
    m_w,n_w = W.H.ishape
    z_1 = np.zeros((m_w,n_w),dtype=np.complex_)
    z_2 = np.zeros((m,n),dtype=np.complex_)
    z_3 = np.zeros((m,n),dtype=np.complex_)
    z = x.copy()
    lst_time  = []
    lst_cost = []
    lst_psnr = []
    lst_ssim = []
    if verbose:
        pbar = tqdm(total=num_iters, desc="FISTA Wavelet+TV", \
                    leave=True)
            
    L_W = np.sqrt(Power_Iter_handle(W.H*W,m,n,tol = 1e-2))
    L_T_HT = ((lamda*L_W)**2+8*beta**2)
    if Pred is not None:
        L_T_HT = L_T_HT*np.min(Pred)
        PredInv_op = lambda x: PredInv*x
    else:
        PredInv_op = lambda x: x
        
    lst_cost.append(0.5*np.linalg.norm(A*x-b)**2+\
                    lamda*np.sum(np.abs(W*x))+\
                        beta*obj_TV(x,TV_bound=TV_bound,TV_type=TV_type))
    lst_time.append(0)
    if original is not None:
        lst_psnr.append(PSNR(np.abs(original),np.abs(x)))
        lst_ssim.append(SSIM(np.abs(original),np.abs(x)))
    t_k_1 = 1
    for k in range(num_iters):
        start_time = time.perf_counter()
        x_old = x.copy()
        x = z.copy()
        gr = PredInv_op(A.H*A*x-AHb)
        x,z_1,z_2,z_3 = Reg_WaveletTV(x - gr,W,z_1,z_2,z_3,lamda,beta,L_T_HT,Binv = PredInv_op,\
                                      MaxIter = Maxsub_Iter,TV_type = TV_type,TV_bound= TV_bound)
        t_k = t_k_1
        t_k_1 = (1+np.sqrt(1+4*t_k**2))/2
        z = x + ((t_k-1)/t_k_1) * (x - x_old)
        end_time = time.perf_counter()
        lst_time.append(end_time - start_time)
        if original is not None:
            lst_psnr.append(PSNR(np.abs(original),np.abs(x)))
            lst_ssim.append(SSIM(np.abs(original),np.abs(x)))

        lst_cost.append(0.5*np.linalg.norm(A*x-b)**2+\
                        lamda*np.sum(np.abs(W*x))+\
                            beta*obj_TV(x,TV_bound=TV_bound,TV_type=TV_type))
        if save != None:
            tp = sp.get_array_module(x)
            np.save("%s/time.npy" % save, np.cumsum(lst_time))
            np.save("%s/cost.npy" % save, lst_cost)
            if original is not None:
                np.save("%s/psnr.npy" % save, lst_psnr)
                np.save("%s/ssim.npy" % save, lst_ssim)
            if SaveIter:
                tp.save("%s/iter_%03d.npy" % (save, k), x)
        if verbose:
            pbar.set_postfix(cost="%0.5f%%" % lst_cost[-1])
            pbar.update()
            pbar.refresh()
    if verbose:
        pbar.set_postfix(cost="%0.5f%%" % lst_cost[-1])
        pbar.close()
        print(np.cumsum(lst_time)[-1])
        print(lst_cost[-1])
    return x,lst_cost,lst_psnr,lst_ssim,np.cumsum(lst_time)

def PrimalDual_WaveletTV(num_iters,A, b,lamda = 0,beta = 0,TV_bound = 'Dirchlet',TV_type = 'l1',verbose = True,
        save=None,Maxsub_Iter = 50,W=None,original=None,SaveIter=True):
    """
  Solve the following optimization problem using fast proximal gradient
  descent:
  .. math:
for \min_x \frac{1}{2} \| A x - b \|_2^2 + lamda \|Wx\|_1 + beta TV(x)

  Assumes MaxEig(A.H * A) = 1.
  Inputs:
    num_iters (Int): Maximum number of iterations.
    A (Linop): Forward model.
    b (Array): Measurement.
    proxg (Prox): Proximal operator of g.
    verbose: print the process of the running.
    save (None or String): If specified, path to save iterations and
                           timings.
  Returns:
    x (Array): Reconstruction.
  """
    AHb = A.H(b)
    x_k = AHb.copy()
    x_k_temp = x_k
    m,n = x_k.shape
    lst_time  = []
    lst_cost = []
    lst_psnr = []
    lst_ssim = []
    if verbose:
        pbar = tqdm(total=num_iters, desc="Primal Dual Wavelet+TV", \
                    leave=True)
    lst_cost.append(0.5*np.linalg.norm(A*x_k-b)**2+\
                    lamda*np.sum(np.abs(W*x_k))+\
                        beta*obj_TV(x_k,TV_bound=TV_bound,TV_type=TV_type))
    Dx_op = lambda x: TVfun.GetGradSingle(beta*x,TV_bound= TV_bound,Dir='x-axis',isAdjoint = False)
    DxT_op = lambda x: TVfun.GetGradSingle(beta*x,TV_bound= TV_bound,Dir='x-axis',isAdjoint = True)
    Dy_op = lambda x: TVfun.GetGradSingle(beta*x,TV_bound= TV_bound,Dir='y-axis',isAdjoint = False)
    DyT_op = lambda x: TVfun.GetGradSingle(beta*x,TV_bound= TV_bound,Dir='y-axis',isAdjoint = True)
    Wx = lambda x: W*(lamda*x)
    WTx = lambda x: W.H*(lamda*x)
    L_W = np.sqrt(Power_Iter_handle(W.H*W,m,n,tol = 1e-2))
    L = ((lamda*L_W)**2+8*beta**2+1)
    p_k = A*x_k
    p_k = np.zeros(p_k.shape,dtype=np.complex_)
    q_1_k,q_2_k = TVfun.Ltrans(x_k,m,n,TV_bound)
    q_3_k = W*x_k
    q_1_k,q_2_k,q_3_k = np.zeros(q_1_k.shape,dtype=np.complex_),\
    np.zeros(q_2_k.shape,dtype=np.complex_),\
    np.zeros(q_3_k.shape,dtype=np.complex_)
    tau = 1/L
    sigma = 1/L
    theta = 1
    lst_time.append(0)
    if original is not None:
        lst_psnr.append(PSNR(np.abs(original),np.abs(x_k)))
        lst_ssim.append(SSIM(np.abs(original),np.abs(x_k)))
    for k in range(num_iters):
        start_time = time.perf_counter()
        p_k = (p_k+sigma*(A*x_k_temp-b))/(1+sigma)
        q_1_k = (q_1_k+sigma*Dx_op(x_k_temp))
        q_2_k = (q_2_k+sigma*Dy_op(x_k_temp))
        q_3_k = (q_3_k+sigma*Wx(x_k_temp))
        # run projection
        q_1_k,q_2_k = TVfun.TV_Projection(q_1_k,q_2_k,m,n,TV_bound=TV_bound,TV_type=TV_type) 
        q_3_k = q_3_k/np.maximum(np.abs(q_3_k),1)
        x_k_1 = x_k-tau*(A.H*p_k+DxT_op(q_1_k)+DyT_op(q_2_k)+WTx(q_3_k))
        x_k_temp = x_k_1+theta*(x_k_1-x_k)
        x_k = x_k_1
        end_time = time.perf_counter()
        lst_time.append(end_time - start_time)
        if original is not None:
            lst_psnr.append(PSNR(np.abs(original),np.abs(x_k)))
            lst_ssim.append(SSIM(np.abs(original),np.abs(x_k)))

        lst_cost.append(0.5*np.linalg.norm(A*x_k-b)**2+\
                        lamda*np.sum(np.abs(W*x_k))+\
                            beta*obj_TV(x_k,TV_bound=TV_bound,TV_type=TV_type))
        if save != None:
            tp = sp.get_array_module(x_k)
            np.save("%s/time.npy" % save, np.cumsum(lst_time))
            np.save("%s/cost.npy" % save, lst_cost)
            if original is not None:
                np.save("%s/psnr.npy" % save, lst_psnr)
                np.save("%s/ssim.npy" % save, lst_ssim)
            if SaveIter:
                tp.save("%s/iter_%03d.npy" % (save, k), x_k)
        if verbose:
            pbar.set_postfix(cost="%0.5f%%" % lst_cost[-1])
            pbar.update()
            pbar.refresh()

    if verbose:
        pbar.set_postfix(cost="%0.5f%%" % lst_cost[-1])
        pbar.close()
        print(np.cumsum(lst_time)[-1])
        print(lst_cost[-1])
    return x_k,lst_cost,lst_psnr,lst_ssim,np.cumsum(lst_time)


def QNP_WaveletTV(num_iters,A,b,lamda = 0,beta = 0,gamma = 1.6,a_k = 1,\
                  Maxsub_Iter = 100,W=None,TV_bound = 'Dirchlet',\
           save=None,SaveIter= True,verbose = True,original=None,TV_type = 'l1'):
  """Unconstrained Optimization.
  
  Solve the following optimization problem with the quasi-Newton proximal gradient
  descent:

  .. math::
    \min_x \frac{1}{2} \| A x - b \|_2^2 + \lamda \|Wx\|_1+ \beta TV(x): 
        W represents the wavelet transform.
  
  Inputs:
    num_iters (Int): Maximum number of iterations.
    ptol (Float): Percentage tolerance between iterates.
    A (Linop): Forward model.
    b (Array): Measurements.
    save (None or String): If specified, path to save iterations and
                           timings.
    verbose (Bool): Print information.

  Returns:
    x (Array): Reconstruction.
  """
  AHb = A.H(b)
  x = AHb.copy()
  size_m,size_n = x.shape
  m_w,n_w = W.H.ishape
  z_1 = np.zeros((m_w,n_w),dtype=np.complex_)
  z_2 = np.zeros((size_m,size_n),dtype=np.complex_)
  z_3 = np.zeros((size_m,size_n),dtype=np.complex_)
  lst_time  = []
  lst_cost = []
  lst_psnr = []
  lst_ssim = []
  if verbose:
      pbar = tqdm(total=num_iters, desc = "QNP Wavelet+l1TV", \
                  leave=True)
  x_old = x.copy()
  L_W = Power_Iter_handle(W.H*W,size_m,size_n,tol = 1e-2)
  L_T_HT = ((lamda*L_W)**2+8*beta**2)
  lst_cost.append(0.5*np.linalg.norm(A*x-b)**2+ lamda*np.sum(np.abs(W*x))+\
                  beta*obj_TV(x,TV_bound=TV_bound,TV_type=TV_type))
  lst_time.append(0)
  if original is not None:
      lst_psnr.append(PSNR(np.abs(original),np.abs(x)))
      lst_ssim.append(SSIM(np.abs(original),np.abs(x)))
  for k in range(num_iters):
      start_time = time.perf_counter()
      gr = A.H*A*x-AHb#P()#A.N(x) - AHb
      if k==0:
          # normal TV
          temp_grad = x-gr
          x,z_1,z_2,z_3 = Reg_WaveletTV(temp_grad,W,z_1,z_2,z_3,lamda,beta,L_T_HT,\
                                        MaxIter = Maxsub_Iter,TV_type = TV_type,TV_bound= TV_bound)
          gr_old = gr.copy()
      else:
          y_k = gr - gr_old
          s_k = x-x_old
          x_old = x.copy()
          gr_old = gr.copy()
          y_k_dot = np.real(np.vdot(y_k,y_k))
          s_k_dot = np.real(np.vdot(s_k,s_k))
          tau_BB = y_k_dot/np.real(np.vdot(s_k,y_k))
          if tau_BB<0:
              temp_grad = x_old - gr    # run ISTA
              D = 1
              L_Binv = 1
              Bx_inv = lambda x: x
          else:
              H_0 = gamma*tau_BB
              D = H_0
              temp_1 = y_k-D*s_k;
              temp_2 = np.real(np.vdot(temp_1,s_k))
              if np.abs(temp_2)<=1e-8*np.sqrt(s_k_dot)*np.linalg.norm(temp_1):
                  u = 0
                  u_sign = 0
                  Bx_inv = lambda xx: (a_k/D)*xx
                  L_Binv = a_k/D
              else:
                  u = temp_1.reshape((size_m*size_n,1),order='F')/np.sqrt(np.abs(temp_2))
                  u_inv = u/D
                  u_sign = np.array((np.sign(temp_2),))
                  u_u_dot = np.vdot(u_inv,u)
                  u_sign_scale = a_k*u_sign/(1+u_sign*u_u_dot)
                  Bx_inv = lambda xx: ((a_k/D)*xx-np.reshape((u_sign_scale*(u_inv.conj().T@\
                                                                            xx.reshape((size_m*size_n,1),order='F')))*u_inv,(size_m,size_n),order='F'))
                  temp_grad = x_old-Bx_inv(gr_old)
                  if u_sign>0:
                      L_Binv = a_k/D
                  else:
                      L_Binv = a_k*np.real(1/(D-u.conj().T@u))
          x,z_1,z_2,z_3 = Reg_WaveletTV(temp_grad,W,z_1,z_2,z_3,\
                                        lamda,beta,L_Binv*L_T_HT,\
                                            Binv = Bx_inv,\
                                                MaxIter = Maxsub_Iter,\
                                                    TV_type = TV_type,TV_bound=TV_bound)
      end_time = time.perf_counter()
      lst_time.append(end_time - start_time)
      lst_cost.append(0.5*np.linalg.norm(A*x-b)**2+ lamda*np.sum(np.abs(W*x))+\
                      beta*obj_TV(x,TV_bound=TV_bound,TV_type=TV_type))
      if verbose:
          pbar.set_postfix(cost="%0.5f%%" % lst_cost[-1])
          pbar.update()
          pbar.refresh()
      if original is not None:
          lst_psnr.append(PSNR(np.abs(original),np.abs(x)))
          lst_ssim.append(SSIM(np.abs(original),np.abs(x)))
      if save != None:
          tp = sp.get_array_module(x)
          np.save("%s/time.npy" % save, np.cumsum(lst_time))
          np.save("%s/cost.npy" % save, lst_cost)
          if original is not None:
              np.save("%s/psnr.npy" % save, lst_psnr)
              np.save("%s/ssim.npy" % save, lst_ssim)
          if SaveIter:
              tp.save("%s/iter_%03d.npy" % (save, k), x)
  if verbose:
      pbar.set_postfix(cost="%0.5f%%" % lst_cost[-1])
      pbar.close()
      print(np.cumsum(lst_time)[-1])
      print(lst_cost[-1])
  return x,lst_cost,lst_psnr,lst_ssim,np.cumsum(lst_time)


def Smooth_l1(x,mu):
    y = np.sum(np.sqrt(np.abs(x)**2+mu))
    return y

# implement the partial smooth method and use backtracking line search
def FISTA_WaveletTV_Smooth(num_iters, A, b,W=None,L_f=1,eta=1.1,mu=0.1,Num_iter = 100,\
                           TV_bound = 'Dirchlet',TV_type = 'l1',lamda = 0,beta=0,\
             save=None, verbose = True,\
             original=None,SaveIter=True):
  """Unconstrained Optimization.

  Solve the following optimization problem using FISTA:

  .. math::
    \min_x \frac{1}{2} \| A x - b \|_2^2 + \lamda\|Wx\|_1+ \beta*TV(x)
  backtracking line search is used for the choice of step-size 
  we use the partial smooth method to smooth the wavelet part.

  Inputs:
    num_iters (Int): Maximum number of iterations.
    ptol (Float): Percentage tolerance between iterates.
    A (Linop): Forward model.
    b (Array): Measurements.
    save (None or String): If specified, path to save iterations and
                           timings.
    verbose (Bool): Print information.

  Returns:
    x (Array): Reconstruction.
    and lst_cost,lst_psnr,lst_ssim,lst_time
  """
  AHb = A.H(b)
  x = AHb.copy()
  z = x.copy()
  lst_time  = []
  lst_cost = []
  lst_cost_t= []
  lst_psnr = []
  lst_ssim = []
  P_1 = None
  P_2 = None
  if verbose:
      pbar = tqdm(total=num_iters, desc="FISTA+Smooth+Wavelet+TV", \
                  leave=True)
  obj_par = lambda x: (0.5*np.linalg.norm(A*x-b)**2+lamda*Smooth_l1(W*x,mu))
  
  lst_cost.append(0.5*np.linalg.norm(A*x-b)**2+lamda*np.sum(np.abs(W*x))+\
                  beta*obj_TV(x,TV_bound,TV_type))
  lst_cost_t.append(obj_par(x)+beta*obj_TV(x,TV_bound,TV_type))
  lst_time.append(0)
  if original is not None:
      lst_psnr.append(PSNR(np.abs(original),np.abs(x)))
      lst_ssim.append(SSIM(np.abs(original),np.abs(x)))
  t_k_1 = 1
  for k in range(num_iters):
      start_time = time.perf_counter()
      x_old = x.copy()
      x = z.copy()
      wx = W*x
      grad_1_wx = wx/np.sqrt(np.abs(wx)**2+mu)
      gr = A.H*A*x-AHb +lamda*W.H*grad_1_wx
      temp_grad = x-gr/L_f
      # solve the TV proxmal operator
      while True:
          x_temp,P_1,P_2 = WPMTVReg_Complex(temp_grad,beta/L_f,P_1=P_1,P_2=P_2,\
                                            Num_iter=Num_iter,TV_bound=TV_bound,TV_type=TV_type)
          if obj_par(x_temp)>obj_par(x)+np.real(np.vdot(x_temp-x,gr))+(L_f/2)*np.linalg.norm(x_temp-x)**2:
              L_f = eta*L_f
              temp_grad = x-gr/L_f
          else:
              break
      x = x_temp   
      t_k = t_k_1
      t_k_1 = (1+np.sqrt(1+4*t_k**2))/2
      z = x + ((t_k-1)/t_k_1) * (x - x_old)
      end_time = time.perf_counter()
      lst_time.append(end_time - start_time)
      if original is not None:
          lst_psnr.append(PSNR(np.abs(original),np.abs(x)))  
          lst_ssim.append(SSIM(np.abs(original),np.abs(x)))
      lst_cost.append(0.5*np.linalg.norm(A*x-b)**2+lamda*np.sum(np.abs(W*x))+\
                      beta*obj_TV(x,TV_bound,TV_type)) 
      lst_cost_t.append(obj_par(x)+beta*obj_TV(x,TV_bound,TV_type))
      if save != None:
        np.save("%s/time.npy" % save, np.cumsum(lst_time))
        np.save("%s/cost.npy" % save, lst_cost)
        np.save("%s/cost_t.npy" % save, lst_cost_t)
        if original is not None:
            np.save("%s/psnr.npy" % save, lst_psnr)
            np.save("%s/ssim.npy" % save, lst_ssim)
        if SaveIter:
            np.save("%s/iter_%03d.npy" % (save, k), x)
      if verbose:
          pbar.set_postfix(cost="%0.5f%%" % lst_cost[-1])
          pbar.update()
          pbar.refresh()
  if verbose:
      pbar.set_postfix(cost="%0.5f%%" % lst_cost[-1])
      pbar.close()
      print(np.cumsum(lst_time)[-1])
      print(lst_cost[-1])
  return x,lst_cost,lst_psnr,lst_ssim,np.cumsum(lst_time),lst_cost_t  


def QNP_WaveletTV_Smooth(num_iters,A,b,lamda=0,beta = 0,\
                         gamma = 1.6,a_k = 1,W=None,L_f=1,eta=1.1,mu=0.1,Num_iter = 100,\
                         TV_bound = 'Dirchlet',\
           TV_type = 'l1',save=None,SaveIter= True,verbose = False,original=None):
  """Unconstrained Optimization.

  Solve the following optimization problem using quasi-Newton proximal method:
      
  .. math::
    \min_x \frac{1}{2} \| A x - b \|_2^2 + TV(x)

  Assumes MaxEig(A.H * A) = 1.  
  Inputs:
    num_iters (Int): Maximum number of iterations.
    ptol (Float): Percentage tolerance between iterates.
    A (Linop): Forward model.
    b (Array): Measurements.
    save (None or String): If specified, path to save iterations and
                           timings.
    verbose (Bool): Print information.

  Returns:
    x (Array): Reconstruction.
  """
  AHb = A.H(b)
  x = AHb.copy()
  size_m,size_n = x.shape
  lst_time  = []
  lst_cost = []
  lst_cost_t = []
  lst_psnr = []
  lst_ssim = []
  if verbose:
      pbar = tqdm(total=num_iters, desc = "QNP+Smooth+Wavelet+TV", \
                  leave=True)
  x_old = x.copy()
  obj_par = lambda x: (0.5*np.linalg.norm(A*x-b)**2+lamda*Smooth_l1(W*x,mu))
  lst_cost.append(0.5*np.linalg.norm(A*x-b)**2+lamda*np.sum(np.abs(W*x))+\
                  beta*obj_TV(x,TV_bound,TV_type))
  lst_cost_t.append(obj_par(x)+beta*obj_TV(x,TV_bound,TV_type))
  lst_time.append(0)
  if original is not None:
      lst_psnr.append(PSNR(np.abs(original),np.abs(x)))
      lst_ssim.append(SSIM(np.abs(original),np.abs(x)))
  alpha = 1
  for k in range(num_iters):
      start_time = time.perf_counter()
      wx = W*x
      grad_1_wx = wx/np.sqrt(np.abs(wx)**2+mu)
      gr = A.H*A*x-AHb +lamda*W.H*grad_1_wx
      if k==0:
          # normal TV
          temp_grad = x-gr/L_f
          while True:
              x_temp,P_1,P_2 = WPMTVReg_Complex(temp_grad,beta/L_f,Num_iter=Num_iter,TV_bound=TV_bound,TV_type=TV_type)
              if obj_par(x_temp)>obj_par(x)+np.real(np.vdot(x_temp-x,gr))+(L_f/2)*np.linalg.norm(x_temp-x)**2:
                  L_f = eta*L_f
                  temp_grad = x-gr/L_f
              else:
                  break
          x = x_temp
          gr_old = gr.copy()
      else:
          y_k = gr - gr_old
          s_k = x-x_old
          x_old = x.copy()
          gr_old = gr.copy()
          y_k_dot = np.real(np.vdot(y_k,y_k))
          tau_BB = y_k_dot/np.real(np.vdot(s_k,y_k))
          if tau_BB<0:
              x = x_old - gr# run ISTA
              D = L_f
              sigma_max = 1/L_f
              Bx_inv = lambda xx: xx              
          else:
              H_0 = gamma*tau_BB
              D = H_0
              temp_1 = y_k-D*s_k;
              temp_2 = np.real(np.vdot(temp_1,s_k))
              if np.abs(temp_2)<=1e-8*np.sqrt(y_k_dot)*np.linalg.norm(temp_1):
                  u = 0
                  u_sign = 0
                  sigma_max = 1
                  Bx_inv = lambda xx: (a_k/D)*xx
              else:
                  u = temp_1.reshape((size_m*size_n,1),order='F')/np.sqrt(np.abs(temp_2))
                  u_inv = u/D
                  u_sign = np.array((np.sign(temp_2),))
                  u_u_dot = np.vdot(u_inv,u)
                  u_sign_scale = a_k*u_sign/(1+u_sign*u_u_dot)
                  Bx_inv = lambda xx: ((a_k/D)*xx-np.reshape((u_sign_scale*(u_inv.conj().T@\
                                                                            xx.reshape((size_m*size_n,1),order='F')))*u_inv,(size_m,size_n),order='F'))
                  if u_sign>0:
                      sigma_max = a_k/D
                  else:
                      sigma_max = a_k/np.real((D-u.conj().T@u))
                  temp_grad = x_old-alpha*Bx_inv(gr_old)
          x,P_1,P_2 = WPMTVReg_Complex(temp_grad,beta,Bx_inv,\
                                       P_1,P_2,Num_iter=Num_iter,\
                                           TV_bound=TV_bound,TV_type = TV_type,\
                                               sigma_max=sigma_max)
      end_time = time.perf_counter()
      lst_time.append(end_time - start_time)

      lst_cost.append(0.5*np.linalg.norm(A*x-b)**2+lamda*np.sum(np.abs(W*x))+\
                      beta*obj_TV(x,TV_bound,TV_type))
      lst_cost_t.append(obj_par(x)+beta*obj_TV(x,TV_bound,TV_type))
      if original is not None:
          lst_psnr.append(PSNR(np.abs(original),np.abs(x)))
          lst_ssim.append(SSIM(np.abs(original),np.abs(x)))
      if verbose:
          pbar.set_postfix(cost="%0.5f%%" % lst_cost[-1])
          pbar.update()
          pbar.refresh()
      if save != None:
          tp = sp.get_array_module(x)
          np.save("%s/time.npy" % save, np.cumsum(lst_time))
          np.save("%s/cost.npy" % save, lst_cost)
          np.save("%s/cost_t.npy" % save, lst_cost_t)
          if original is not None:
              np.save("%s/psnr.npy" % save, lst_psnr)
              np.save("%s/ssim.npy" % save, lst_ssim)
          if SaveIter:
              tp.save("%s/iter_%03d.npy" % (save, k), x)
  if verbose:
      pbar.set_postfix(cost="%0.5f%%" % lst_cost[-1])
      pbar.close()
      print(np.cumsum(lst_time)[-1])
      print(lst_cost[-1])
  return x,lst_cost,lst_psnr,lst_ssim,np.cumsum(lst_time),lst_cost_t
