"""
we define TV utilities function here.
Forward and backward operator.
Complex version

"""
import numpy as np

# functions for TV proximal part
def Lforward(P_1,P_2,TV_bound='Dirchlet',isComplex = False):
    m2,n2 = P_1.shape
    m1,n1 = P_2.shape
    if TV_bound=='Neumann':
        if n2!=n1+1:
            print('dimensions are not consistent\n')
        if m1!=m2+1:
            print('dimension are not consistent\n')
        m = m2+1
        n = n2
        if isComplex:
            X = np.zeros((m,n),dtype=np.complex_)
        else:
            X = np.zeros((m,n))
        X[0:m-1,:] = P_1
        X[:,0:n-1] = X[:,0:n-1]+P_2
        X[1:m,:] = X[1:m,:]-P_1
        X[:,1:n] = X[:,1:n]-P_2
    elif TV_bound=='Dirchlet':
        m = m2
        n = n2
        if isComplex:
            X = np.zeros((m,n),dtype=np.complex_)
        else:
            X = np.zeros((m,n))
        X[0:m-1,:] = P_1[0:m-1,:]
        X[:,0:n-1] = X[:,0:n-1]+P_2[:,0:n-1]
        X[1:m,:] = X[1:m,:]-P_1[0:m-1,:]
        X[:,1:n] = X[:,1:n]-P_2[:,0:n-1]
        # correct boundary
        X[0:m-1,n-1] = X[0:m-1,n-1]+P_2[0:m-1,n-1]
        X[m-1,0:n-1] = X[m-1,0:n-1]+P_1[m-1,0:n-1]
        X[m-1,n-1] = X[m-1,n-1]+P_1[m-1,n-1]+P_2[m-1,n-1]
    elif TV_bound=='Periodic':
        m = m2
        n = n2
        if isComplex:
            X = np.zeros((m,n),dtype=np.complex_)
        else:
            X = np.zeros((m,n))
        X[0:m-1,:] = P_1[0:m-1,:]
        X[:,0:n-1] = X[:,0:n-1]+P_2[:,0:n-1]
        X[1:m,:] = X[1:m,:]-P_1[0:m-1,:]
        X[:,1:n] = X[:,1:n]-P_2[:,0:n-1]
        # correct boundary
        X[0:m-1,n-1] = X[0:m-1,n-1]+P_2[0:m-1,n-1]
        X[0:m-1,0] = X[0:m-1,0]-P_2[0:m-1,n-1]
        X[m-1,0:n-1] = X[m-1,0:n-1]+P_1[m-1,0:n-1]
        X[0,0:n-1] = X[0,0:n-1]-P_1[m-1,0:n-1]
        X[m-1,n-1] = X[m-1,n-1]+P_1[m-1,n-1]+P_2[m-1,n-1]
        X[0,n-1] = X[0,n-1]-P_1[m-1,n-1]
        X[m-1,0] = X[m-1,0]-P_2[m-1,n-1]
    return X

def Ltrans(X,m,n,TV_bound):
    if TV_bound == 'Neumann':
        P_1 = X[0:m-1,:]-X[1:m,:]
        P_2 = X[:,0:n-1]-X[:,1:n]
    elif TV_bound=='Dirchlet':
        P_1 =  np.vstack((X[0:m-1,:]-X[1:m,:],X[m-1,:]))
        P_2 =  np.column_stack((X[:,0:n-1]-X[:,1:n],X[:,n-1]))
    elif TV_bound=='Periodic':
        P_1 = X-np.vstack((X[1:m,:],X[0,:]))
        P_2 = X-np.column_stack((X[:,1:n-1],X[:,0]))
    return P_1,P_2

def GetGradSingle(X,TV_bound='Dirchlet',Dir='x-axis',isAdjoint = False):
     """
    Define the x/y direction gradient operation with adjoint operator.
   """
     m,n =X.shape
     if isAdjoint:
         if Dir == 'x-axis':
             if TV_bound == 'Dirchlet':
                 Dx = np.vstack((X[0,:],X[1:m-1,:]-X[0:m-2,:]))
                 Dx = np.vstack((Dx,X[m-1,:]-X[m-2,:]))
             elif TV_bound == 'Neumann':
                 Dx = np.vstack((X[0,:],X[1:m-1,:]-X[0:m-2]))
                 Dx = np.vstack((Dx,-X[m-2,:]))
             elif TV_bound == 'Periodic':
                 Dx = np.vstack((X[0,:]-X[m-1,:],X[1:m,:]-X[0:m-1,:]))
         elif Dir == 'y-axis':
             if TV_bound == 'Dirchlet':
                 Dx = np.column_stack((X[:,0],X[:,1:n-1]-X[:,0:n-2]))
                 Dx = np.column_stack((Dx,X[:,n-1]-X[:,n-2]))
             elif TV_bound == 'Neumann':
                 Dx = np.column_stack((X[:,0],X[:,1:n-1]-X[:,0:n-2]))
                 Dx = np.column_stack((Dx,-X[:,n-2]))
             elif TV_bound == 'Periodic':
                 Dx = np.column_stack((X[:,0]-X[:,n-1],X[:,1:n]-X[:,0:n-1]))
     else:
         if Dir == 'x-axis':
             if TV_bound == 'Dirchlet':
                 Dx = np.vstack((X[0:m-1,:]-X[1:m,:],X[m-1,:]))
             elif TV_bound == 'Neumann':
                 Dx = np.vstack((X[0:m-1,:]-X[1:m,:],np.zeros((1,n),dtype=np.complex_)))
             elif TV_bound == 'Periodic':
                 Dx = np.vstack((X[0:m-1,:]-X[1:m,:],X[m-1,:]-X[0,:]))
         elif Dir == 'y-axis':
             if TV_bound == 'Dirchlet':
                 Dx = np.column_stack((X[:,0:n-1]-X[:,1:n],X[:,n-1]))
             elif TV_bound == 'Neumann':
                 Dx = np.column_stack((X[:,0:n-1]-X[:,1:n],np.zeros((m,1),dtype=np.complex_)))
             elif TV_bound == 'Periodic':
                 Dx = np.column_stack((X[:,0:n-1]-X[:,1:n],X[:,n-1]-X[:,0]))
     return Dx
 
    
def proxg_TV_Complex(TR_off,v,P_1=None,P_2=None,Num_iter=50,TV_bound='Dirchlet',TV_type = 'l1',tol=1e-6):
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
    for iter in range(Num_iter):
        x_old = x_out
        P_1_old = P_1
        P_2_old = P_2
        x_out = v-TR_off*Lforward(R_1,R_2,TV_bound,isComplex=True)
        re = np.linalg.norm(x_old-x_out)/np.linalg.norm(x_out)
        if re<tol or iter == Num_iter:
            break
        Q_1,Q_2 = Ltrans(x_out,m,n,TV_bound)
        P_1 = R_1+1/(8*TR_off)*Q_1
        P_2 = R_2+1/(8*TR_off)*Q_2
        #perform project step
        if TV_type == 'iso':
            if TV_bound == 'Neumann':
                temp = np.vstack((P_1,np.zeros((1,n),dtype=np.complex_)))**2+np.column_stack((P_2,np.zeros((m,1),dtype=np.complex_)))**2
                temp = np.sqrt(np.maximum(temp,1))
                P_1 = P_1/temp[0:m-1,:]
                P_2 = P_2/temp[:,0:n-1]
            elif TV_bound == 'Dirchlet' or TV_bound == 'Periodic':
                temp = P_1**2+P_2**2
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

def Reg_Transform_WaveletTV(v,T_1,T_2,T_3,T_1_adjoint,T_2_adjoint,\
                             T_3_adjoint,z_1,z_2,z_3,L_T,\
                             Binv = None,MaxIter = 100,TV_type = 'l1',TV_bound='Dirchlet'):
    x_1_old = z_1.copy()
    x_2_old = z_2.copy()
    x_3_old = z_3.copy()
    m,n = v.shape
    T_1v = T_1(v)
    T_2v = T_2(v)
    T_3v = T_3(v)
    t_k_1 = 1
    for k in range(MaxIter):
        temp = T_1_adjoint(z_1)+T_2_adjoint(z_2)+T_3_adjoint(z_3)
        if Binv is not None:
            temp = Binv(temp)
        grad_1 = T_1(temp)-T_1v
        grad_2 = T_2(temp)-T_2v
        grad_3 = T_3(temp)-T_3v
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
        #R_1 = P_1 + (t_k-1)/t_k_1*(P_1-P_1_old)
        #R_2 = P_2 + (t_k-1)/t_k_1*(P_2-P_2_old)
        #step = (k)/(k + 5)
        z_1 = x_1_new + ((t_k-1)/t_k_1) * (x_1_new - x_1_old)
        z_2 = x_2_new + ((t_k-1)/t_k_1) * (x_2_new - x_2_old)
        z_3 = x_3_new + ((t_k-1)/t_k_1) * (x_3_new - x_3_old)
        x_1_old = x_1_new.copy()
        x_2_old = x_2_new.copy()
        x_3_old = x_3_new.copy()
        #obj.append(eval_fun(z_1,z_2,z_3))
        #obj.append(eval_fun(v-lamda*T.H*x_old))  
        #obj_2.append(eval_fun_2(x_old))
    if Binv is not None:
        x = v-Binv(T_1_adjoint(x_1_old)+T_2_adjoint(x_2_old)+T_3_adjoint(x_3_old))        
    else:
        x = v-(T_1_adjoint(x_1_old)+T_2_adjoint(x_2_old)+T_3_adjoint(x_3_old))
    return x,x_1_old,x_3_old,x_3_old


def TV_Projection(P_1,P_2,m,n,TV_bound='Dirchlet',TV_type='l1'):    
      if TV_type == 'iso':
          if TV_bound == 'Neumann':
              temp = np.vstack((P_1,np.zeros((1,n),dtype=np.complex_)))**2+np.column_stack((P_2,np.zeros((m,1),dtype=np.complex_)))**2
              temp = np.sqrt(np.maximum(temp,1))
              P_1 = P_1/temp[0:m-1,:]
              P_2 = P_2/temp[:,0:n-1]
          elif TV_bound == 'Dirchlet' or TV_bound == 'Periodic':
              temp = P_1**2+P_2**2
              temp = np.sqrt(np.maximum(temp,1))
              P_1 = P_1/temp
              P_2 = P_2/temp                            
      elif TV_type == 'l1':
          P_1 = P_1/np.maximum(np.abs(P_1),1)
          P_2 = P_2/np.maximum(np.abs(P_2),1)
      return P_1,P_2 