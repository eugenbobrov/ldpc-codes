import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from time import clock
from skimage.io import imshow, imread
from skimage.color import rgb2gray
from skimage.filters import  threshold_otsu
from skimage.transform import resize


class DegenerateMatrixError(Exception):
    pass
    

def make_generator_matrix(H):
    """
    Create code generator matrix using given check matrix.
    The function must raise DegenerateMatrixError exception, if given check matrix is degenerate

    :param H: check matrix of size (M, N), np.array
    :return G: generator matrix of size (N, K), np.array
    :return ind: indices for systematic coding, i.e. G[ind,:] is unity matrix
    """

    M, N = H.shape
    K = N - M
    T = H.astype(np.bool)
    G = np.zeros((N, K), np.uint8)
    jnd = np.zeros(M, np.uint16)
    m = 0
    for n in np.arange(N):
        if not T[m, n]:
            if not np.any(T[m:, n]):
                continue
            j = np.argmax(T[m:, n]) + m
            T[[m, j], :] = T[[j, m], :]
        T[m+1:, :][T[m+1:, n], :] ^=  T[m, :]
        jnd[m] = n
        m += 1
        if m == M:
            break
    if jnd[M - 1] == 0:
        raise DegenerateMatrixError
    m = M - 1
    for n in np.arange(N)[::-1]:
        if not np.any(n == jnd):
            continue
        T[:m, :][T[:m, n], :] ^= T[m, :]
        m -= 1
        if m == 0:
            break
    if m != 0:
        raise DegenerateMatrixError
    ind = list(set(np.arange(N)).difference(set(jnd)))
    G[jnd, :] = T[:, ind]
    G[ind, :] = np.eye(K)
    return G, ind
        

def update_messages_h_to_e(mu_h_to_e, mu_e_to_h, s, fi, vi, ovi, nci, trim=1e-8):
    """
    Updates messages (in place) from one factor to a set of variables.

    :param mu_h_to_e: all messages from factors to variables, 3D numpy array of size (M, N, D)
    :param mu_e_to_h: all messages from variables to factors, 3D numpy array of size (M, N, D)
    :param s: input syndroms, numpy array of size (M, D)
    :param fi: index of selected factor, a number
    :param vi: indices of all variables than are connected to factor
    :param ovi: indices of variable for updated messages
    :param nci: indices of syndromes for updated messages
    :param trim: trim value for updated messages
    """
    vi, ovi, nci = np.array(vi, np.intp), np.array(ovi, np.intp), np.array(nci, np.intp)    
    
    delta_pk_brace = 1 - 2*mu_e_to_h[fi, vi[:, None], nci[None, :]]
    delta_pl = np.prod(delta_pk_brace, axis=0)[None, :]
    delta_pl = delta_pl/(1 - 2*mu_e_to_h[fi, ovi[:, None], nci[None, :]])
    pl = (1 - delta_pl)/2
    
    res = s[fi, nci]*(1 - pl) + (1 - s[fi, nci])*pl
    mu_h_to_e[fi, ovi[:, None], nci[None, :]] = np.minimum(np.maximum(res, trim), 1 - trim)        
    

def update_messages_e_to_h_and_beliefs(mu_e_to_h, beliefs, mu_h_to_e, q,
                                       vi, fi, ofi, nci, trim=1e-8):
    """
    Updates messages (in place) from one variable to a set of factors and updates belief
    for this variable.
    
    :param mu_e_to_h: all messages from variables to factors, 3D numpy array of size (M, N, D)
    :param beliefs: all beliefs, numpy array of size (N, D)
    :param mu_h_to_e: all messages from factors to variables, 3D numpy array of size (M, N, D)
    :param q: channel error probability
    :param vi: index of selected variable, a number
    :param fi: indices of all factors that are connected to selected variable
    :param ofi: indices of factors for updated messages
    :param nci: indices of syndromes for updated messages
    :param trim: trim value for updated messages
    """
    fi, ofi, nci = np.array(fi, np.intp), np.array(ofi, np.intp), np.array(nci, np.intp)
    
    mu = mu_h_to_e[fi[:, None], vi, nci[None, :]]    
    b1 = np.prod(mu, axis=0)*q
    b0 = np.prod(1 - mu, axis=0)*(1 - q)
    beliefs[vi, nci] = b1/(b1 + b0)

    muo = mu_h_to_e[ofi[:, None], vi, nci[None, :]]
    m1 = b1[None, :]/muo
    m0 = b0[None, :]/(1 - muo)
        
    res = m1/(m1 + m0)
    mu_e_to_h[ofi[:, None], vi, nci[None, :]] = np.minimum(np.maximum(res, trim), 1 - trim)


def decode(s, H, q, schedule='parallel', max_iter=100, tol_beliefs=1e-4, display=False):
    """
    LDPC decoding procedure for syndrome probabilistic model.
    
    :param s: a set of syndromes, numpy array of size (M, D)
    :param H: LDPC check matrix, numpy array of size (M, N)
    :param q: channel error probability
    :param schedule: a schedule, possible values are 'parallel' and 'sequential'
    :param max_iter: maximal number of iterations
    :param tol_beliefs: tolerance for beliefs stabilization
    :param display: verbosity level
    :return e: decoded error vectors, numpy array of size (N, D)
    :return results: additional results, a dictionary with fields:
        'num_iter': number of iterations for each syndrome decoding, numpy array of length D
        'status': status (0, 1, 2) for each syndrome decoding, numpy array of length D
    """

    H = np.array(H, np.uint8)
    M, N = H.shape
    D = s.shape[1] if len(s.shape) > 1 else 1
    num_iter = np.zeros(D, np.uint8)
    status = np.ones(D, np.uint8)*2
    beliefs = q*np.ones((N, D))
    hat_e = beliefs >= 0.5
    mu_e_to_h = q*np.ones((M, N, D))*H[:, :, None]
    mu_h_to_e = np.zeros_like(mu_e_to_h)
    nci = np.arange(D)
    
    if display:
            print("start decoding")
            print("num_iter : status : nci")
            
    for j in np.arange(max_iter):

        if len(nci) > 0: 
            num_iter[nci] += 1
        else: 
            return hat_e, {'num_iter':num_iter, 'status':status}
        beliefs_old = np.copy(beliefs)
        
        if schedule=='parallel':
            for m in np.arange(M):
                update_messages_h_to_e(mu_h_to_e, mu_e_to_h, s, m,
                    np.arange(N)[H[m, :]==True], np.arange(N)[H[m, :]==True], nci)
            for n in np.arange(N):
                update_messages_e_to_h_and_beliefs(mu_e_to_h, beliefs, mu_h_to_e, q, n,
                    np.arange(M)[H[:, n]==True], np.arange(M)[H[:, n]==True], nci)
                    
        if schedule=='sequential':
            for n in np.arange(N):
                for m in np.arange(M):
                    update_messages_h_to_e(mu_h_to_e, mu_e_to_h, s, m, 
                        np.arange(N)[H[m, :]==True], [n], nci)
                update_messages_e_to_h_and_beliefs(mu_e_to_h, beliefs, mu_h_to_e, q, n,
                    np.arange(M)[H[:, n]==True], np.arange(M)[H[:, n]==True], nci)
                    
        if display:
            print(num_iter, "     ", status, "     ", len(nci))
            
        np.save(str(j + 1), hat_e.ravel())
        
        hat_e = beliefs >= 0.5
        ci = np.all(np.dot(H, hat_e[:, nci]) % 2 == s[:, nci], axis=0)
        status[nci[ci]] = 0
        nci = nci[ci==False]
        
        ci = np.all(np.abs(beliefs_old[:, nci] - beliefs[:, nci]) < tol_beliefs, axis=0)
        status[nci[ci]] = 1  
        nci = nci[ci==False]
        
    return hat_e, {'num_iter':num_iter, 'status':status}
    

def estimate_errors(H, q, D=100, display=False, schedule='parallel'):
    """
    Estimate error characteristics for given LDPC code
    
    :param H: LDPC check matrix, numpy array of size (m, n)
    :param q: channel error probability
    :param D: number of Monte Carlo simulations
    :param display: verbosity level
    :param schedule: decoding procedure, possible values are 'sequential' and 'parallel'
    :return err_bit: mean bit error, a number
    :return err_block: mean block error, a number
    :return diver: mean divergence, a number
    """
    
    M, N = H.shape    
    e = npr.choice(2, (N, D), True, (1 - q, q)).astype(np.uint8)
    s = np.dot(H, e) % 2
    hat_e, results = decode(s, H, q, schedule=schedule, display=display)
    if display:
        print(e)
        print(hat_e.astype(np.uint8))
    diver = np.mean(results['status'] == 2)
    err_bit = np.mean((e != hat_e)[:, results['status'] < 2])
    err_block = np.mean(np.any((e != hat_e)[:, results['status'] < 2], axis=0))
    return err_bit, err_block, diver
   

def make_test_image(path="spoon.png"):
    im = imread(path)
    gray = rgb2gray(im)
    b = gray < threshold_otsu(gray)
    return resize(b, output_shape=(30, 30)).astype(np.uint8)   
   

def make_matrices(M, N, display=False, num_factor=10):
    
    flag = True
    while flag:
        try:
            t = clock()
            H = np.zeros((M, N), np.uint8)
            mask = npr.choice(M, (num_factor, N), replace=True)
            for n in range(H.shape[1]): 
                H[mask[:, n], n] = 1
            G, ind = make_generator_matrix(H)
            flag = False
            if display:
                print("H made by: ", clock() - t, "sec")
        except DegenerateMatrixError:
            if display:
                print("failed")
    return H, G, ind   
   
   
def experiment_r_n_err(D=100, q=0.05, display=False, schedule='parallel'):
    
    arrayR = np.arange(0.1, 0.9, 0.1)
    arrayN = np.array((1e2, 1e3), np.uint32)
    lenR, lenN = len(arrayR), len(arrayN)
    
    err_bit, err_block = np.empty((lenR, lenN)),np.empty((lenR, lenN))
    
    for i, R in enumerate(arrayR):
        for j, N in enumerate(arrayN):
            K = np.uint32(R*N)
            M = N - K
            H, _, _ = make_matrices(M, N, display=display)           
            err_bit[i, j], err_block[i, j] = (
                estimate_errors(H, q, D=D, display=display, schedule=schedule))
    
    plt.plot(arrayR[:, None]*np.ones((lenR, lenN)), err_bit)
    plt.xlabel("code speed R")
    plt.ylabel("mean bit error")
    plt.legend(("N=100", "N=1000"))
    plt.show()
    plt.savefig("speed_mean_bit_error.svg")
    
    plt.plot(arrayR[:, None]*np.ones((lenR, lenN)), err_block)
    plt.xlabel("code speed R")
    plt.ylabel("mean block error")
    plt.legend(("N=100", "N=1000"))
    plt.show()
    plt.savefig("speed_mean_block_error.svg")       
    

def decode_image(im):
    
    plt.figure(figsize=(5, 5))
    imshow(im==False)
    plt.axis("off")
    plt.savefig("base.png")
    plt.show()
    u = np.ravel(im)
    K = u.shape[0]
    N = 4*K
    M = N - K
    print(M, N)
    H, G, ind = make_matrices(M, N, num_factor=11)
    print("H, G was made, shapes:", H.shape, G.shape)
    v = np.dot(G, u) % 2
    plt.figure(figsize=(4, 12))
    imshow(v.reshape((4*np.sqrt(K).astype(np.uint8)),
                     (np.sqrt(K)).astype(np.uint8))==False)
    plt.axis("off")
    plt.savefig("start.png")
    plt.show()
    q = 0.05
    e = npr.choice(2, N, True, (1 - q, q)).astype(np.uint8)
    print("noise probability:", q, "|mistakes of im:",
          np.sum(e[ind]), "|all mistakes:", np.sum(e))
    w = v ^ e
    plt.figure(figsize=(5,5))
    print(K)
    imshow(w[ind].reshape(np.sqrt(K).astype(np.uint8),
           np.sqrt(K).astype(np.uint8))==False)
    plt.axis("off")
    plt.savefig("noised.png")
    plt.show()
    plt.figure(figsize=(4, 12))
    imshow(w.reshape((4*np.sqrt(K).astype(np.uint8)),
                     (np.sqrt(K)).astype(np.uint8))==False)
    plt.axis("off")
    plt.savefig("0.png")
    plt.show()
    s = np.dot(H, w) % 2
    hat_e, results = decode(s[:, None], H, q, display=True)
    print(results)
    for j in np.arange(results['num_iter'][0]):
        hat_e = np.load(str(j + 1) + ".npy")
        hat_v = w ^ hat_e
        plt.figure(figsize=(4, 12))
        imshow(hat_v.reshape((4*np.sqrt(K).astype(np.uint8)),
                             (np.sqrt(K)).astype(np.uint8))==False)
        plt.axis('off')
        plt.savefig(str(j + 1) + ".png",)
        plt.show()
    
    print("num_iter:", results['num_iter'], "status:", results['status'])
    print("all mistakes:", np.sum(e), "|mistakes of im:", np.sum(e[ind]), "|missed all mistakes:", 
          np.sum(e ^ hat_e), "|missed mistakes of im:", np.sum(e[ind] ^ hat_e[ind]))
    hat_u = hat_v[ind]
    hat_im = hat_u.reshape(im.shape)
    plt.figure(figsize=(5, 5))
    imshow(hat_im==False)
    plt.axis('off')
    plt.savefig("end.png")
    plt.show()