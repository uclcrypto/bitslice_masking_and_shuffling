import gen_modes
import numpy as np
import sampling as rust
from joblib import Memory           
cachedir = ".cache"   
mem = Memory(cachedir,verbose=0)             


#(label,modes,mi_s estimator, mi_ms estimator)
algos = [("linear_shu_everything",gen_modes.linear_shu_everything,None,None),
            ("isw_shu_tuples",gen_modes.isw_shu_tuples,None,None),
            ("isw_shu_shares",gen_modes.isw_shu_shares,None,None),
            ("lin_m_shares",None,rust.get_mi_lin_m,rust.get_mi_lin_m_shares),
            ("lin_m_tuples",None,rust.get_mi_lin_m,rust.get_mi_lin_m_tuples),
            ("shuffling_mnew",None,rust.get_pi_shuffling_mnew,rust.get_mi_shuffling),
            ("shuffling_mac12",None,rust.get_pi_shuffling_mac12,rust.get_mi_shuffling)]

@mem.cache(ignore=["timeout_sec"]) 
def mim_mims(alg,parameters,std,precison,timeout_sec):
    """
        computes the security gain for given std

        modes_generator: a function
        parameters: tuple with parameters
        std: noise standard deviation
    """
   
    alg,modes_generator,method_mi_m,method_mi_ms = list(filter(lambda x: x[0] == alg,algos))[0]

    
    if modes_generator is not None: # modes based sampling
        matrix_perm = modes_generator(*parameters,permute=True) 
        mi_ms = rust.get_mi_modes(precison,
                        timeout_sec,
                        matrix_perm,
                        std*np.ones(matrix_perm.shape[3]))
        if mi_ms < 0:
            raise TimeoutError
        
        matrix_no_perm = modes_generator(*parameters,permute=False)
        mi_m = rust.get_mi_modes(precison,
                        timeout_sec, 
                        matrix_no_perm,
                        std*np.ones(matrix_no_perm.shape[3]))

    else:
        if len(parameters) == 2:
            eta,d = parameters
            mi_ms = method_mi_ms(precison,
                                    timeout_sec,
                                    eta,
                                    d,
                                    std)
            if mi_ms < 0:
                raise TimeoutError
            mi_m = method_mi_m(precison,
                                    timeout_sec,
                                    eta,
                                    d,
                                    std)
        else: 
            eta = parameters[0]
            mi_ms = method_mi_ms(precison,
                                    timeout_sec,
                                    eta,
                                    std)
            if mi_ms < 0:
                raise TimeoutError
            mi_m = method_mi_m(precison,
                                    timeout_sec,
                                    eta,
                                    std)
    
    if mi_m < 0:
        raise TimeoutError

    return mi_m,mi_ms

def sample_algo(alg,parameters,precison,timeout_sec,stds):
    """
        alg: algorithm to sample, can be:
               - linear_shuf_everyting
        parameters: list of tuples containing parameters
    """
    mi_m = np.zeros(len(stds))
    mi_ms = np.zeros(len(stds))
    for i,std in enumerate(stds):
        try:
            mi_m[i],mi_ms[i] = mim_mims(alg,
                                    parameters,
                                    std,
                                    precison,
                                    timeout_sec)
        except TimeoutError:
            print("\n !!!!!!! Sampling timeout after %d [sec] !!!!!!!! \n"%(timeout_sec))
            return stds[:i],mi_m[:i],mi_ms[:i]

        print("\n----> std %.4f, mi_m %.4f, mi_ms %.4f, ratio %.4f <----"%(stds[i],mi_m[i],mi_ms[i],mi_m[i]/mi_ms[i]))
    return stds,mi_m,mi_ms

