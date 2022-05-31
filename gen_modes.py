import numpy as np
from joblib import Memory          
import itertools
cachedir = ".cache"   
mem = Memory(cachedir,verbose=0)             


@mem.cache
def isw_shu_shares(eta,D,permute):
    """
        generate all the possible modes for 
        shuffling shares applied to eta ISW
        where one mode correspond to one combination of 
        randomness.
        
        eta:        size of input vectors
        D:          masking order
        permutate:  True: shuffling + masking, 
                    False: masking only
    """
    pi = eta
    a_encs = np.array(np.meshgrid(*(np.arange(2**pi),)*((D-1)))).T.reshape(-1,(D-1))
    b_encs = np.array(np.meshgrid(*(np.arange(2**pi),)*((D-1)))).T.reshape(-1,(D-1))
    n_steps = D + D*(D-1)
    if permute:
        perms = np.array(list(itertools.permutations(np.arange(pi))),dtype=np.uint32)
    else:
        perms = np.array([np.arange(pi)],dtype=np.uint32)
    randomness = np.array(np.meshgrid(*(np.arange(2**pi),)*(D*(D-1)//2))).T.reshape(-1,D*(D-1)//2)
   
    modes = len(a_encs)*len(b_encs)*(len(perms)**n_steps)*len(randomness)

    print("There is 2**%.3f modes"%(np.log2(modes)))
    if np.log2(modes)>= 32:
        print("In cannot run 2^%f for you. You are too agressive with parameters %d %d ..."%(np.log2(modes),pi,D));
        return -1

    ndim = 3*(D)*pi + pi*13*(D)*(D-1)//2
    print(ndim)
    f = np.zeros((2**pi,2**pi,modes,ndim))
    for y_all in range(2**pi):
        for x_all in range(2**pi):
            all_combinations = (a_encs,b_encs,randomness,)+(perms,)*n_steps
            for m,comb in enumerate(itertools.product(*all_combinations)):
                a = np.zeros(D,dtype=np.uint8)
                b = np.zeros(D,dtype=np.uint8)
                c = np.zeros(D,dtype=np.uint8)

                a[:D-1] = comb[0]; a[D-1] = np.bitwise_xor.reduce(comb[0])^y_all
                b[:D-1] = comb[1]; b[D-1] = np.bitwise_xor.reduce(comb[1])^x_all

                random_tape = comb[2]
                perms_at_step = comb[3:]

                step = 0
                dim = 0
                # ISW starts here
                for i in range(D):
                    perm = perms_at_step[step]
                    for p in range(pi): 
                        ai = (a[i]>>perm[p])&0x1;
                        bi = (b[i]>>perm[p])&0x1;
                        ci = ai&bi;
                        c[i] ^= (ci << perm[p]);

                        f[y_all,x_all,m,dim] = ai; dim+=1;
                        f[y_all,x_all,m,dim] = bi; dim+=1;
                        f[y_all,x_all,m,dim] = ci; dim+=1;
                    step+=1
                step_r = 0

                tape = np.zeros(pi,dtype=np.uint8)
                for i in range(D):
                    for j in range(i+1,D):
                        perm = perms_at_step[step]
                        for p in range(pi): 
                            r = (random_tape[step_r]>>perm[p])&0x1
                            tape[perm[p]] = r;

                            f[y_all,x_all,m,dim] = r; dim+=1;
                            
                            ai = (a[i]>>perm[p])&0x1;
                            f[y_all,x_all,m,dim] = ai; dim+=1;
                            
                            bj = (b[j]>>perm[p])&0x1;
                            f[y_all,x_all,m,dim] = bj; dim+=1;
                            
                            ci = (c[i]>>perm[p])&0x1;
                            f[y_all,x_all,m,dim] = ci; dim+=1;
                            
                            aibj = ai & bj;
                            f[y_all,x_all,m,dim] = aibj; dim+=1;

                            tmp = r^aibj;
                            f[y_all,x_all,m,dim] = tmp; dim+=1;

                            ci_n = ci ^ tmp;
                            f[y_all,x_all,m,dim] = ci_n; dim+=1;
                            c[i] ^= (tmp << perm[p]);
                        
                        step +=1

                        perm = perms_at_step[step]
                        for p in range(pi): 
                            r = tape[perm[p]]; 
                            
                            ai = (a[j]>>perm[p])&0x1;
                            f[y_all,x_all,m,dim] = ai; dim+=1;
                            
                            bj = (b[i]>>perm[p])&0x1;
                            f[y_all,x_all,m,dim] = bj; dim+=1;
                            
                            ci = (c[j]>>perm[p])&0x1;
                            f[y_all,x_all,m,dim] = ci; dim+=1;
                            
                            aibj = ai & bj;
                            f[y_all,x_all,m,dim] = aibj; dim+=1;

                            tmp = r^aibj;
                            f[y_all,x_all,m,dim] = tmp; dim+=1;

                            ci_n = ci ^ tmp;
                            f[y_all,x_all,m,dim] = ci_n; dim+=1;
                            c[j] ^= (tmp << perm[p]);
                        step +=1
                        step_r +=1;

                assert np.bitwise_xor.reduce(c) == y_all & x_all

    return f 



@mem.cache
def isw_shu_tuples(eta,D,permute):
    """
        generate all the possible modes for 
        shuffling tuples applied to eta ISW
        where one mode correspond to one combination of 
        randomness.
        
        eta:        size of input vectors
        D:          masking order
        permutate:  True: shuffling + masking, 
                    False: masking only
    """
    pi = eta
    a_encs = np.array(np.meshgrid(*(np.arange(2**pi),)*((D-1)))).T.reshape(-1,(D-1))
    b_encs = np.array(np.meshgrid(*(np.arange(2**pi),)*((D-1)))).T.reshape(-1,(D-1))
    n_steps = 1 
    if permute:
        perms = np.array(list(itertools.permutations(np.arange(pi))),dtype=np.uint32)
    else:
        perms = np.array([np.arange(pi)],dtype=np.uint32)
    randomness = np.array(np.meshgrid(*(np.arange(2**pi),)*(D*(D-1)//2))).T.reshape(-1,D*(D-1)//2)
   
    modes = len(a_encs)*len(b_encs)*(len(perms)**n_steps)*len(randomness)

    print("There is 2**%.3f modes"%(np.log2(modes)))
    if np.log2(modes)>= 32:
        print("In cannot run 2^%f for you. You are too agressive with parameters %d %d ..."%(np.log2(modes),pi,D));
        return -1

    ndim = 3*(D)*pi + pi*13*(D)*(D-1)//2
    f = np.zeros((2**pi,2**pi,modes,ndim))
    for y_all in range(2**pi):
        for x_all in range(2**pi):
            all_combinations = (a_encs,b_encs,randomness,)+(perms,)*n_steps
            for m,comb in enumerate(itertools.product(*all_combinations)):
                a = np.zeros(D,dtype=np.uint8)
                b = np.zeros(D,dtype=np.uint8)
                c = np.zeros(D,dtype=np.uint8)

                a[:D-1] = comb[0]; a[D-1] = np.bitwise_xor.reduce(comb[0])^y_all
                b[:D-1] = comb[1]; b[D-1] = np.bitwise_xor.reduce(comb[1])^x_all

                random_tape = comb[2]
                perms_at_step = comb[3:]
                perm = perms_at_step[0]
                dim = 0
                # ISW starts here
                for i in range(D):
                    for p in range(pi): 
                        ai = (a[i]>>perm[p])&0x1;
                        bi = (b[i]>>perm[p])&0x1;
                        ci = ai&bi;
                        c[i] ^= (ci << perm[p]);

                        f[y_all,x_all,m,dim] = ai; dim+=1;
                        f[y_all,x_all,m,dim] = bi; dim+=1;
                        f[y_all,x_all,m,dim] = ci; dim+=1;
                step_r = 0
                tape = np.zeros(pi,dtype=np.uint8)
                for i in range(D):
                    for j in range(i+1,D):
                        for p in range(pi): 
                            r = (random_tape[step_r]>>perm[p])&0x1
                            tape[perm[p]] = r;

                            f[y_all,x_all,m,dim] = r; dim+=1;
                            
                            ai = (a[i]>>perm[p])&0x1;
                            f[y_all,x_all,m,dim] = ai; dim+=1;
                            
                            bj = (b[j]>>perm[p])&0x1;
                            f[y_all,x_all,m,dim] = bj; dim+=1;
                            
                            ci = (c[i]>>perm[p])&0x1;
                            f[y_all,x_all,m,dim] = ci; dim+=1;
                            
                            aibj = ai & bj;
                            f[y_all,x_all,m,dim] = aibj; dim+=1;

                            tmp = r^aibj;
                            f[y_all,x_all,m,dim] = tmp; dim+=1;

                            ci_n = ci ^ tmp;
                            f[y_all,x_all,m,dim] = ci_n; dim+=1;
                            c[i] ^= (tmp << perm[p]);
                        
                        for p in range(pi): 
                            r = tape[perm[p]]; 
                            
                            ai = (a[j]>>perm[p])&0x1;
                            f[y_all,x_all,m,dim] = ai; dim+=1;
                            
                            bj = (b[i]>>perm[p])&0x1;
                            f[y_all,x_all,m,dim] = bj; dim+=1;
                            
                            ci = (c[j]>>perm[p])&0x1;
                            f[y_all,x_all,m,dim] = ci; dim+=1;
                            
                            aibj = ai & bj;
                            f[y_all,x_all,m,dim] = aibj; dim+=1;

                            tmp = r^aibj;
                            f[y_all,x_all,m,dim] = tmp; dim+=1;

                            ci_n = ci ^ tmp;
                            f[y_all,x_all,m,dim] = ci_n; dim+=1;
                            c[j] ^= (tmp << perm[p]);
                        step_r +=1;

                assert np.bitwise_xor.reduce(c) == y_all & x_all

    return f 

@mem.cache                
def linear_shu_everything(eta,D,permute):
    """
        generate all the possible modes for 
        shuffling everything applied to linear layers
        where one mode correspond to one combination of 
        randomness.

        
        eta:        size of input vectors
        D:          masking order
        permutate:  True: shuffling + masking, 
                    False: masking only
    """
    a_encs = np.array(np.meshgrid(*(np.arange(2**eta),)*((D-1)))).T.reshape(-1,(D-1))
    if permute:
        perms = np.array(list(itertools.permutations(np.arange(eta*D))),dtype=np.uint32)
    else:       
        perms = np.array([np.arange(eta*D)],dtype=np.uint32)
    modes = len(a_encs)*len(perms)

    print("There is 2**%.3f modes"%(np.log2(modes))) 
    if np.log2(modes)>= 32:
        print("In cannot run 2^%f for you. You are too agressive with parameters %d %d ..."%(np.log2(modes),eta,D));
        return -1                          

    ndim = eta*D          
    f = np.zeros((2**eta,1,modes,ndim))                                                                   
    for y_all in range(2**eta):                   
        for x_all in [0]:
            all_combinations = (a_encs,)+(perms,)
            for m,comb in enumerate(itertools.product(*all_combinations)):
                a = np.zeros(D,dtype=np.uint8)
                a[:D-1] = comb[0]; 
                a[D-1] = np.bitwise_xor.reduce(comb[0])^y_all
                perm = comb[1]
                dim = 0
                for p in perm:
                    share = p%D
                    bit = p//D 
                    f[y_all,x_all,m,dim] = (a[share]>>bit)&0x1; dim+=1;
    return f 

