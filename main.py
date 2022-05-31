from it_analyses import sample_algo
import numpy as np
import os

stds = np.logspace(np.log10(0.1),np.log10(5),20) 
stds_unpro = np.logspace(np.log10(0.1),np.log10(10),20) 
timeout_sec = 60*60 #4*3600
precision = .5E-2
precision_unpro = .2E-2

#tuples (eta,D,precision,timeout,stds)
run = [ # Equation 13 against Algorithm 2
        ("shuffling_mnew", [(2,),(4,),(6,)], precision_unpro, timeout_sec,stds_unpro),
        # Equation 12 against Algorithm 2
        ("shuffling_mac12",[(2,),(4,),(6,)], precision_unpro, timeout_sec,stds_unpro),
        # Algorithm 3
        ("lin_m_tuples",[(2,2),(3,2),(4,2),(2,3)],precision,timeout_sec,stds),
        # Algorithm 4
        ("lin_m_shares",[(2,2),(3,2),(4,2),(2,3)],precision,timeout_sec,stds),
        # Algorithm 5
        ("linear_shu_everything",[(2,2),(2,3),(3,2)],precision,timeout_sec,stds),
        # Algorithm 6
        ("isw_shu_tuples",[(2,2)],precision,timeout_sec,stds),
        # Algorithm 7
        ("isw_shu_shares",[(2,2)],precision,timeout_sec,stds)]

DIR = "./data/"
if __name__ == "__main__":
    os.system("mkdir -p "+DIR)

    for alg, parameters_list,precision,timeout_sec,stds in run:
        res = []
        for parameters in parameters_list:
            print("##########################")
            print("# Running ",alg)
            print("# Parameters", parameters)
            print("##########################\n")

            std,mi_m,mi_ms = sample_algo(alg,
                                    parameters,
                                    precision,
                                    timeout_sec,
                                    stds)
            res.append({"param":parameters,
                            "alg":alg,
                            "std":std,"mi_m":mi_m,
                            "mi_ms":mi_ms})

            np.savez(DIR+alg+".npz",data=res,allow_pickle=True)
