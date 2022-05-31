import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os
DIR="./data/"
DIR_FIGS="./figs/"
os.system("mkdir -p "+(DIR_FIGS))

if __name__ == "__main__":
    for fname in glob(DIR+"/*.npz"):
        print("plot for:",fname)
        dic = np.load(fname,allow_pickle=True)["data"]
        plt.figure();

        # plot ratio
        plt.subplot(211)
        for curves in dic:
            if len(curves["param"]) == 2:
                eta,d = curves["param"]
                label = "eta = %d, d = %d"%(eta,d)
                ylabel = "MI_m/MI_{m+s}"
            else:
                label = "eta = %d"%(curves["param"][0])
                ylabel = "PI/MI"
            plt.semilogx(curves["std"]**2,
                        curves["mi_m"]/curves["mi_ms"],
                        marker=".",
                        alpha=.6,
                        label=label)
        plt.grid(True,which="both",ls="--")
        plt.xlabel("sigma**2")
        plt.ylabel(ylabel)
        plt.legend()

        # plot information
        plt.subplot(212)
        for curves in dic:
            if len(curves["param"]) == 2:
                eta,d = curves["param"]
                label = "eta = %d, d = %d"%(eta,d)
            else:
                label = "eta = %d"%(curves["param"][0])
           
            plt.loglog(curves["std"]**2,
                        curves["mi_ms"],
                        marker=".",
                        alpha=.6,
                        label=label)
        plt.grid(True,which="both",ls="--")
        plt.xlabel("sigma**2")
        plt.ylabel("Information")
        plt.legend()
        plt.suptitle(dic[0]["alg"])

        plt.savefig(DIR_FIGS+dic[0]["alg"]+".png")
    plt.show()
