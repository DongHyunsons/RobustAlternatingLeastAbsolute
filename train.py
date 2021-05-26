import pandas as pd
import numpy as np
import sys

from function import *

def main(Input):
    d = np.load('data/d.npy')
    RDmatrix = pd.read_csv('data/RDmatrix_real.csv', index_col= 0 )
    Omega = pd.read_csv('data/Omega_real.csv', index_col = 0)
    testOmega=Omega.mul(d)              # 실제로 존재하는 데이터 중에서 10퍼센트 뽑음 랜덤으로
    trainOmega=Omega.sub(testOmega)     # 실제로 존재하는 데이터 중에서 90퍼센트 뽑음
    
    if(Input == "als"):
        als = ALS(RDmatrix,trainOmega,testOmega)
        als.ALScalculate()
        als.AUC()
    
    elif(Input == "ralaBack"):
        if(len(sys.argv)>2):
            back = RALA_back(RDmatrix ,trainOmega , testOmega , sys.argv[2])
            back.RALA_calculate_back()
            back.AUC()
        else:
            back = RALA_back(RDmatrix ,trainOmega , testOmega )
            back.RALA_calculate_back()
            back.AUC()
        
    elif(Input == "ralaLP"):
        LP = RALA_LP( RDmatrix ,trainOmega , testOmega)
        LP.RALA_calculate_LP()
        LP.AUC()
        

if __name__ == "__main__":
    if(len(sys.argv) < 2 ):
        print("please Input algorithm name")
        print("python train.py als")
        print("python train.py ralaLP")
        print("python train.py ralaBack")
        
    else:
        main(sys.argv[1])
