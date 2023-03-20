####define all the anomalous dimension: 321 SM, 3221S LRMSSM, 321S MSSM S1, S2 left or right
import numpy as np
##non susy ones
gamma321S1 = [2,9/4,23/20]
gamma321S2 = [2,9/4,11/2]
 
gamma3211S1 = [2, 9/4, 3/4, 1/4]
gamma3211S2 = [2, 9/4, 3/4, 1/4]

gamma3221S1 = [2, 9/4, 9/4, 1/4] 
gamma3221S2 = [2, 9/4, 9/4, 1/4] 
## susy ones
gamma3221SS1 = [4/3, 3/2, 3/2, 1/6] 
gamma3221SS2 = [4/3, 3/2, 3/2, 1/6] 
gamma422S1 = [15/4, 9/4, 9/4]
gamma422S2 = [15/4, 9/4, 9/4]
gamma421S1 = [15/4, 9/4, 3/4] 
gamma421S2 = [15/4, 9/4, 3/4]
gamma333S1 = [2, 2, 4] 
gamma333S2 = [2, 4, 2]
gamma4221S1 = [15/8, 9/4, 0, 7/8]
gamma4221S2 = [15/4, 9/4, 0, 1/2]

a321SM = [-7,-19/6,41/10]
a321 = [-3, 1, 33/5] ####MSSM
a3221 = [-3, 2, 4, 21/2]


AL = 1.2  #SUSYvalue

#implement  wino ratio, just small correction probably

##define all the beta coefficients at one loop a_i


def MZ_MSUSYS1(alpha321SM_MZ = [],alpha321SM_MSUSY = []) :
    p = 1
    for i in range(len(a321SM)):
        p = p*((alpha321SM_MZ[i]/alpha321SM_MSUSY[i])**(-(gamma321S1[i]/a321SM[i])))
    return p

def MSUSY_M1S1(alpha321MSSM_MSUSY,alpha321MSSM_M1):
    p = 1
    for i in range(len(a321)):
        p = p*((alpha321MSSM_MSUSY[i]/alpha321MSSM_M1[i])**(-(gamma321S1[i]/a321[i])))
    return p

def M1_MXS1(alpha3221_M1,alpha3221_MX):
    p = 1
    for i in range(len(a3221)):
        p = p*((alpha3221_M1[i]/alpha3221_MX[i])**(-(gamma3221S1[i]/a3221[i])))
    return p


def MZ_MSUSYS2(alpha321SM_MZ,alpha321SM_MSUSY):
    p = 1
    for i in range(len(a321SM)):
        p = p*((alpha321SM_MZ[i]/alpha321SM_MSUSY[i])**(-(gamma321S2[i]/a321SM[i])))
    return p

def MSUSY_M1S2(alpha321MSSM_MSUSY,alpha321MSSM_M1):
    p = 1
    for i in range(len(a321)):
        p = p*((alpha321MSSM_MSUSY[i]/alpha321MSSM_M1[i])**(-(gamma321S2[i]/a321[i])))
    return p

def M1_MXS2(alpha3221_M1,alpha3221_MX):
    p = 1
    for i in range(len(a3221)):
        p = p*((alpha3221_M1[i]/alpha3221_MX[i])**(-(gamma3221S2[i]/a3221[i])))
    return p
        

def AS1_modelMSSM_LRMSSM_GUT(alpha321SM_MZ,alpha321SM_MSUSY,alpha321MSSM_MSUSY,alpha321MSSM_M1,alpha3221MSSM_M1, alpha3221MSSM_MX):
    as1 = MZ_MSUSYS1(alpha321SM_MZ,alpha321SM_MSUSY)*MSUSY_M1S1(alpha321MSSM_MSUSY,alpha321MSSM_M1)*M1_MXS1(alpha3221MSSM_M1,alpha3221MSSM_MX)
    return as1     
    

def AS2_modelMSSM_LRMSSM_GUT(alpha321SM_MZ,alpha321SM_MSUSY,alpha321MSSM_MSUSY,alpha321MSSM_M1,alpha3221MSSM_M1, alpha3221MSSM_MX):
    as2 = MZ_MSUSYS2(alpha321SM_MZ,alpha321SM_MSUSY)*MSUSY_M1S2(alpha321MSSM_MSUSY,alpha321MSSM_M1)*M1_MXS2(alpha3221MSSM_M1,alpha3221MSSM_MX)
    return as2  
    
    

def load_couplings():
    alpha321SM_MZ = np.loadtxt("../alpha321SM_MZ.txt")
    alpha321SM_MSUSY = np.loadtxt("../alpha321SM_MSUSY.txt")
    alpha321MSSM_MSUSY = np.loadtxt("../alpha321MSSM_MSUSY.txt")
    alpha321MSSM_M1 = np.loadtxt("../alpha321MSSM_M1.txt")
    alpha3221MSSM_M1 = np.loadtxt("../alpha3221MSSM_M1.txt")
    alpha3221MSSM_MX = np.loadtxt("../alpha3221MSSM_MX.txt")
    return alpha321SM_MZ,alpha321SM_MSUSY, alpha321MSSM_MSUSY, alpha321MSSM_M1,alpha3221MSSM_M1,alpha3221MSSM_MX
    
    
if __name__ == "__main__": 
    alpha321SM_MZ,alpha321SM_MSUSY, alpha321MSSM_MSUSY, alpha321MSSM_M1,alpha3221MSSM_M1,alpha3221MSSM_MX = load_couplings()
    as1 = MZ_MSUSYS1(alpha321SM_MZ,alpha321SM_MSUSY)*MSUSY_M1S1(alpha321MSSM_MSUSY,alpha321MSSM_M1)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
