#the matrix of the yukawa couplings at high temperature are called h,f,g respectively for Y_10, Y_126 and Y_120.

#from reference 1506.08468

from channels import pionchannel
from running import breakingchain
from yukawa.yukawamatrices import YukawaMatrices
import numpy as np

import optparse

op = optparse.OptionParser(usage=__doc__)
opts, args = op.parse_args() 

mp= 1.0 #GeV
mpi0 = 0.136 #GeV
Pi0udLuLp = 0.1234 #GeV    #use actual numbers
MGUT = 3e15
MSUSY = 10**6
####define all the matrices here in function of the free parameters
def protondecay(h,f,g,Yd,Yu,Ye,x,y,Uup,Uel,Udown,alpha321SM_MZ,alpha321SM_MSUSY,alpha321MSSM_MSUSY,alpha321MSSM_M1,alpha3221_M1, alpha3221_MX):
    OhL = pionchannel.OhudL(h,f,g,Yd,Yu,Ye,x,y,Uup,Uel,Udown,MGUT)
    
    OhR = pionchannel.OhudR(h,f,g,Yd,Yu,Ye,x,y,Uup,Uel,Udown,MGUT)
    Ow= pionchannel.OW(h,f,g,Yd,Yu,Ye,x,y,Uup,Uel,Udown,alpha321SM_MZ,MGUT)	 
    AS1 = breakingchain.AS1_modelMSSM_LRMSSM_GUT(alpha321SM_MZ,alpha321SM_MSUSY,alpha321MSSM_MSUSY,alpha321MSSM_M1,alpha3221_M1, alpha3221_MX)
    AS2 = breakingchain.AS2_modelMSSM_LRMSSM_GUT(alpha321SM_MZ,alpha321SM_MSUSY,alpha321MSSM_MSUSY,alpha321MSSM_M1,alpha3221_M1, alpha3221_MX) 
    ALSUSY = breakingchain.AL
    
    Kin = (mp/(32*np.pi))*((1-(mpi0**2/mp**2))**2)*(ALSUSY**2)*(AS1**2) 
    ChContr = (AS1**2)*(OhL) +(AS2**2)*(OhR) +(Ow)*(AS1**2+AS2**2)
    HadrFact = Pi0udLuLp
    Gamma = Kin*ChContr*HadrFact
     
    return Gamma
 
def load_parameter2():
    points = np.loadtxt("SUSY140323/GUTFIT.txt")
    #oldscan = np.loadtxt("ParaTable_10.txt") ##good for testing
    #N = len(points[0,:]);
    likelihood = points[:,1]
    Parameters = points[:,2:]
    #print(x[-1])
    return Parameters, likelihood, points
  

def load_couplings():
    alpha321SM_MZ = np.loadtxt("alpha321SM_MZ.txt")
    alpha321SM_MSUSY = np.loadtxt("alpha321SM_MSUSY.txt")
    alpha321MSSM_MSUSY = np.loadtxt("alpha321MSSM_MSUSY.txt")
    alpha321MSSM_M1 = np.loadtxt("alpha321MSSM_M1.txt")
    alpha3221MSSM_M1 = np.loadtxt("alpha3221MSSM_M1.txt")
    alpha3221MSSM_MX = np.loadtxt("alpha3221MSSM_MX.txt")
    return alpha321SM_MZ,alpha321SM_MSUSY, alpha321MSSM_MSUSY, alpha321MSSM_M1,alpha3221MSSM_M1,alpha3221MSSM_MX
    

if __name__ == "__main__":
    ###pass the folder as first argumenti in the execution of the program
    #build a similar stuff for the gauge couplings
    #import parameters from the tab, object TabsParams
    TabsParams,Likelihood, Points = load_parameter2()
    #import data and create BrChParams
    Results = []
    alpha321SM_MZ,alpha321SM_MSUSY, alpha321MSSM_MSUSY, alpha321MSSM_M1,alpha3221MSSM_M1,alpha3221MSSM_MX = load_couplings()
    
    FreeHiggsMix = [0.1,0.1,0.1]
    x = [1,1,1,1,1,1,1,1,1,1,1]########play with these values to see how the lifetime changes, very interesting is studying the parameter V16
    y = [1,1,1,1,1,1,1,1,1,1,1]
    for i in range(100):
       YUK = YukawaMatrices(TabsParams[i,:],FreeHiggsMix)
       h = YUK.matrix_Y10()
       f = YUK.matrix_Y126()
       g = YUK.matrix_Y120()
       Yu = YUK.matrix_Yu()
       Yd = YUK.matrix_Yd()
       Ye = YUK.matrix_Ye()
       Uup = YUK.matrix_Uup()
       Udown = YUK.matrix_Udown()    
       Uel = YUK.matrix_Uel()
        
       #call yukawa maatrices here with those parameters as input
        
       decayrate = protondecay(h,f,g,Yd,Yu,Ye,x,y,Uup,Uel,Udown,alpha321SM_MZ,alpha321SM_MSUSY,alpha321MSSM_MSUSY,alpha321MSSM_M1,alpha3221MSSM_M1, alpha3221MSSM_MX)
       #lifetime = 1/decayrate
       Results.append(1/decayrate)
    
    Results = np.array(Results)
    np.savetxt("PD.txt",Results)
        
        


