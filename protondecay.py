#the matrix of the yukawa couplings at high temperature are called h,f,g respectively for Y_10, Y_126 and Y_120.


#check parity of pion channel
#from reference 1506.08468

from channels import pionchannel, knuchannel
from running import breakingchain
from yukawa.yukawamatrices import YukawaMatrices
import numpy as np

import optparse

op = optparse.OptionParser(usage=__doc__)
opts, args = op.parse_args() 

mp= 0.93828 #GeV
mpi0 = 0.1349768 #GeV
mK0 = 0.497611 #GeV
K0usRuLpe = 0.1033 #GeV^2
K0usLuLpe = 0.0572 #GeV^2
Pi0udLuLp = 0.1345 #GeV^2  
Pi0udRuL = -0.1314 
KpusLdL = 0.041 #GeV^2
MGUT = 9e15
MSUSY = 10**6
####define all the matrices here in function of the free parameters
def protondecaypchannel(h,f,g,Yd,Yu,Ye,x,y,Uup,Uel,Udown,alpha321SM_MZ,alpha321SM_MSUSY,alpha321MSSM_MSUSY,alpha321MSSM_M1,alpha3221_M1, alpha3221_MX):
    #checkparity here!
    OhL = pionchannel.OhudL(h,f,g,Yd,Yu,Ye,x,y,Uup,Uel,Udown,MGUT)
    
    OhR = pionchannel.OhudR(h,f,g,Yd,Yu,Ye,x,y,Uup,Uel,Udown,MGUT)
    Ow= pionchannel.OW(h,f,g,Yd,Yu,Ye,x,y,Uup,Uel,Udown,alpha321SM_MZ,MGUT)	 
    AS1 = breakingchain.AS1_modelMSSM_LRMSSM_GUT(alpha321SM_MZ,alpha321SM_MSUSY,alpha321MSSM_MSUSY,alpha321MSSM_M1,alpha3221_M1, alpha3221_MX)
    AS2 = breakingchain.AS2_modelMSSM_LRMSSM_GUT(alpha321SM_MZ,alpha321SM_MSUSY,alpha321MSSM_MSUSY,alpha321MSSM_M1,alpha3221_M1, alpha3221_MX) 
    ALSUSY = breakingchain.AL
    
    Kin = (mp/(32*np.pi))*((1-(mpi0**2/mp**2))**2)*(ALSUSY**2) 
    ChContr = (AS1**2)*(OhL) +(AS2**2)*(OhR) +(Ow)*(AS2**2) #check left and right here!!
    HadrFact = Pi0udLuLp
    Gamma = Kin*ChContr*HadrFact*(1.52e+24/3.17058e-08)
     
    return Gamma

def protondecayknuchannel(h,f,g,Yd,Yu,Ye,x,y,Uup,Unu,Udown,alpha321SM_MZ,alpha321SM_MSUSY,alpha321MSSM_MSUSY,alpha321MSSM_M1,alpha3221_M1, alpha3221_MX):
    #check if you considered all contribute
    OhusRuL = knuchannel.OhusRuL(h,f,g,Yd,Yu,Ye,x,y,Uup,Unu,Udown,MGUT)
    
    OhusLuL = knuchannel.OhusLuL(h,f,g,Yd,Yu,Ye,x,y,Uup,Unu,Udown,MGUT)
    Ow= knuchannel.OW(h,f,g,Yd,Yu,Ye,x,y,Uup,Unu,Udown,alpha321SM_MZ,MGUT)	 
    AS1 = breakingchain.AS1_modelMSSM_LRMSSM_GUT(alpha321SM_MZ,alpha321SM_MSUSY,alpha321MSSM_MSUSY,alpha321MSSM_M1,alpha3221_M1, alpha3221_MX)
    AS2 = breakingchain.AS2_modelMSSM_LRMSSM_GUT(alpha321SM_MZ,alpha321SM_MSUSY,alpha321MSSM_MSUSY,alpha321MSSM_M1,alpha3221_M1, alpha3221_MX) 
    ALSUSY = breakingchain.AL
    
    Kin = (mp/(32*np.pi))*((1-(mK0**2/mp**2))**2)*(ALSUSY**2) 
    #print(Kin)
    ChContr = (AS2**2)*(OhusRuL)*(K0usRuLpe**2) +(AS1**2)*(OhusLuL)*(K0usLuLpe**2) +(Ow)*(AS1**2)*(KpusLdL**2)
    #print(Kin*ChContr)
    #ChContr = (Ow)*(AS1**2)*(K0usLuLpe**2)
    Gamma = Kin*ChContr*(1.52e+24/3.17058e-08)  ##GeV->Year
     
    return Gamma
 
def load_parameter2():
    points = np.loadtxt("TEST160123/GUTFIT.txt")
    #oldscan = np.loadtxt("ParaTable_10.txt") ##good for testing
    #N = len(points[0,:]);
    likelihood = points[:,1]
    Parameters = points[:,2:]
    #print(x[-1])
    return Parameters, likelihood, points
def load_test():
    points = np.loadtxt("arc_chi_100.txt")
    #oldscan = np.loadtxt("ParaTable_10.txt") ##good for testing
    
    #print(x[-1])
    return points  

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
    TabsParams = load_test()
    #import data and create BrChParams
    Results = []
    alpha321SM_MZ,alpha321SM_MSUSY, alpha321MSSM_MSUSY, alpha321MSSM_M1,alpha3221MSSM_M1,alpha3221MSSM_MX = load_couplings()
    OptKnuchannel = False
    FreeHiggsMix = [1,0.01,1]
    x = [1,1,1,1,1,1,1,1,1,1,1]########play with these values to see how the lifetime changes, very interesting is studying the parameter V16
    y = [1,1,1,1,1,1,1,1,1,1,1]
   # print(len(TabsParams[:,0]))
    for i in range(len(TabsParams[:,0])):
       #print(TabsParams[i,:])     
       YUK = YukawaMatrices(TabsParams[i,:],FreeHiggsMix)
       h = YUK.matrix_Y10()
       f = YUK.matrix_Y126()
       g = YUK.matrix_Y120()
       Yu = YUK.matrix_Yu()
       Yd = YUK.matrix_Yd()
       Ye = YUK.matrix_Ye()
       #msel, Vl = np.linalg.eigh(np.conjugate(np.transpose(Ye))@Ye)
       #print(g)
       Uup = YUK.matrix_Uup()
       Udown = YUK.matrix_Udown()    
       Uel = YUK.matrix_Uel()
       #add diag matrix for neutrino
       Unu = YUK.matrix_Unu() 
       #call yukawa maatrices here with those parameters as input
       #lifetime = 1/decayrate
       if OptKnuchannel:
           decayrate = protondecayknuchannel(h, f, g, Yd, Yu, Ye, x, y, Uup, Unu, Udown, alpha321SM_MZ, alpha321SM_MSUSY, alpha321MSSM_MSUSY, alpha321MSSM_M1, alpha3221MSSM_M1, alpha3221MSSM_MX)
       else: 
           decayrate =protondecaypchannel(h, f, g, Yd, Yu, Ye, x, y, Uup, Unu, Udown, alpha321SM_MZ, alpha321SM_MSUSY, alpha321MSSM_MSUSY, alpha321MSSM_M1, alpha3221MSSM_M1, alpha3221MSSM_MX)
       Results.append(1/decayrate)
    
    Results = np.array(Results)
    
    
    
    if OptKnuchannel:
        np.savetxt("PDknuchannel_test.txt",Results)
    else:
        np.savetxt("PDpchannel_test.txt",Results)
        


