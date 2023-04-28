#the matrix of the yukawa couplings at high temperature are called h,f,g respectively for Y_10, Y_126 and Y_12


#THIS program has to: compute PD, redefine all the GUTFIT para and GUTFIT obs with the same order
#Throw away 1 every 10 points + being parallelized should take 10/20 minutes instead of 8 hours to run the programs
 
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
betaH = 0.015
####define all the matrices here in function of the free parameters
def protondecaypchannel(h,f,g,Yd,Yu,Ye,x,y,Uup,Uel,Udown,alpha321SM_MZ,alpha321SM_MSUSY,alpha321MSSM_MSUSY,alpha321MSSM_M1,alpha3221_M1, alpha3221_MX):
    #checkparity here!
    #I don't know if we need to consider it
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

def protondecayknuchannel(h,f,g,Yd,Yu,Ye,x,y,Uup,Unu,Udown,alpha321SM_MZ,alpha321SM_MSUSY,alpha321MSSM_MSUSY,alpha321MSSM_M1,alpha3221_M1, alpha3221_MX,mwino,MSUSY):
    #check if you considered all contribute
    #OhusRuL = knuchannel.OhusRuL(h,f,g,Yd,Yu,Ye,x,y,Uup,Unu,Udown,MGUT)
    
    Oh = knuchannel.Oh(h,f,g,Yd,Yu,Ye,x,y,Uup,Unu,Udown,MGUT,mwino,MSUSY)
    Ow= knuchannel.OW(h,f,g,Yd,Yu,Ye,x,y,Uup,Unu,Udown,alpha321SM_MZ,MGUT,mwino,MSUSY)	 
    AS1 = breakingchain.AS1_modelMSSM_LRMSSM_GUT(alpha321SM_MZ,alpha321SM_MSUSY,alpha321MSSM_MSUSY,alpha321MSSM_M1,alpha3221_M1, alpha3221_MX)
    AS2 = breakingchain.AS2_modelMSSM_LRMSSM_GUT(alpha321SM_MZ,alpha321SM_MSUSY,alpha321MSSM_MSUSY,alpha321MSSM_M1,alpha3221_M1, alpha3221_MX) 
    ALSUSY = breakingchain.AL
    
    Kin = (mp/(32*np.pi))*((1-(mK0**2/mp**2))**2)*(ALSUSY**2) 
    #print(Kin)
    ChContr = (AS2**2)*(Oh)*(betaH**2) +(Ow)*(AS1**2)*(betaH**2)
    #print(Kin*ChContr)
    #ChContr = (Ow)*(AS1**2)*(K0usLuLpe**2)
    Gamma = Kin*ChContr*(1.52e+24/3.17058e-08)  ##GeV->Year
     
    return Gamma
 
def load_parameter2():
    #points = np.loadtxt("s/GUTFIT.txt"%args[0])
    #oldscan = np.loadtxt("ParaTable_10.txt") ##good for testing
    #N = len(points[0,:]);
    Parameters = np.loadtxt("test2404/GUTFITpara.txt")
    likelihood = np.loadtxt("test2404/GUTFITlikelihood.txt")

    #print(x[-1])
    return Parameters, likelihood
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
    TabsParams, likelihood = load_parameter2()
    #import data and create BrChParams
    Results = []
    lifetime = []
    alpha321SM_MZ,alpha321SM_MSUSY, alpha321MSSM_MSUSY, alpha321MSSM_M1,alpha3221MSSM_M1,alpha3221MSSM_MX = load_couplings()
    print("N. of points:")
    print(len(likelihood))
    FreeHiggsMix = [1,1,1]
    x = [1,1,1,1,1,1,1,1,1,1,1]########play with these values to see how the lifetime changes, very interesting is studying the parameter V16
    y = [1,1,1,1,1,1,1,1,1,1,1]
   # print(len(TabsParams[:,0]))
    mwino = 1000
    #MSUSY = 10e+6
    #len(TabsParams[:,0])
    for i in range(len(likelihood)):
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
       lifetime = []
       print("Point started:")
       print(i)
       #print(lifetime)
       for j in range(15):
           MSUSY = 10**(4+(j*(5))/14) 
           lifetime.append(1/protondecayknuchannel(h, f, g, Yd, Yu, Ye, x, y, Uup, Unu, Udown, alpha321SM_MZ, alpha321SM_MSUSY, alpha321MSSM_MSUSY, alpha321MSSM_M1, alpha3221MSSM_M1, alpha3221MSSM_MX,mwino,MSUSY))
       print("Point finished")    
           #print("lifetime:")
           #print(lifetime)
       
       Results.append(lifetime)
       
       
       #print("Results:")
       #print(Results[0])
       #lifetime.clear()
    #print(Results)
    #Results = np.array(Results)
    #print(Results)
    
    
    np.savetxt("PDknuchannel_test.txt",Results)
        


