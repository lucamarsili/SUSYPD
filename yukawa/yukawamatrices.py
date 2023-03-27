from inputconstants import constants as cns

import numpy as np
def matrix_diag3(d1,d2,d3):
    return np.array([[d1, 0.0, 0.0], [0.0, d2, 0.0], [0.0, 0.0, d3]], dtype = np.complex64)

# Generic Rotations #
def matrix_rot23(th23):
    return np.array([[1.0,          0.0 , 0.0],
                    [0.0,  np.cos(th23), np.sin(th23)],
                     [0.0, -np.sin(th23), np.cos(th23)]],dtype = np.complex64)

def matrix_rot12(th12):
    return np.array([[ np.cos(th12), np.sin(th12), 0.0],
                    [-np.sin(th12), np.cos(th12), 0.0],
                     [          0.0,  0.0,         1.0]], dtype = np.complex64)

def matrix_rot13(th13, delta):
    return np.array([[ np.cos(th13), 0.0, np.sin(th13) * np.exp(-1j * delta)],
                    [ 0.0, 1.0, 0.0                               ],
                    [-np.sin(th13)* np.exp(1j * delta), 0.0, np.cos(th13)]],
                    dtype=np.complex64)

def matrix_vckm(th12, th13, th23, delta):
    return matrix_rot23(th23) @ matrix_rot13(th13, delta) @ matrix_rot12(th12)

# Phase Matrices #

def matrix_phase(a1, a2, a3):
    return np.array([[np.exp(1j * a1),             0.0,             0.0],
                    [            0.0, np.exp(1j * a2),             0.0],
                    [            0.0,             0.0, np.exp(1j * a3)]],
                    dtype=np.complex64)

def matrix_phase2(a1, a2):
    return np.array([[np.exp(1j * a1),             0.0,             0.0],
                    [            0.0, np.exp(1j * a2),             0.0],
                     [            0.0,             0.0,             1.0]],
                        dtype=np.complex64)
####everything is buildt from Yu and Yd, Yu is assumed to be diagonal and Yd not, therefore keep the same notation of 1506.6879 we have Udown = Vckm and Uup = Id. 
#Regarding Uel we cannot assume Yel is diagonal and therefore we need to compute Uel.

def load_parameter2():
    points = np.loadtxt("../TEST160123/GUTFIT.txt")
    predictions = np.loadtxt("../TEST160123/GUTFITpredictions.txt")
    #oldscan = np.loadtxt("ParaTable_10.txt") ##good for testing
    #N = len(points[0,:]);
    
    Parameters = points[:,2:]
    #print(x[-1])
    return Parameters, predictions

 
class YukawaMatrices: 
    def __init__(self,FreeParams = [],FreeHiggsMix = [],TanBeta = 10):
        self.sign_parameter = 0
        self.a1 = FreeParams[0]
        self.a2 = FreeParams[1]
        self.m0 = 5e-11
    
        self.r1 = FreeParams[2]
        self.r2 = FreeParams[3]
        self.cnu =FreeParams[4]
        self.ce = FreeParams[5]
        
        self.V11 = FreeHiggsMix[0] #Following Bowen's parameterization
        self.V16 = FreeHiggsMix[1] #benchmarkpoint here would provide an univola link between GW and proton decay signal!!!!! fix this == fix ratio MN3 M1
        self.V17 = FreeHiggsMix[2]
        self.yurand = 1
        self.ycrand = 1
        self.ytrand = 1
        self.ydrand = 1
        self.ysrand = 1
        self.ybrand = 1
        
        self.TanBeta = TanBeta
        
    def repdown(self,CD = []):
        self.ydrand = CD[0]
        self.ysrand = CD[1]
        self.ybrand = CD[2]



    def repup(self,CU = []):
        self.yurand = CU[0]
        self.ycrand = CU[1]
        self.ytrand = CU[2]

    def minusup(self):
        self.yurand = -self.yurand
        self.ycrand = -self.ycrand
        self.ybrand = -self.ybrand
        
    def randomsign(self):
        s = self.sign_parameter
        C1 = [+1,+1,+1]
        C2 = [+1,+1,-1]
        C3 = [+1,-1,+1]
        C4 = [-1,+1,+1]
        C1m = [-1,-1,-1]
        C2m = [-1,-1,+1]
        C3m = [-1,+1,-1]
        C4m = [+1,-1,-1]
           
        C = [C1,C2,C3,C4]
        Cm = [C1m,C2m, C3m, C4m]
        for i in range(0,32):
            if (s < (i+1)/32):
                if(i<16):
                    self.repdown(C[int(i/4)])#check if the code is right
                    self.repup(C[i%4])
                    
                    break
                else:
                    i =  i-16
                    self.repdown(C[int(i/4)])#check if the code is right
                    self.repup(Cm[i%4])
                    
                    break
                    
    def matrix_Yd(self, th12 = cns.th12BF, th13 = cns.th13BF, th23 = cns.th23BF, delta = cns.deltaBF, yd = cns.ydBF, ys = cns.ysBF, yb = cns.YbBF): 
        '''
        define the input parameters and 1. is it fine to use the Mz value? 2. does the parameterization holds also at lower scale???, which valus to use for h,f and g??
        '''
        
        Pa      = matrix_phase2(self.a1, self.a2)
        Vckm    = matrix_vckm(th12, th13, th23, delta)
        ydrand  = yd
        ysrand  = -ys
        ybrand  = yb
        Yddiag  = matrix_diag3(ydrand, ysrand, ybrand)
        Vckmc   = np.conj(Vckm)
        Yukd    = Pa @ Vckm @ Yddiag  @ np.transpose(Vckmc) @  np.conj(Pa)
        
    
        return  Yukd
    
    def matrix_Yu(self,yu = cns.yuBF,yc = cns.ycBF,yt = cns.YtBF):
        
        yurand  = yu
        ycrand  = yc
        ytrand  = -yt
       
        Yuku = matrix_diag3(yurand,ycrand,ytrand)
        return Yuku
    
    
    def matrix_Y10(self):   ###notation of the paper, still multiply by free parameters!!! we are calling it h in the other files!!!
        Yu = self.matrix_Yu()
        Yd = self.matrix_Yd()
        h = -(Yu/(self.r2-1))+(self.r2*np.real(Yd))/(self.r1*(self.r2-1))
        return h/self.V11
    
    def matrix_Y126(self):
        Yu = self.matrix_Yu()
        Yd = self.matrix_Yd()
        f = -(Yu/(self.r2-1))-(np.real(Yd))/(self.r1*(self.r2-1))
        return np.sqrt(3)*self.r1*(f/self.V16)
    def matrix_Y120(self):
        Yd = self.matrix_Yd()
        hp = -1j*np.imag(Yd)/self.r1
        return (1j*hp*self.r1*((self.ce+3)/(2*(self.ce+1))))/self.V17
    
    
    def matrix_Uup(self):
        return np.identity(3)
    
    def matrix_Udown(self):
        Vckm    = matrix_vckm(th12 = cns.th12BF, th13 = cns.th13BF, th23 = cns.th23BF, delta = cns.deltaBF)
        return Vckm
    
    def matrix_Ynu(self):
        Yd        = self.matrix_Yd()
        Yu        = self.matrix_Yu()
        ReYd      = np.real(Yd)
        ImYd      = np.imag(Yd)
        cnulogged = self.cnu
        r2logged  = self.r2
        r1logged  = self.r1
        type1p1   = (8 * r2logged * (r2logged+1) * Yu)/(r2logged-1) 
        type1p2   = -(16 * r2logged*r2logged * ReYd)/(r1logged * (r2logged-1))
        type1p3   = ((r2logged-1)/r1logged) * (r1logged * Yu + 1j * cnulogged * ImYd) @ np.linalg.inv(r1logged * Yu - ReYd) @ (r1logged * Yu - 1j * cnulogged * ImYd)
        type1     = (self.m0) * (type1p1 + type1p2 + type1p3)
        return  type1
 
    
    def matrix_Ye(self):
        Yu = self.matrix_Yu()
        Yd = self.matrix_Yd()
        part1 = -(4*self.r1*Yu)/(self.r2-1)
        part2 = ((self.r2+3)*np.real(Yd))/(self.r2-1)
        part3  = 1j*self.ce*np.imag(Yd)
        return part1 +part2+part3
    
    def matrix_Uel(self):
        Ye = self.matrix_Ye()
        Yesq = np.conjugate(np.transpose(Ye))@Ye
        mass, Vl  = np.linalg.eigh(Yesq)
        return Vl
    
    def matrix_Unu(self):
        Ynu = self.matrix_Ynu()
        Ynusq = np.conjugate(np.transpose(Ynu))@Ynu
        mass, Vl  = np.linalg.eigh(Ynusq)
        return Vl
    
    
    
if __name__ == "__main__": 
    Par, Pred = load_parameter2()
    for i in range(100):
        YUK = YukawaMatrices(Par[i,:], [1,1,1])
        msqleptons, VL = np.linalg.eigh(YUK.matrix_Ye()@np.conjugate(np.transpose(YUK.matrix_Ye())))
        print(YUK.matrix_Y10())
        print(YUK.matrix_Y120())
        print(YUK.matrix_Y126())
       # msqlnutrino, UL = np.linalg.eigh(YUK.matrix_Ye()@np.conjugate(np.transpose(YUK.matrix_Ye())))
        #print(msqleptons)
        #print(Pred[i,10:13])
        #print(YUK.matrix_Yu())
    
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

                


    










                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
