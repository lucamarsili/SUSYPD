import numpy as np
#define here the yukawa matrices 

#change transpose with conjugate!!!!!!!!!!!!!!!!!!!!!!1

#define the CL coefficients

#U,V muight be either the ckms and the upmns matrix, the order count
#RotateLeft is a boolean variable that take account for the two possible way of rotatin the matrices
#a,b,c,d are the indexes of h,f,g
#x,y is the array with the free parameters

def CL(x,y,h,f,g,U,V,a,b,c,d, RotateU = True,RotateV = True,L = True):
    
    #rotate the matrices in the mass basis
    if RotateU: 
        hU = np.dot(h,U)
        fU = np.dot(f,U)
        gU= np.dot(g,U)
    else:
        hU = np.dot(U,h)
        fU = np.dot(U,f)
        gU= np.dot(U,g)
    if RotateV: 
        hV = np.dot(h,V)
        fV = np.dot(f,V)
        gV= np.dot(g,V)
    else:
        hV = np.dot(V,h)
        fV = np.dot(V,f)
        gV= np.dot(V,g)
    #consider all the possible combinations for both x and y following 0606.08648, then different choices on x and y 
    #will allow us to distringuish between left and right
    part1R = x[0]*hU[a][b]*hV[c][d] + x[1]*fU[a][b]*fV[c][d] + x[2]*gU[a][b]*gV[c][d] + x[3]*hU[a][b]*fV[c][d] + x[4]*fU[a][b]*hV[c][d] +x[5]*fU[a][b]*gV[c][d] 
    part2R = x[6]*gU[a][b]*fV[c][d] + x[7]*hU[a][b]*gV[c][d] + x[8]*gU[a][b]*hV[c][d] + x[9]*fU[a][d]*gV[b][c] + x[10]*gU[a][d]*gV[b][c] 
    part1L = x[0]*hU[a][b]*hV[c][d] + x[1]*fU[a][b]*fV[c][d] - x[3]*hU[a][b]*fV[c][d] - x[4]*fU[a][b]*hV[c][d] +y[5]*fU[a][b]*gV[c][d]
    part2L = y[7]*hU[a][b]*gV[c][d] + y[9]*gU[a][c]*fV[b][d] + y[10]*gU[a][c]*gV[b][d]
    
    if L: 
        return part1L+ part2L 
    else:
        return part2R + part1R

###############################################################################################################################################
#neutral wino channel 

def CW0A(h,f,g,x,y,Uup,Unu,Udown,nu):
    SumDiagrA =0
    RotateUup = False
    RotateUdown = False
    SumDiagrA = SumDiagrA +CL(x,y,h,f,g,np.conjugate(Uup),np.conjugate(Udown),0,0,1,nu,RotateUup,RotateUdown)-CL(x,y,h,f,g,np.conjugate(Uup),np.conjugate(Udown),0,1,0,nu,RotateUup,RotateUdown)                                                                                                        
    return SumDiagrA


def CW0B(h,f,g,x,y,Uup,Unu,Udown,nu):
    SumDiagrB =0
    RotateUnu = True
    RotateUdown = True
    SumDiagrB = SumDiagrB + CL(x,y,h,f,g,np.conjugate(np.transpose(Udown)),np.conjugate(np.transpose(Unu)),0,0,1,nu,RotateUdown,RotateUnu)-CL(x,y,h,f,g,np.conjugate(np.transpose(Udown)),np.conjugate(np.transpose(Unu)),0,1,0,nu,RotateUdown,RotateUnu)                                                                                                        
    return SumDiagrB

############################################################################################################
#charged wino channel
def CIW(h,f,g,x,y,Uup,Unu,Udown,nu):
    SumDiagr = 0 
    RotateUdown = False
    RotateUup = False
    for l in range(3):
        #j = 0 null contribute!!
        #j = 1 ip = 0 i = 0 null contribute
        #j = 1 ip = 0 i = 2
        SumDiagr = SumDiagr + CL(x,y,h,f,g,np.conjugate(Udown),np.conjugate(Uup),2,1,0,l,RotateUdown,RotateUup)*Udown[2][0]*Unu[l][nu]
        SumDiagr = SumDiagr + CL(x,y,h,f,g,np.conjugate(Udown),np.conjugate(Uup),0,2,1,l,RotateUdown,RotateUup)*Udown[2][0]*Unu[l][nu]
        SumDiagr = SumDiagr + CL(x,y,h,f,g,np.conjugate(Udown),np.conjugate(Uup),1,0,2,l,RotateUdown,RotateUup)*Udown[2][0]*Unu[l][nu]
        
        SumDiagr = SumDiagr - CL(x,y,h,f,g,np.conjugate(Udown),np.conjugate(Uup),1,2,0,l,RotateUdown,RotateUup)*Udown[2][0]*Unu[l][nu]
        SumDiagr = SumDiagr - CL(x,y,h,f,g,np.conjugate(Udown),np.conjugate(Uup),0,1,2,l,RotateUdown,RotateUup)*Udown[2][0]*Unu[l][nu]
        SumDiagr = SumDiagr - CL(x,y,h,f,g,np.conjugate(Udown),np.conjugate(Uup),2,0,1,l,RotateUdown,RotateUup)*Udown[2][0]*Unu[l][nu]
    return (1/2)*SumDiagr

        
def CIVW(h,f,g,x,y,Uup,Unu,Udown,nu):
    SumDiagr = 0 
    RotateUdown = True
    RotateUnu = True 
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if(k<j):
                    if (j ==0):
                        SumDiagr = SumDiagr + CL(x,y,h,f,g,np.conjugate(np.transpose(Udown)),np.conjugate(np.transpose(Unu)),i,j,k,nu,RotateUdown,RotateUnu)*Udown[i][1]*Uup[k][0] - CL(x,y,h,f,g,Udown,Unu,i,k,j,nu,RotateUdown,RotateUnu)*Udown[i][1]*Uup[k][0]
                    if (j ==1): 
                        SumDiagr = SumDiagr + CL(x,y,h,f,g,np.conjugate(np.transpose(Udown)),np.conjugate(np.transpose(Unu)),i,j,k,nu,RotateUdown,RotateUnu)*Udown[i][0]*Uup[k][0] - CL(x,y,h,f,g,Udown,Unu,i,k,j,nu,RotateUdown,RotateUnu)*Udown[i][0]*Uup[k][0]
    return -(1/2)*SumDiagr



########################################################################################################################################



#neutral higgsino

def CIIIh0(h,f,g,Yd,Yu,Ye,x,y,Uup,Unu,Udown,nu):
    SumDiagrj2 = 0
    SumDiagrudR =0
    RotateUdown = True
    RotateUnu  = True
    
    for i in range(3): 
        for k in range(3): 
            if (k!= 1):
                SumDiagrj2 = SumDiagrj2 + CL(x,y,h,f,g,np.conjugate(np.transpose(Udown)),np.conjugate(np.transpose(Unu)),i,1,k,nu,RotateUdown,RotateUnu)*(np.conjugate(np.transpose(Yu))[i][0])*(np.conjugate(np.transpose(Yd))[k][0]) 
                SumDiagrj2 = SumDiagrj2 - CL(x,y,h,f,g,np.conjugate(np.transpose(Udown)),np.conjugate(np.transpose(Unu)),i,k,1,nu,RotateUdown,RotateUnu)*(np.conjugate(np.transpose(Yu))[i][0])*(np.conjugate(np.transpose(Yd))[k][0])
            if (k!= 0):
                SumDiagrudR = SumDiagrudR + CL(x,y,h,f,g,np.conjugate(np.transpose(Udown)),np.conjugate(np.transpose(Unu)),i,0,k,nu,RotateUdown,RotateUnu)*(np.conjugate(np.transpose(Yu))[i][0])*(np.conjugate(np.transpose(Yd))[k][1]) 
                SumDiagrudR = SumDiagrudR - CL(x,y,h,f,g,np.conjugate(np.transpose(Udown)),np.conjugate(np.transpose(Unu)),i,k,0,nu,RotateUdown,RotateUnu)*(np.conjugate(np.transpose(Yu))[i][0])*(np.conjugate(np.transpose(Yd))[k][1])
    return SumDiagrj2, SumDiagrudR #j2 = usR




    

#charged higgsino

def CIIIhpm(h,f,g,Yd,Yu,Ye,x,y,Uup,Unu,Udown,nu):
    SumDiagr = 0 
    RotateUdown = True
    RotateUnu  = True
    ###include only i' = 1 where it goes to contribute to udRuL I dono't know about the other channel, very straightforward to include it if it is the case
    for i in range(3):
        for k in range(3): 
            if (k!= 1):
                SumDiagr = SumDiagr + CL(x,y,h,f,g,np.conjugate(np.transpose(Udown)),np.conjugate(np.transpose(Unu)),i,1,k,nu,RotateUdown,RotateUnu)*(np.conjugate(np.transpose(Yd))[i][0])*(np.conjugate(np.transpose(Yu))[k][0]) 
                SumDiagr = SumDiagr - CL(x,y,h,f,g,np.conjugate(np.transpose(Udown)),np.conjugate(np.transpose(Unu)),i,k,1,nu,RotateUdown,RotateUnu)*(np.conjugate(np.transpose(Yd))[i][0])*(np.conjugate(np.transpose(Yu))[k][0])
    return -SumDiagr


def CIVhpm(h,f,g,Yd,Yu,Ye,x,y,Uup,Unu,Udown, nu,L =False): 
    SumDiagr = 0 
    RotateUdown = False
    RotateUup  = False
    ###only weird channel with j = 1 and ip=0, whatabout parity in the nucleus????
    #i = 2
    for l in range(3):
        SumDiagr = SumDiagr + CL(x,y,h,f,g,np.conjugate(np.transpose(Udown)),np.conjugate(np.transpose(Uup)),2,1,0,l,RotateUdown,RotateUup,L)*Yu[2][0]*Ye[l][nu]
        SumDiagr = SumDiagr + CL(x,y,h,f,g,np.conjugate(np.transpose(Udown)),np.conjugate(np.transpose(Uup)),0,2,1,l,RotateUdown,RotateUup,L)*Yu[2][0]*Ye[l][nu]
        SumDiagr = SumDiagr + CL(x,y,h,f,g,np.conjugate(np.transpose(Udown)),np.conjugate(np.transpose(Uup)),1,0,2,l,RotateUdown,RotateUup,L)*Yu[2][0]*Ye[l][nu]
    
        SumDiagr = SumDiagr - CL(x,y,h,f,g,np.conjugate(np.transpose(Udown)),np.conjugate(np.transpose(Uup)),1,2,0,l,RotateUdown,RotateUup,L)*Yu[2][0]*Ye[l][nu]
        SumDiagr = SumDiagr - CL(x,y,h,f,g,np.conjugate(np.transpose(Udown)),np.conjugate(np.transpose(Uup)),0,1,2,l,RotateUdown,RotateUup,L)*Yu[2][0]*Ye[l][nu]
        SumDiagr = SumDiagr - CL(x,y,h,f,g,np.conjugate(np.transpose(Udown)),np.conjugate(np.transpose(Uup)),2,0,1,l,RotateUdown,RotateUup,L)*Yu[2][0]*Ye[l][nu]
    return SumDiagr 
        
        
        

####################defining the functions passed to the proton decay, we have Ow, OhudR and OhudL

def OW(h,f,g,Yd,Yu,Ye,x,y,Uup,Unu,Udown,alpha321SM_MZ,MGUT, mwino, MSUSY):
    #mwino = 1000
    #MSUSY = 1000000
    Diagr = []
    for nu in range(3):
        Diagr.append(CIW(h,f,g,x,y,Uup,Unu,Udown,nu)+ CIVW(h, f, g, x, y, Uup, Unu, Udown,nu) + CW0A(h, f, g, x, y, Uup, Unu, Udown,nu) + CW0B(h, f, g, x, y, Uup, Unu, Udown,nu))
    Diagr = np.asarray(Diagr)
    #Op = ((1j*alpha321SM_MZ[1])/4*np.pi)*(1/MGUT)*(mwino/(MSUSY**2))*Diagr
    #print(Diagr)
    #print(Op)
    #print(np.conjugate(Op)*Op)
    #return CW0B(h, f, g, x, y, Uup, Unu, Udown)
    return sum (np.abs(((1j*alpha321SM_MZ[1])/4*np.pi)*(1/MGUT)*(mwino/(MSUSY**2))*D)**2 for D in Diagr )
    
def Oh(h,f,g,Yd,Yu,Ye,x,y,Uup,Unu,Udown,MGUT,mwino,MSUSY):
    #mwino = 1000
    #MSUSY = 1000000
    Diagr = []
    for nu in range(3):
        udLsR, udR = CIIIh0(h,f,g,Yd,Yu,Ye,x,y,Uup,Unu,Udown,nu) 
        usRuL, udRR = CIIIh0(h,f,g,Yd,Yu,Ye,x,y,Uup,Unu,Udown,nu)
        Diagr.append(udR + CIIIhpm(h,f,g,Yd,Yu,Ye,x,y,Uup,Unu,Udown,nu)+CIVhpm(h, f, g, Yd, Yu, Ye, x, y, Uup, Unu, Udown,nu) + usRuL)
    #Op = (1j/16*np.pi)*(1/MGUT)*(mwino/(MSUSY**2))*Diagr
    return sum (np.abs((1j/16*np.pi)*(1/MGUT)*(mwino/(MSUSY**2))*D)**2 for D in Diagr)


    

if __name__ == "__main__":
    print("Debug! Using Benchmark point of old scan")
    
    h = np.array([[-4.50654e-06,-0.000522849,0.000186454],[-0.000522849,-0.00227386,-0.00559954],[0.000186454,-0.00559954,0.0754728]])
    f = np.array([[-1.15961e-08,-3.82153e-06,1.3628e-06],[-3.82153e-06,-0.00027017,-0.0000409273],[1.3628e-06,-0.0000409273,-0.0033514]])
    g = 1j*np.array([[0,-5.22136e-06,-0.0000208256],[5.22136e-06,0.,-0.000113134],[0.0000208256,0.000113134,0]])
    x = [1,1,1,1,1,1,1,1,1,1,1]########play with these values to see how the lifetime changes, very interesting is studying the parameter V16
    y = [1,1,1,1,1,1,1,1,1,1,1]
    Uel = [[0.824942, 0.544983, -0.144706 + 0.0390418*1j],[-0.546527 + 0.00116754*1j, 0.83669 + 0.000771313*1j, 0.0354365],[0.140417 + 0.0325542*1j, 0.0498073 + 0.0215064*1j, 0.988069]]
    Udown = [[0.974259, 0.225409, 0.00111616 - 0.00294036*1j],[-0.225304 - 0.000102674*1j, 0.973629 - 0.0000237552*1j, 0.0358412],[0.00699228 - 0.00286285*1j, -0.0351704 - 0.000662361*1j, 0.999353]]
    #Uel =  [[-0.93057364, 0.36574057,-0.016324542],[0.27487502+0.24134359j,0.69930416+0.6141376j,-0.0016874338+0.0016577225j],[0.013473443+0.0069363383j,-0.0041697496-0.0050052875j,-0.86146843-0.5075432j]]
    #Udown = [[9.734046459197998047e-01+0.000000000000000000e+00j,2.251969873905181885e-01+0.000000000000000000e+00j,1.492936164140701294e-02-3.932932391762733459e-02j],[-2.254479676485061646e-01-1.394745486322790384e-04j,9.742484092712402344e-01-3.226740955142304301e-05j,3.636769484728574753e-03],[-1.372464932501316071e-02-3.831701353192329407e-02j,-6.911328062415122986e-03-8.864633738994598389e-03j,9.991081357002258301e-01+0.000000000000000000e+00j]]
    #Yd = [[1.78e-06+6.9e-13j,5.82e-05+2.67e-05j,-0.00066-0.000195j],[5.82e-05+2.67e-05j,0.000296,-3.02e-05-5.38e-05j],[-0.00066-0.000195j,-3.02e-05-5.38e-05j,-0.016]]
    #Yu = [[-6.9e-06,0,0],[0,0.00359,0],[0,0,0.9861]]
    #Ye = [[-7.45e-07,-2.17e-05+1.99e-05j,0.000246-0.000145j],[-2.17e-05+1.99e-05j,-6.78e-05,1.123e-05-4.0e-05j],[0.000246+0.000145j,1.123e-05-4.0e-05j,0.0175]]
    #print(CIIIW(h, f, g, x, y, [[1,0,0],[0,1,0],[0,0,1]], Uel, Udown))
    print(CIW(h, f, g, x, y, [[1,0,0],[0,1,0],[0,0,1]], Uel, Udown)+CIVW(h, f, g, x, y, [[1,0,0],[0,1,0],[0,0,1]], Uel, Udown))
    













        
        






























        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        