from Yukawa import yukawamatrices
import numpy as np

#define here the yukawa matrices 



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
    #consider all the possible combinations for both x and y following 1606.08648, then different choices on x and y 
    #will allow us to distringuish between left and right
    part1R = x[0]*hU[a][b]*hV[c][d] + x[1]*fU[a][b]*fV[c][d] + x[2]*gU[a][b]*gV[c][d] + x[3]*gU[a][b]*gV[c][d] + x[5]*fU[a][b]*gV[c][d] 
    part2R = x[6]*gU[a][b]*fV[c][d] + x[7]*hU[a][b]*gV[c][d] + x[8]*gU[a][b]*hV[c][d] + x[9]*fU[a][d]*hV[b][c] + x[10]*gU[a][d]*gV[b][c] 
    part1L = x[0]*hU[a][b]*hV[c][d] + x[1]*fU[a][b]*fV[c][d] - x[3]*hU[a][b]*fV[c][d] - x[4]*fU[a][b]*hV[c][d]
    part2L = y[7]*hU[a][b]*gV[c][d] + y[9]*gU[a][c]*fV[b][d] + y[10]*gU[a][c]*gV[b][d]
    
    if L: 
        return part1L+ part2L 
    else:
        return part1R+part2R
    
    
#################bino, gluino and neutral wino dressing do not contribute to pion channel
##############################charged wino channel##############################################

def CIIW(h,f,g,x,y,Uup,Uel,Udown):
    #call mixing matrices and hfg
    SumDiagr = 0
    RotateUup = False
    RotateUel = True
    SumDiagr =+CL(x,y,h,f,g,np.conjugate(Uup),Uel,1,2,3,1,RotateUup,RotateUel) #do the permutation at this point
    SumDiagr =+CL(x,y,h,f,g,np.conjugate(Uup),Uel,2,3,1,1,RotateUup,RotateUel) #do the permutation at this point
    SumDiagr =+CL(x,y,h,f,g,np.conjugate(Uup),Uel,3,1,2,1,RotateUup,RotateUel) #do the permutation at this point
    
    SumDiagr =-CL(x,y,h,f,g,np.conjugate(Uup),Uel,2,1,3,1,RotateUup,RotateUel) #do the permutation at this point
    SumDiagr =-CL(x,y,h,f,g,np.conjugate(Uup),Uel,3,2,1,1,RotateUup,RotateUel) #do the permutation at this point
    SumDiagr =-CL(x,y,h,f,g,np.conjugate(Uup),Uel,1,3,2,1,RotateUup,RotateUel) #do the permutation at this point
    
    SumDiagr = (1/12)*SumDiagr*Udown[3,1]*Uup[2][1] #change here it imght be wrong!!! #simply you need to consider also j = 3 and k = 2 with an opposite sign!!!!!
    
    return SumDiagr

def CIIIW(h,f,g,x,y,Uup,Uel,Udown):
    SumDiagr = 0
    RotateUup = False
    RotateUdown = False
    for j in range(2,3):
        for l in range(3):
            SumDiagr =+(1/4)*(CL(x,y,h,f,g,np.conjugate(Uup),np.conjugate(Udown),1,j,1,l,RotateUup,RotateUdown)- CL(x,y,h,f,g,np.conjugate(Uup),np.conjugate(Udown),1,1,j,l,RotateUup,RotateUdown))*Uup[j,1]*Uel[l][1] 

                                                                                                        
    return SumDiagr


#Diagrams of Higgsino channel###################à

#############################charged higgsino

def CIhpm(h,f,g,Yd,Yu,Ye,x,y,Uup,Uel,Udown):
    SumDiagr = 0
    RotateUup = False
    RotateUel = True
    for j in range(3):
        for k in range(3):
            if (j<k):
               SumDiagr =+ (CL(x,y,h,f,g,np.conjugate(Uup),Uel,1,j,k,1,RotateUup,RotateUel)-CL(h,f,g,np.conjugate(Uup),Uel,1,k,j,1,RotateUup,RotateUel))*(np.conjugate(np.transpose(Yd)))[k][1]*(np.conjugate(np.transpose(Yu)))[j][1]
    return SumDiagr

                       
def CIIhpm(h,f,g,Yd,Yu,Ye,x,y,Uup,Uel,Udown):
    SumDiagr = 0
    RotateUup = False
    RotateUdown = False
    for j in range(3):
        for k in range(3):
            if (j<k):
               for l in range(3): 
                   SumDiagr =+ (CL(x,y,h,f,g,np.conjugate(Uup),np.conjugate(Udown),1,j,k,l,RotateUup,RotateUdown)-CL(h,f,g,np.conjugate(Uup),np.conjugate(Udown),1,k,j,l,RotateUup,RotateUdown))*(np.conjugate(np.transpose(Yu)))[j][1]*(np.conjugate(np.transpose(Ye)))[l][1]
    return -SumDiagr

def CVhpm(h,f,g,Yd,Yu,Ye,x,y,Uup,Uel,Udown,L = False):
    SumDiagr = 0
    RotateUup = False
    RotateUel = True
    
    #j = 2 k = 3
    SumDiagr =+CL(x,y,h,f,g,np.conjugate(Uup),Uel,1,2,3,1,RotateUup,RotateUel,L)*Yu[3][1]*Yd[2][1] #do the permutation at this point
    SumDiagr =+CL(x,y,h,f,g,np.conjugate(Uup),Uel,2,3,1,1,RotateUup,RotateUel,L)*Yu[3][1]*Yd[2][1] #do the permutation at this point
    SumDiagr =+CL(x,y,h,f,g,np.conjugate(Uup),Uel,3,1,2,1,RotateUup,RotateUel,L)*Yu[3][1]*Yd[2][1] #do the permutation at this point
    
    SumDiagr =-CL(x,y,h,f,g,np.conjugate(Uup),Uel,2,1,3,1,RotateUup,RotateUel,L)*Yu[3][1]*Yd[2][1] #do the permutation at this point
    SumDiagr =-CL(x,y,h,f,g,np.conjugate(Uup),Uel,3,2,1,1,RotateUup,RotateUel,L)*Yu[3][1]*Yd[2][1] #do the permutation at this point
    SumDiagr =-CL(x,y,h,f,g,np.conjugate(Uup),Uel,1,3,2,1,RotateUup,RotateUel,L)*Yu[3][1]*Yd[2][1] #do the permutation at this point
    
    #j = 3 k = 2
    SumDiagr =+CL(x,y,h,f,g,np.conjugate(Uup),Uel,1,3,2,1,RotateUup,RotateUel,L)*Yu[2][1]*Yd[3][1] #do the permutation at this point
    SumDiagr =+CL(x,y,h,f,g,np.conjugate(Uup),Uel,3,2,1,1,RotateUup,RotateUel,L)*Yu[2][1]*Yd[3][1] #do the permutation at this point
    SumDiagr =+CL(x,y,h,f,g,np.conjugate(Uup),Uel,2,1,3,1,RotateUup,RotateUel,L)*Yu[2][1]*Yd[3][1] #do the permutation at this point
    
    SumDiagr =-CL(x,y,h,f,g,np.conjugate(Uup),Uel,3,1,2,1,RotateUup,RotateUel,L)*Yu[2][1]*Yd[3][1] #do the permutation at this point
    SumDiagr =-CL(x,y,h,f,g,np.conjugate(Uup),Uel,2,3,1,1,RotateUup,RotateUel,L)*Yu[2][1]*Yd[3][1] #do the permutation at this point
    SumDiagr =-CL(x,y,h,f,g,np.conjugate(Uup),Uel,1,2,3,1,RotateUup,RotateUel,L)*Yu[2][1]*Yd[3][1] #do the permutation at this point
    
    
    SumDiagr = (1/12)*SumDiagr #change here it imght be wrong!!! 
    
   
    ###write permutation here
    
    return SumDiagr

################neutral higgsino contribution

def CIh0(h,f,g,Yd,Yu,Ye,x,y,Uup,Uel,Udown):
    SumDiagr = 0
    RotateUup = False
    RotateUel = True
    for l in range(3):
        #i = 2 j = 3
        SumDiagr =+CL(x,y,h,f,g,np.conjugate(Uup),Uel,2,3,1,l,RotateUup,RotateUel)*np.conjugate(np.transpose(Yu))[2][1]*np.conjugate(np.transpose(Yd))[l][1]  #do the permutation at this point
        SumDiagr =+CL(x,y,h,f,g,np.conjugate(Uup),Uel,1,2,3,l,RotateUup,RotateUel)*np.conjugate(np.transpose(Yu))[2][1]*np.conjugate(np.transpose(Yd))[l][1]  #do the permutation at this point
        SumDiagr =+CL(x,y,h,f,g,np.conjugate(Uup),Uel,3,1,2,l,RotateUup,RotateUel)*np.conjugate(np.transpose(Yu))[2][1]*np.conjugate(np.transpose(Yd))[l][1]  #do the permutation at this point
        
        SumDiagr =-CL(x,y,h,f,g,np.conjugate(Uup),Uel,2,1,3,l,RotateUup,RotateUel)*np.conjugate(np.transpose(Yu))[2][1]*np.conjugate(np.transpose(Yd))[l][1]  #do the permutation at this point
        SumDiagr =-CL(x,y,h,f,g,np.conjugate(Uup),Uel,3,2,1,l,RotateUup,RotateUel)*np.conjugate(np.transpose(Yu))[2][1]*np.conjugate(np.transpose(Yd))[l][1]  #do the permutation at this point
        SumDiagr =-CL(x,y,h,f,g,np.conjugate(Uup),Uel,1,3,2,l,RotateUup,RotateUel)*np.conjugate(np.transpose(Yu))[2][1]*np.conjugate(np.transpose(Yd))[l][1]  #do the permutation at this point
    
        # i = 3 j = 2
        SumDiagr =+CL(x,y,h,f,g,np.conjugate(Uup),Uel,3,2,1,l,RotateUup,RotateUel)*np.conjugate(np.transpose(Yu))[3][1]*np.conjugate(np.transpose(Ye))[l][1]  #do the permutation at this point
        SumDiagr =+CL(x,y,h,f,g,np.conjugate(Uup),Uel,1,3,2,l,RotateUup,RotateUel)*np.conjugate(np.transpose(Yu))[3][1]*np.conjugate(np.transpose(Ye))[l][1]  #do the permutation at this point
        SumDiagr =+CL(x,y,h,f,g,np.conjugate(Uup),Uel,2,1,3,l,RotateUup,RotateUel)*np.conjugate(np.transpose(Yu))[3][1]*np.conjugate(np.transpose(Ye))[l][1]  #do the permutation at this point

        SumDiagr =-CL(x,y,h,f,g,np.conjugate(Uup),Uel,3,1,2,l,RotateUup,RotateUel)*np.conjugate(np.transpose(Yu))[3][1]*np.conjugate(np.transpose(Ye))[l][1]  #do the permutation at this point
        SumDiagr =-CL(x,y,h,f,g,np.conjugate(Uup),Uel,2,3,1,l,RotateUup,RotateUel)*np.conjugate(np.transpose(Yu))[3][1]*np.conjugate(np.transpose(Ye))[l][1]  #do the permutation at this point
        SumDiagr =-CL(x,y,h,f,g,np.conjugate(Uup),Uel,1,2,3,l,RotateUup,RotateUel)*np.conjugate(np.transpose(Yu))[3][1]*np.conjugate(np.transpose(Ye))[l][1]  #do the permutation at this point

    
    return -(1/12)*SumDiagr
############################

def CIIh0(h,f,g,Yd,Yu,Ye,x,y,Uup,Uel,Udown):
    SumDiagr = 0
    RotateUdown = False
    RotateUup = False
    
    #j = 2 k = 3
    SumDiagr =+CL(x,y,h,f,g,np.conjugate(Udown),np.conjugate(Uup),1,2,3,1,RotateUdown,RotateUup)*np.conjugate(np.transpose(Yd))[3][1]*np.conjugate(np.transpose(Yu))[2][1]  #do the permutation at this point
    SumDiagr =+CL(x,y,h,f,g,np.conjugate(Udown),np.conjugate(Uup),3,1,2,1,RotateUdown,RotateUup)*np.conjugate(np.transpose(Yd))[3][1]*np.conjugate(np.transpose(Yu))[2][1]  #do the permutation at this point
    SumDiagr =+CL(x,y,h,f,g,np.conjugate(Udown),np.conjugate(Uup),2,3,1,1,RotateUdown,RotateUup)*np.conjugate(np.transpose(Yd))[3][1]*np.conjugate(np.transpose(Yu))[2][1]  #do the permutation at this point
        
    SumDiagr =-CL(x,y,h,f,g,np.conjugate(Udown),np.conjugate(Uup),2,1,3,1,RotateUdown,RotateUup)*np.conjugate(np.transpose(Yd))[3][1]*np.conjugate(np.transpose(Yu))[2][1]  #do the permutation at this point
    SumDiagr =-CL(x,y,h,f,g,np.conjugate(Udown),np.conjugate(Uup),1,3,2,1,RotateUdown,RotateUup)*np.conjugate(np.transpose(Yd))[3][1]*np.conjugate(np.transpose(Yu))[2][1]  #do the permutation at this point
    SumDiagr =-CL(x,y,h,f,g,np.conjugate(Udown),np.conjugate(Uup),3,2,1,1,RotateUdown,RotateUup)*np.conjugate(np.transpose(Yd))[3][1]*np.conjugate(np.transpose(Yu))[2][1]  #do the permutation at this point
    
    
    #j = 3 k = 2
    SumDiagr =+CL(x,y,h,f,g,np.conjugate(Udown),np.conjugate(Uup),1,3,2,1,RotateUdown,RotateUup)*np.conjugate(np.transpose(Yd))[2][1]*np.conjugate(np.transpose(Yu))[3][1]  #do the permutation at this point
    SumDiagr =+CL(x,y,h,f,g,np.conjugate(Udown),np.conjugate(Uup),2,1,3,1,RotateUdown,RotateUup)*np.conjugate(np.transpose(Yd))[2][1]*np.conjugate(np.transpose(Yu))[3][1]  #do the permutation at this point
    SumDiagr =+CL(x,y,h,f,g,np.conjugate(Udown),np.conjugate(Uup),3,2,1,1,RotateUdown,RotateUup)*np.conjugate(np.transpose(Yd))[2][1]*np.conjugate(np.transpose(Yu))[3][1]  #do the permutation at this point
        
    SumDiagr =-CL(x,y,h,f,g,np.conjugate(Udown),np.conjugate(Uup),3,1,2,1,RotateUdown,RotateUup)*np.conjugate(np.transpose(Yd))[2][1]*np.conjugate(np.transpose(Yu))[3][1]  #do the permutation at this point
    SumDiagr =-CL(x,y,h,f,g,np.conjugate(Udown),np.conjugate(Uup),1,2,3,1,RotateUdown,RotateUup)*np.conjugate(np.transpose(Yd))[2][1]*np.conjugate(np.transpose(Yu))[3][1]  #do the permutation at this point
    SumDiagr =-CL(x,y,h,f,g,np.conjugate(Udown),np.conjugate(Uup),2,3,1,1,RotateUdown,RotateUup)*np.conjugate(np.transpose(Yd))[2][1]*np.conjugate(np.transpose(Yu))[3][1]  #do the permutation at this point
    
    return -(1/12)*SumDiagr


def CVh0(h,f,g,Yd,Yu,Ye,x,y,Uup,Uel,Udown, L = False):
    SumDiagr = 0
    RotateUdown = False
    RotateUup = False
    
    #j = 2 k = 3
    SumDiagr =+CL(x,y,h,f,g,np.conjugate(Udown),np.conjugate(Uup),1,2,3,1,RotateUdown,RotateUup,L)*Yu[3][1]*Yd[2][1]  #do the permutation at this point
    SumDiagr =+CL(x,y,h,f,g,np.conjugate(Udown),np.conjugate(Uup),3,1,2,1,RotateUdown,RotateUup,L)*Yu[3][1]*Yd[2][1]  #do the permutation at this point
    SumDiagr =+CL(x,y,h,f,g,np.conjugate(Udown),np.conjugate(Uup),2,3,1,1,RotateUdown,RotateUup,L)*Yu[3][1]*Yd[2][1]  #do the permutation at this point
    
    SumDiagr =-CL(x,y,h,f,g,np.conjugate(Udown),np.conjugate(Uup),1,3,2,1,RotateUdown,RotateUup,L)*Yu[3][1]*Yd[2][1]  #do the permutation at this point
    SumDiagr =-CL(x,y,h,f,g,np.conjugate(Udown),np.conjugate(Uup),2,1,3,1,RotateUdown,RotateUup,L)*Yu[3][1]*Yd[2][1]  #do the permutation at this point
    SumDiagr =-CL(x,y,h,f,g,np.conjugate(Udown),np.conjugate(Uup),3,2,1,1,RotateUdown,RotateUup,L)*Yu[3][1]*Yd[2][1]  #do the permutation at this point
   
    #j = 3 k = 2
    SumDiagr =+CL(x,y,h,f,g,np.conjugate(Udown),np.conjugate(Uup),1,3,2,1,RotateUdown,RotateUup,L)*Yu[2][1]*Yd[3][1]  #do the permutation at this point
    SumDiagr =+CL(x,y,h,f,g,np.conjugate(Udown),np.conjugate(Uup),2,1,3,1,RotateUdown,RotateUup,L)*Yu[2][1]*Yd[3][1]  #do the permutation at this point
    SumDiagr =+CL(x,y,h,f,g,np.conjugate(Udown),np.conjugate(Uup),3,2,1,1,RotateUdown,RotateUup,L)*Yu[2][1]*Yd[3][1]  #do the permutation at this point

    SumDiagr =-CL(x,y,h,f,g,np.conjugate(Udown),np.conjugate(Uup),1,2,3,1,RotateUdown,RotateUup,L)*Yu[2][1]*Yd[3][1]  #do the permutation at this point
    SumDiagr =-CL(x,y,h,f,g,np.conjugate(Udown),np.conjugate(Uup),3,1,2,1,RotateUdown,RotateUup,L)*Yu[2][1]*Yd[3][1]  #do the permutation at this point
    SumDiagr =-CL(x,y,h,f,g,np.conjugate(Udown),np.conjugate(Uup),2,3,1,1,RotateUdown,RotateUup,L)*Yu[2][1]*Yd[3][1]  #do the permutation at this point
  
    return -(1/12)*SumDiagr


































#return Oh