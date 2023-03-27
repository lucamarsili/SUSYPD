
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
    #consider all the possible combinations for both x and y following 0606.08648, then different choices on x and y 
    #will allow us to distringuish between left and right
    part1R = x[0]*hU[a][b]*hV[c][d] + x[1]*fU[a][b]*fV[c][d] + x[2]*gU[a][b]*gV[c][d] + x[3]*gU[a][b]*gV[c][d] + x[5]*fU[a][b]*gV[c][d] 
    part2R = x[6]*gU[a][b]*fV[c][d] + x[7]*hU[a][b]*gV[c][d] + x[8]*gU[a][b]*hV[c][d] + x[9]*fU[a][d]*hV[b][c] + x[10]*gU[a][d]*gV[b][c] 
    part1L = x[0]*hU[a][b]*hV[c][d] + x[1]*fU[a][b]*fV[c][d] - x[3]*hU[a][b]*fV[c][d] - x[4]*fU[a][b]*hV[c][d]
    part2L = y[7]*hU[a][b]*gV[c][d] + y[9]*gU[a][c]*fV[b][d] + y[10]*gU[a][c]*gV[b][d]
    
    if L: 
        return part1L+ part2L 
    else:
        return part2R + part1R
    
    
#################bino, gluino and neutral wino dressing do not contribute to pion channel
##############################charged wino channel##############################################

def CIIW(h,f,g,x,y,Uup,Uel,Udown):
    #call mixing matrices and hfg
    SumDiagr = 0
    RotateUup = False
    RotateUel = True
    #j = 1 k = 2
    
    SumDiagr = SumDiagr +CL(x,y,h,f,g,np.conjugate(Uup),np.conjugate(np.transpose(Uel)),0,1,2,0,RotateUup,RotateUel)*Udown[2][0]*Uup[1][0] #do the permutation at this point
    SumDiagr = SumDiagr +CL(x,y,h,f,g,np.conjugate(Uup),np.conjugate(np.transpose(Uel)),2,0,1,0,RotateUup,RotateUel)*Udown[2][0]*Uup[1][0] #do the permutation at this point
    SumDiagr = SumDiagr +CL(x,y,h,f,g,np.conjugate(Uup),np.conjugate(np.transpose(Uel)),1,2,0,0,RotateUup,RotateUel)*Udown[2][0]*Uup[1][0] #do the permutation at this point
    
    SumDiagr = SumDiagr -CL(x,y,h,f,g,np.conjugate(Uup),np.conjugate(np.transpose(Uel)),1,0,2,0,RotateUup,RotateUel)*Udown[2][0]*Uup[1][0] #do the permutation at this point
    SumDiagr = SumDiagr -CL(x,y,h,f,g,np.conjugate(Uup),np.conjugate(np.transpose(Uel)),2,1,0,0,RotateUup,RotateUel)*Udown[2][0]*Uup[1][0] #do the permutation at this point
    SumDiagr = SumDiagr -CL(x,y,h,f,g,np.conjugate(Uup),np.conjugate(np.transpose(Uel)),0,2,1,0,RotateUup,RotateUel)*Udown[2][0]*Uup[1][0] #do the permutation at this point
    
    #j = 2 k = 1
    SumDiagr = SumDiagr +CL(x,y,h,f,g,np.conjugate(Uup),np.conjugate(np.transpose(Uel)),0,2,1,0,RotateUup,RotateUel)*Udown[1][0]*Uup[2][0] #do the permutation at this point
    SumDiagr = SumDiagr +CL(x,y,h,f,g,np.conjugate(Uup),np.conjugate(np.transpose(Uel)),1,0,2,0,RotateUup,RotateUel)*Udown[1][0]*Uup[2][0] #do the permutation at this point
    SumDiagr = SumDiagr +CL(x,y,h,f,g,np.conjugate(Uup),np.conjugate(np.transpose(Uel)),2,1,0,0,RotateUup,RotateUel)*Udown[1][0]*Uup[2][0] #do the permutation at this point
    
    SumDiagr = SumDiagr -CL(x,y,h,f,g,np.conjugate(Uup),np.conjugate(np.transpose(Uel)),0,1,2,0,RotateUup,RotateUel)*Udown[1][0]*Uup[2][0] #do the permutation at this point
    SumDiagr = SumDiagr -CL(x,y,h,f,g,np.conjugate(Uup),np.conjugate(np.transpose(Uel)),2,0,1,0,RotateUup,RotateUel)*Udown[1][0]*Uup[2][0] #do the permutation at this point
    SumDiagr = SumDiagr -CL(x,y,h,f,g,np.conjugate(Uup),np.conjugate(np.transpose(Uel)),1,2,0,0,RotateUup,RotateUel)*Udown[1][0]*Uup[2][0] #do the permutation at this point
    
    
    return (1/12)*SumDiagr

def CIIIW(h,f,g,x,y,Uup,Uel,Udown):
    SumDiagr = 0
    RotateUup = False
    RotateUdown = False
    for j in range(3):
        for k in range(3):
            if (j<k):
                for l in range(3):
                    
                    SumDiagr = SumDiagr +(1/4)*(CL(x,y,h,f,g,np.conjugate(Uup),np.conjugate(Udown),0,j,k,l,RotateUup,RotateUdown)- CL(x,y,h,f,g,np.conjugate(Uup),np.conjugate(Udown),0,k,j,l,RotateUup,RotateUdown))*Uup[j][0]*Uel[l][0] 
                                                                                                        
    return -SumDiagr


#Diagrams of Higgsino channel###################à

#############################charged higgsino

def CIhpm(h,f,g,Yd,Yu,Ye,x,y,Uup,Uel,Udown): ##change it!!
    SumDiagr = 0
    RotateUup = False
    RotateUel = True
    
    #j = 1 k = 2
    
    SumDiagr = SumDiagr + CL(x,y,h,f,g,np.conjugate(Uup),np.conjugate(np.transpose(Uel)),0,1,2,0,RotateUup,RotateUel)*(np.conjugate(np.transpose(Yd)))[2][0]*(np.conjugate(np.transpose(Yu)))[1][0]
    SumDiagr = SumDiagr + CL(x,y,h,f,g,np.conjugate(Uup),np.conjugate(np.transpose(Uel)),2,0,1,0,RotateUup,RotateUel)*(np.conjugate(np.transpose(Yd)))[2][0]*(np.conjugate(np.transpose(Yu)))[1][0]
    SumDiagr = SumDiagr + CL(x,y,h,f,g,np.conjugate(Uup),np.conjugate(np.transpose(Uel)),1,2,0,0,RotateUup,RotateUel)*(np.conjugate(np.transpose(Yd)))[2][0]*(np.conjugate(np.transpose(Yu)))[1][0]
    
    SumDiagr = SumDiagr - CL(x,y,h,f,g,np.conjugate(Uup),np.conjugate(np.transpose(Uel)),1,0,2,0,RotateUup,RotateUel)*(np.conjugate(np.transpose(Yd)))[2][0]*(np.conjugate(np.transpose(Yu)))[1][0]
    SumDiagr = SumDiagr - CL(x,y,h,f,g,np.conjugate(Uup),np.conjugate(np.transpose(Uel)),2,1,0,0,RotateUup,RotateUel)*(np.conjugate(np.transpose(Yd)))[2][0]*(np.conjugate(np.transpose(Yu)))[1][0]
    SumDiagr = SumDiagr - CL(x,y,h,f,g,np.conjugate(Uup),np.conjugate(np.transpose(Uel)),0,2,1,0,RotateUup,RotateUel)*(np.conjugate(np.transpose(Yd)))[2][0]*(np.conjugate(np.transpose(Yu)))[1][0]
    
    #j = 2 k = 1
    
    SumDiagr = SumDiagr + CL(x,y,h,f,g,np.conjugate(Uup),np.conjugate(np.transpose(Uel)),0,2,1,0,RotateUup,RotateUel)*(np.conjugate(np.transpose(Yd)))[1][0]*(np.conjugate(np.transpose(Yu)))[2][0]
    SumDiagr = SumDiagr + CL(x,y,h,f,g,np.conjugate(Uup),np.conjugate(np.transpose(Uel)),1,0,2,0,RotateUup,RotateUel)*(np.conjugate(np.transpose(Yd)))[1][0]*(np.conjugate(np.transpose(Yu)))[2][0]
    SumDiagr = SumDiagr + CL(x,y,h,f,g,np.conjugate(Uup),np.conjugate(np.transpose(Uel)),2,1,0,0,RotateUup,RotateUel)*(np.conjugate(np.transpose(Yd)))[1][0]*(np.conjugate(np.transpose(Yu)))[2][0]
    
    SumDiagr = SumDiagr - CL(x,y,h,f,g,np.conjugate(Uup),np.conjugate(np.transpose(Uel)),2,0,1,0,RotateUup,RotateUel)*(np.conjugate(np.transpose(Yd)))[1][0]*(np.conjugate(np.transpose(Yu)))[2][0]
    SumDiagr = SumDiagr - CL(x,y,h,f,g,np.conjugate(Uup),np.conjugate(np.transpose(Uel)),1,2,0,0,RotateUup,RotateUel)*(np.conjugate(np.transpose(Yd)))[1][0]*(np.conjugate(np.transpose(Yu)))[2][0]
    SumDiagr = SumDiagr - CL(x,y,h,f,g,np.conjugate(Uup),np.conjugate(np.transpose(Uel)),0,1,2,0,RotateUup,RotateUel)*(np.conjugate(np.transpose(Yd)))[1][0]*(np.conjugate(np.transpose(Yu)))[2][0]
    
    return SumDiagr

                       
def CIIhpm(h,f,g,Yd,Yu,Ye,x,y,Uup,Uel,Udown):
    SumDiagr = 0
    RotateUup = False
    RotateUdown = False
    for j in range(3):
        for k in range(3):
            if (j<k):
               for l in range(3):
                   SumDiagr = SumDiagr +(CL(x,y,h,f,g,np.conjugate(Uup),np.conjugate(Udown),0,j,k,l,RotateUup,RotateUdown)-CL(x,y,h,f,g,np.conjugate(Uup),np.conjugate(Udown),0,k,j,l,RotateUup,RotateUdown))*(np.conjugate(np.transpose(Yu)))[j][0]*(np.conjugate(np.transpose(Ye)))[l][0]
    return -SumDiagr

def CVhpm(h,f,g,Yd,Yu,Ye,x,y,Uup,Uel,Udown,L = False):
    SumDiagr = 0
    RotateUup = False
    RotateUel = True
    
    #j = 1 k = 2
    SumDiagr = SumDiagr +CL(x,y,h,f,g,np.conjugate(Uup),np.conjugate(np.transpose(Uel)),0,1,2,0,RotateUup,RotateUel,L)*Yu[2][0]*Yd[1][0] #do the permutation at this point
    SumDiagr = SumDiagr +CL(x,y,h,f,g,np.conjugate(Uup),np.conjugate(np.transpose(Uel)),1,2,0,0,RotateUup,RotateUel,L)*Yu[2][0]*Yd[1][0] #do the permutation at this point
    SumDiagr = SumDiagr +CL(x,y,h,f,g,np.conjugate(Uup),np.conjugate(np.transpose(Uel)),2,0,1,0,RotateUup,RotateUel,L)*Yu[2][0]*Yd[1][0] #do the permutation at this point
    
    SumDiagr = SumDiagr -CL(x,y,h,f,g,np.conjugate(Uup),np.conjugate(np.transpose(Uel)),0,2,1,0,RotateUup,RotateUel,L)*Yu[2][0]*Yd[1][0] #do the permutation at this point
    SumDiagr = SumDiagr -CL(x,y,h,f,g,np.conjugate(Uup),np.conjugate(np.transpose(Uel)),1,0,2,0,RotateUup,RotateUel,L)*Yu[2][0]*Yd[1][0] #do the permutation at this point
    SumDiagr = SumDiagr -CL(x,y,h,f,g,np.conjugate(Uup),np.conjugate(np.transpose(Uel)),2,1,0,0,RotateUup,RotateUel,L)*Yu[2][0]*Yd[1][0] #do the permutation at this point
    
    #j = 2 k = 1
    SumDiagr = SumDiagr +CL(x,y,h,f,g,np.conjugate(Uup),np.conjugate(np.transpose(Uel)),0,2,1,0,RotateUup,RotateUel,L)*Yu[1][0]*Yd[1][0] #do the permutation at this point
    SumDiagr = SumDiagr +CL(x,y,h,f,g,np.conjugate(Uup),np.conjugate(np.transpose(Uel)),1,0,2,0,RotateUup,RotateUel,L)*Yu[1][0]*Yd[1][0] #do the permutation at this point
    SumDiagr = SumDiagr +CL(x,y,h,f,g,np.conjugate(Uup),np.conjugate(np.transpose(Uel)),2,1,0,0,RotateUup,RotateUel,L)*Yu[1][0]*Yd[1][0] #do the permutation at this point
    
    SumDiagr = SumDiagr -CL(x,y,h,f,g,np.conjugate(Uup),np.conjugate(np.transpose(Uel)),0,1,2,0,RotateUup,RotateUel,L)*Yu[1][0]*Yd[1][0] #do the permutation at this point
    SumDiagr = SumDiagr -CL(x,y,h,f,g,np.conjugate(Uup),np.conjugate(np.transpose(Uel)),1,2,0,0,RotateUup,RotateUel,L)*Yu[1][0]*Yd[1][0] #do the permutation at this point
    SumDiagr = SumDiagr -CL(x,y,h,f,g,np.conjugate(Uup),np.conjugate(np.transpose(Uel)),2,0,1,0,RotateUup,RotateUel,L)*Yu[1][0]*Yd[1][0] #do the permutation at this point
    
    
    SumDiagr = (1/12)*SumDiagr #change here it imght be wrong!!! 
    
   
    ###write permutation here
    
    return SumDiagr

################neutral higgsino contribution

def CIh0(h,f,g,Yd,Yu,Ye,x,y,Uup,Uel,Udown):
    SumDiagr = 0
    RotateUup = False
    RotateUel = True
    for l in range(3):
        #i = 1 j = 2
        SumDiagr = SumDiagr +CL(x,y,h,f,g,np.conjugate(Uup),np.conjugate(np.transpose(Uel)),1,2,0,l,RotateUup,RotateUel)*np.conjugate(np.transpose(Yu))[1][0]*np.conjugate(np.transpose(Yd))[l][0]  #do the permutation at this point
        SumDiagr = SumDiagr +CL(x,y,h,f,g,np.conjugate(Uup),np.conjugate(np.transpose(Uel)),0,1,2,l,RotateUup,RotateUel)*np.conjugate(np.transpose(Yu))[1][0]*np.conjugate(np.transpose(Yd))[l][0]  #do the permutation at this point
        SumDiagr = SumDiagr +CL(x,y,h,f,g,np.conjugate(Uup),np.conjugate(np.transpose(Uel)),2,0,1,l,RotateUup,RotateUel)*np.conjugate(np.transpose(Yu))[1][0]*np.conjugate(np.transpose(Yd))[l][0]  #do the permutation at this point
        
        SumDiagr = SumDiagr -CL(x,y,h,f,g,np.conjugate(Uup),np.conjugate(np.transpose(Uel)),2,1,0,l,RotateUup,RotateUel)*np.conjugate(np.transpose(Yu))[1][0]*np.conjugate(np.transpose(Yd))[l][0]  #do the permutation at this point
        SumDiagr = SumDiagr -CL(x,y,h,f,g,np.conjugate(Uup),np.conjugate(np.transpose(Uel)),0,2,1,l,RotateUup,RotateUel)*np.conjugate(np.transpose(Yu))[1][0]*np.conjugate(np.transpose(Yd))[l][0]  #do the permutation at this point
        SumDiagr = SumDiagr -CL(x,y,h,f,g,np.conjugate(Uup),np.conjugate(np.transpose(Uel)),1,0,2,l,RotateUup,RotateUel)*np.conjugate(np.transpose(Yu))[1][0]*np.conjugate(np.transpose(Yd))[l][0]  #do the permutation at this point
    
        # i = 2 j = 1
        SumDiagr = SumDiagr +CL(x,y,h,f,g,np.conjugate(Uup),np.conjugate(np.transpose(Uel)),2,1,0,l,RotateUup,RotateUel)*np.conjugate(np.transpose(Yu))[2][0]*np.conjugate(np.transpose(Ye))[l][0]  #do the permutation at this point
        SumDiagr = SumDiagr +CL(x,y,h,f,g,np.conjugate(Uup),np.conjugate(np.transpose(Uel)),0,2,1,l,RotateUup,RotateUel)*np.conjugate(np.transpose(Yu))[2][0]*np.conjugate(np.transpose(Ye))[l][0]  #do the permutation at this point
        SumDiagr = SumDiagr +CL(x,y,h,f,g,np.conjugate(Uup),np.conjugate(np.transpose(Uel)),1,0,2,l,RotateUup,RotateUel)*np.conjugate(np.transpose(Yu))[2][0]*np.conjugate(np.transpose(Ye))[l][0]  #do the permutation at this point

        SumDiagr = SumDiagr -CL(x,y,h,f,g,np.conjugate(Uup),np.conjugate(np.transpose(Uel)),1,2,0,l,RotateUup,RotateUel)*np.conjugate(np.transpose(Yu))[2][0]*np.conjugate(np.transpose(Ye))[l][0]  #do the permutation at this point
        SumDiagr = SumDiagr -CL(x,y,h,f,g,np.conjugate(Uup),np.conjugate(np.transpose(Uel)),0,1,2,l,RotateUup,RotateUel)*np.conjugate(np.transpose(Yu))[2][0]*np.conjugate(np.transpose(Ye))[l][0]  #do the permutation at this point
        SumDiagr = SumDiagr -CL(x,y,h,f,g,np.conjugate(Uup),np.conjugate(np.transpose(Uel)),2,0,1,l,RotateUup,RotateUel)*np.conjugate(np.transpose(Yu))[2][0]*np.conjugate(np.transpose(Ye))[l][0]  #do the permutation at this point

    
    return -(1/12)*SumDiagr
############################

def CIIh0(h,f,g,Yd,Yu,Ye,x,y,Uup,Uel,Udown):
    SumDiagr = 0
    RotateUdown = False
    RotateUup = False
    
    #j = 1 k = 2
    SumDiagr = SumDiagr +CL(x,y,h,f,g,np.conjugate(Udown),np.conjugate(Uup),0,1,2,0,RotateUdown,RotateUup)*np.conjugate(np.transpose(Yd))[2][0]*np.conjugate(np.transpose(Yu))[1][0]  #do the permutation at this point
    SumDiagr = SumDiagr +CL(x,y,h,f,g,np.conjugate(Udown),np.conjugate(Uup),2,0,1,0,RotateUdown,RotateUup)*np.conjugate(np.transpose(Yd))[2][0]*np.conjugate(np.transpose(Yu))[1][0]  #do the permutation at this point
    SumDiagr = SumDiagr +CL(x,y,h,f,g,np.conjugate(Udown),np.conjugate(Uup),1,2,0,0,RotateUdown,RotateUup)*np.conjugate(np.transpose(Yd))[2][0]*np.conjugate(np.transpose(Yu))[1][0]  #do the permutation at this point
        
    SumDiagr = SumDiagr -CL(x,y,h,f,g,np.conjugate(Udown),np.conjugate(Uup),0,2,1,0,RotateUdown,RotateUup)*np.conjugate(np.transpose(Yd))[2][0]*np.conjugate(np.transpose(Yu))[1][0]  #do the permutation at this point
    SumDiagr = SumDiagr -CL(x,y,h,f,g,np.conjugate(Udown),np.conjugate(Uup),1,0,2,0,RotateUdown,RotateUup)*np.conjugate(np.transpose(Yd))[2][0]*np.conjugate(np.transpose(Yu))[1][0]  #do the permutation at this point
    SumDiagr = SumDiagr -CL(x,y,h,f,g,np.conjugate(Udown),np.conjugate(Uup),2,1,0,0,RotateUdown,RotateUup)*np.conjugate(np.transpose(Yd))[2][0]*np.conjugate(np.transpose(Yu))[1][0]  #do the permutation at this point
    
    
    #j = 2 k = 1
    SumDiagr = SumDiagr +CL(x,y,h,f,g,np.conjugate(Udown),np.conjugate(Uup),0,2,1,0,RotateUdown,RotateUup)*np.conjugate(np.transpose(Yd))[1][0]*np.conjugate(np.transpose(Yu))[2][0]  #do the permutation at this point
    SumDiagr = SumDiagr +CL(x,y,h,f,g,np.conjugate(Udown),np.conjugate(Uup),1,0,2,0,RotateUdown,RotateUup)*np.conjugate(np.transpose(Yd))[1][0]*np.conjugate(np.transpose(Yu))[2][0]  #do the permutation at this point
    SumDiagr = SumDiagr +CL(x,y,h,f,g,np.conjugate(Udown),np.conjugate(Uup),2,1,0,0,RotateUdown,RotateUup)*np.conjugate(np.transpose(Yd))[1][0]*np.conjugate(np.transpose(Yu))[2][0]  #do the permutation at this point
        
    SumDiagr = SumDiagr -CL(x,y,h,f,g,np.conjugate(Udown),np.conjugate(Uup),0,1,2,0,RotateUdown,RotateUup)*np.conjugate(np.transpose(Yd))[1][0]*np.conjugate(np.transpose(Yu))[2][0]  #do the permutation at this point
    SumDiagr = SumDiagr -CL(x,y,h,f,g,np.conjugate(Udown),np.conjugate(Uup),2,0,1,0,RotateUdown,RotateUup)*np.conjugate(np.transpose(Yd))[1][0]*np.conjugate(np.transpose(Yu))[2][0]  #do the permutation at this point
    SumDiagr = SumDiagr -CL(x,y,h,f,g,np.conjugate(Udown),np.conjugate(Uup),1,2,0,0,RotateUdown,RotateUup)*np.conjugate(np.transpose(Yd))[1][0]*np.conjugate(np.transpose(Yu))[2][0]  #do the permutation at this point
    
    return -(1/12)*SumDiagr


def CVh0(h,f,g,Yd,Yu,Ye,x,y,Uup,Uel,Udown, L = False):
    SumDiagr = 0
    RotateUdown = False
    RotateUup = False
    
    #j = 1 k = 2
    SumDiagr = SumDiagr +CL(x,y,h,f,g,np.conjugate(Udown),np.conjugate(Uup),0,1,2,0,RotateUdown,RotateUup,L)*Yu[2][0]*Yd[1][0]  #do the permutation at this point
    SumDiagr = SumDiagr +CL(x,y,h,f,g,np.conjugate(Udown),np.conjugate(Uup),2,0,1,0,RotateUdown,RotateUup,L)*Yu[2][0]*Yd[1][0]  #do the permutation at this point
    SumDiagr = SumDiagr +CL(x,y,h,f,g,np.conjugate(Udown),np.conjugate(Uup),1,2,0,0,RotateUdown,RotateUup,L)*Yu[2][0]*Yd[1][0]  #do the permutation at this point
    
    SumDiagr = SumDiagr -CL(x,y,h,f,g,np.conjugate(Udown),np.conjugate(Uup),0,2,1,0,RotateUdown,RotateUup,L)*Yu[2][0]*Yd[1][0]  #do the permutation at this point
    SumDiagr = SumDiagr -CL(x,y,h,f,g,np.conjugate(Udown),np.conjugate(Uup),1,0,2,0,RotateUdown,RotateUup,L)*Yu[2][0]*Yd[1][0]  #do the permutation at this point
    SumDiagr =-CL(x,y,h,f,g,np.conjugate(Udown),np.conjugate(Uup),2,1,0,0,RotateUdown,RotateUup,L)*Yu[2][0]*Yd[1][0]  #do the permutation at this point
   
    #j = 2 k = 1
    SumDiagr = SumDiagr +CL(x,y,h,f,g,np.conjugate(Udown),np.conjugate(Uup),0,2,1,0,RotateUdown,RotateUup,L)*Yu[1][0]*Yd[2][0]  #do the permutation at this point
    SumDiagr = SumDiagr +CL(x,y,h,f,g,np.conjugate(Udown),np.conjugate(Uup),1,0,2,0,RotateUdown,RotateUup,L)*Yu[1][0]*Yd[2][0]  #do the permutation at this point
    SumDiagr = SumDiagr +CL(x,y,h,f,g,np.conjugate(Udown),np.conjugate(Uup),1,2,0,0,RotateUdown,RotateUup,L)*Yu[1][0]*Yd[2][0]  #do the permutation at this point

    SumDiagr = SumDiagr -CL(x,y,h,f,g,np.conjugate(Udown),np.conjugate(Uup),0,1,2,0,RotateUdown,RotateUup,L)*Yu[1][0]*Yd[2][0]  #do the permutation at this point
    SumDiagr = SumDiagr -CL(x,y,h,f,g,np.conjugate(Udown),np.conjugate(Uup),1,0,2,0,RotateUdown,RotateUup,L)*Yu[1][0]*Yd[2][0]  #do the permutation at this point
    SumDiagr = SumDiagr -CL(x,y,h,f,g,np.conjugate(Udown),np.conjugate(Uup),2,1,0,0,RotateUdown,RotateUup,L)*Yu[1][0]*Yd[2][0]  #do the permutation at this point
  
    return -(1/12)*SumDiagr



####################defining the functions passed to the proton decay, we have Ow, OhudR and OhudL

def OW(h,f,g,Yd,Yu,Ye,x,y,Uup,Uel,Udown,alpha321SM_MZ,MGUT):
    mwino = 1000
    MSUSY = 100000
    Diagr = CIIW(h,f,g,x,y,Uup,Uel,Udown)+ CIIIW(h, f, g, x, y, Uup, Uel, Udown)
    Op = ((1j*alpha321SM_MZ[1])/4*np.pi)*(1/MGUT)*(mwino/(MSUSY**2))*Diagr
    return np.conjugate(Op)*Op
    
def OhudL(h,f,g,Yd,Yu,Ye,x,y,Uup,Uel,Udown,MGUT):
    mwino = 1000
    MSUSY = 100000
    Diagr = CIIhpm(h, f, g, Yd, Yu, Ye, x, y, Uup, Uel, Udown)+ CVhpm(h, f, g, Yd, Yu, Ye, x, y, Uup, Uel, Udown) + CIh0(h, f, g, Yd, Yu, Ye, x, y, Uup, Uel, Udown) + CVh0(h, f, g, Yd, Yu, Ye, x, y, Uup, Uel, Udown)
    Op = (1j/16*np.pi)*(1/MGUT)*(mwino/(MSUSY**2))*Diagr
    return np.conjugate(Op)*Op

def OhudR(h,f,g,Yd,Yu,Ye,x,y,Uup,Uel,Udown,MGUT):
    mwino = 1000
    MSUSY = 100000
    Diagr = CIhpm(h, f, g, Yd, Yu, Ye, x, y, Uup, Uel, Udown) + CIIh0(h, f, g, Yd, Yu, Ye, x, y, Uup, Uel, Udown)
    Op = (1j/16*np.pi)*(1/MGUT)*(mwino/(MSUSY**2))*Diagr
    return np.conjugate(Op)*Op





if __name__ == "__main__":
    print("Debug!")
    h = np.array([[0.001,0.04,-0.15],[0.04,0.3,0.1],[-0.15,0.2,0.5]])
    f = np.array([[0.002,0.04,-0.15],[0.6,0.3,0.3],[-0.15,-0.2,0.5]])
    g = np.array([[0.1,0.4,0.1],[0.4,0.03,0.1],[-0.15,0.2,-0.5]])
    x = [1,1,1,1,1,1,1,1,1,1,1]########play with these values to see how the lifetime changes, very interesting is studying the parameter V16
    y = [1,1,1,1,1,1,1,1,1,1,1]
    Uel =  [[-0.93057364, 0.36574057,-0.016324542],[0.27487502+0.24134359j,0.69930416+0.6141376j,-0.0016874338+0.0016577225j],[0.013473443+0.0069363383j,-0.0041697496-0.0050052875j,-0.86146843-0.5075432j]]
    Udown = [[9.734046459197998047e-01+0.000000000000000000e+00j,2.251969873905181885e-01+0.000000000000000000e+00j,1.492936164140701294e-02-3.932932391762733459e-02j],[-2.254479676485061646e-01-1.394745486322790384e-04j,9.742484092712402344e-01-3.226740955142304301e-05j,3.636769484728574753e-03],[-1.372464932501316071e-02-3.831701353192329407e-02j,-6.911328062415122986e-03-8.864633738994598389e-03j,9.991081357002258301e-01+0.000000000000000000e+00j]]
    Yd = [[1.78e-06+6.9e-13j,5.82e-05+2.67e-05j,-0.00066-0.000195j],[5.82e-05+2.67e-05j,0.000296,-3.02e-05-5.38e-05j],[-0.00066-0.000195j,-3.02e-05-5.38e-05j,-0.016]]
    Yu = [[-6.9e-06,0,0],[0,0.00359,0],[0,0,0.9861]]
    Ye = [[-7.45e-07,-2.17e-05+1.99e-05j,0.000246-0.000145j],[-2.17e-05+1.99e-05j,-6.78e-05,1.123e-05-4.0e-05j],[0.000246+0.000145j,1.123e-05-4.0e-05j,0.0175]]
    #print(CIIIW(h, f, g, x, y, [[1,0,0],[0,1,0],[0,0,1]], Uel, Udown))
    print(CIIhpm(h, f, g, Yd, Yu, Ye, x, y, [[1,0,0],[0,1,0],[0,0,1]], Uel, Udown))
    

























#return Oh
