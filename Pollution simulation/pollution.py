# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 16:23:19 2016

@author: Archit Datar
"""


import numpy as np
import scipy 
import matplotlib.pyplot as plt
import pandas as pd
from math import ceil

df1= pd.read_excel('input_temp.xlsx',0,header=1,parse_cols=2)

n=df1.Values[0] #kms
Q= df1.Values[1] #gm/s
H= df1.Values[2] #km
loc_source_1= (df1.Values[3],df1.Values[4])  #location of source in km from origin
u_stack= df1.Values[5] #km/s
time= df1.Values[6]
Sunlight_Cloud_Cover=df1.Values[7]

line_x= df1.Values[10]
Q_line = 0.6 #gm/m-s

df2= pd.read_excel('input_temp.xlsx',0,header=1,parse_cols=[5,6,7,8,9,10],index_col=[0])[0:-2]
df3= pd.read_excel('input_temp.xlsx',0,header=2,parse_cols=[13,14,15,16,17,18,19,20],index_col=[0])[0:-2]

        
# using if with pandas funcitons
#finding out how to call one function inside another... see Dalvis tut)        

#units of conc are then, mg/km**3

z= np.zeros((n,n))

XZ= np.zeros((n,n))

def point_profile(x,y,Q,sigma_y,sigma_z,u_stack,H):
    conc_profile=0    
    if x>0:
        if y>0:
            '''
            sigma_y= A*(x*1000)**.903/1000
            sigma_z= B*(x*1000)**p/1000   ''' 
            conc_profile = Q*1000/(scipy.pi*sigma_y*sigma_z*u_stack)*scipy.exp(-1/2*(y/sigma_y)**2)*scipy.exp(-1/2*(H/sigma_z)**2)        
        
    return conc_profile

def conc_line(x,Q,sigma_z):
    if x>0:
        conc= 2*Q_line*10**6/(scipy.sqrt(2*scipy.pi)*sigma_z*u_stack)
    else:
        conc=0
    return conc

def conc_point_z(x,z,Q,sigma_y,sigma_z,u_stack,H):
    conc=0
    if x>0:
        if z>0:
            
            conc = Q*1000/(2*scipy.pi*sigma_y*sigma_z*u_stack)*(scipy.exp(-.5*((z-H)/sigma_z)**2)+scipy.exp(-.5*((z+H)/sigma_z)**2))
                
    return conc
    
    
def get_sutton(x):
    if x>0:
            
        u_ground= u_stack*(.010/H)**0.5 *1000 #m/s
        u_ground= ceil(u_ground)
        
        vel_frame= df2[df2.index==u_ground]
        
        #defining class from table B
        col= np.argwhere(np.array(df2)[0,:]==Sunlight_Cloud_Cover)[0,0]
        row= np.argwhere(np.array(df2.index)==u_ground)[0,0]        
        class_= df2.iloc[row,col]
        
        #getting A,B,p from last table    
        table_row= df3[df3.index==class_]
        A= table_row['A']
        if x<= table_row['x1(km)'].item():
            B= table_row.iloc[:,2]   
            p= table_row.iloc[:,3]
        
        if x> table_row['x1(km)'].item():
            B= table_row.iloc[:,4]   
            p= table_row.iloc[:,5]
        
        
        sigma_y= A*(x*1000)**0.93/1000 #km
        sigma_z= B*(x*1000)**p/1000#km
    else:
        sigma_y=1
        sigma_z=1
    return sigma_y,sigma_z


def conc_point(z):
    conc_profile_list=[]
    for index_element in np.ndenumerate(z):
        
        index= index_element[0]
        x=index[0]+.0000001; y= index[1]+0.0000001
        
        sigma_y,sigma_z=get_sutton(x-2*loc_source_1[0])              
        conc_profile= point_profile(x-loc_source_1[0],y-loc_source_1[1],Q,sigma_y,sigma_z,u_stack,H)+conc_line(x-line_x,Q_line,sigma_z)         
               
        conc_profile_list +=[conc_profile]
        
    conc_profile = np.reshape(conc_profile_list,z.shape)
    conc_profile=conc_profile+z
    return conc_profile

def conc_point_in_z():
    conc_profile_list=[]
    for index_element in np.ndenumerate(XZ):
        
        index= index_element[0]
        x=index[0]+.0000001; z= index[1]+0.0000001
        
        sigma_y,sigma_z=get_sutton(x-loc_source_1[0])              
        conc_profile= conc_point_z(x-loc_source_1[0],z-3,Q,sigma_y,sigma_z,u_stack,H)         
        conc_profile_list +=[conc_profile]
        
    conc_profile = np.reshape(conc_profile_list,XZ.shape)
    conc_profile=conc_profile+z
    return conc_profile

fig= plt.figure()

ax1= fig.add_subplot(111)
plt.contourf(conc_point(z).T,cmap='summer',origin='lower')
plt.colorbar()
plt.set_cmap('hot')
plt.title('On XY plane')
plt.grid()
#ax2= fig.add_subplot(212)
#plt.imshow(conc_point_in_z().T,cmap= 'summer',origin='lower')
#plt.contour(conc_point(z).T,)
#plt.contour(conc_point(z).T,linewidths=2,extent=(-3, 3, -2, 2))    

'''        
dangerous=  np.argwhere(conc_point(z)>= 1e8)
x_start= dangerous[:,0].min;x_end= dangerous[:,0].max
y_start=dangerous[0,:].min; y_end= dangerous[0,:].max

print 'The affected area is : %f-%s' %x_start,%x_end
'''  
# to use the mouse
# bitmapped_graphics.py
#move_mouse.py            
    