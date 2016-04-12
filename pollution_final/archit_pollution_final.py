# -*- coding: utf-8 -*-
"""
Created on Wed Mar 09 16:20:40 2016

@author: Archit Datar
"""
'''
Brief description of code:
We first import all the necessary data from the interface. then we build an array of grids where we
visualose pollution.
An array of distances in x-direction is made and a correspoding array of kxx,kyy,kzz which are eddy diffusivities
are generated from the Sutton Paramters through pandas. This happens in the get_diff function.
Then we also write a function to find Q during the day. When factory is purging, the Q there is max, 
else, it is 0.
Then, accordingly we update the array for every time.
Also, we monitor pollution at the selected location throughout the day.

'''



#importing libraries
import numpy as np
import scipy 
import matplotlib.pyplot as plt
import pandas as pd
from math import ceil
import timeit
from visual import box, vector, color, display, frame, rate, arrow,cylinder, materials,label

start= timeit.default_timer() #to time the program


#importing from excel interface
df1= pd.read_excel('input_temp _for_3d.xlsx',0,header=1,parse_cols=2)

#importing values from interface
H= float(df1.Values[2]) #m # stack height
loc_point_source_dist= [df1.Values[3],df1.Values[4]]
time_factory_start= df1.Values[5]
time_factory_end= df1.Values[6]

#Sunlight_Cloud_Cover=df1.Values[7]
df2= pd.read_excel('input_temp _for_3d.xlsx',0,header=1,parse_cols=[5,6,7,8,9,10],index_col=[0])[0:-2]
df3= pd.read_excel('input_temp _for_3d.xlsx',0,header=2,parse_cols=[13,14,15,16,17,18,19,20],index_col=[0])[0:-2]

#defining lengths
l_x= float(df1.Values[16]);l_y=float(df1.Values[17]);l_z=float(df1.Values[18]) #kms

#defining resolutions in all directions
n_x= int(df1.Values[19]); n_y= int(df1.Values[20]);n_z= int(df1.Values[21])

#defining time and time steps
time_for_simulation= float(df1.Values[22]) #hrs
time_steps_for_simulation= float(df1.Values[23])

#time-step length
dt= time_for_simulation/time_steps_for_simulation

#defining size of cell
delta_x=l_x/n_x;delta_y=l_y/n_y;delta_z= l_z/n_z #kms

# converting location of point source to array
loc_point_source= (int(loc_point_source_dist[0]/delta_x),int(loc_point_source_dist[1]/delta_y),int(((float(H)/1000))/delta_z)-1)  #cell of location of point source

#defining air velocity
p= float(df1.Values[10])/1000 #km/s
q=float(df1.Values[11])/1000 #km/s
r= float(df1.Values[12])/1000 #km/s
u_stack_vec= (p,q,r) #km/s
u_stack_mag= np.linalg.norm(u_stack_vec) #km/s

#generating unit vectors
i_vec= (1,0,0); j_vec= (0,1,0); k_vec= (0,0,1) #unit vectors

#defining the cell in which location of plume exit lies
H_cell= int(H/(1000*delta_z))-1

#concentration of line source
Q_line = 400 #microgram/m**3-s #find a way of getting this from interface 

# location at which we want to monitor pollution throughout the day
location_for_plot= (int(df1.Values[26]/delta_x), int(df1.Values[27]/delta_y), int(df1.Values[28]/delta_z))

#getting an array of kxx, kyy, kzz for the x array
def get_diff_array(x, Sunlight_Cloud_Cover):
    #seasoninng the x array, to rectify the problem of first value False. No idea why this happens  
    x=x.ravel()
    x= np.append(1,x)
    
    # getting kxx, kyy,kzz single values for a single element x        
    def get_diff(x):
        # x to be imput should be in kms    
        if x>0:
            u_ground= u_stack_mag*(.010/(float(H)/1000))**0.5 *1000 #m/s
            if u_ground <=1 :
                u_ground= u_ground+1
                
            u_ground= ceil(u_ground)
            
                    
            #defining class from table B
            col= np.argwhere(np.array(df2)[0,:]==Sunlight_Cloud_Cover)[0,0]
            row= np.argwhere(np.array(df2.index)==int(u_ground))[0,0]        
            class_of_atm = df2.iloc[row,col]
            
            #getting A,B,p from last table    
            table_row= df3[df3.index==class_of_atm]
            A= table_row['A']
            if x<= table_row['x1(km)'].item():
                B= table_row.iloc[:,2]   
                p= table_row.iloc[:,3]
            
            if x> table_row['x1(km)'].item():
                B= table_row.iloc[:,4]   
                p= table_row.iloc[:,5]
            
            
            sigma_y= A*(x*1000)**0.93/1000 #km
            sigma_z= B*(x*1000)**p/1000#km
            kxx=kyy= sigma_y**2* u_stack_mag/(2*x) #km**2/s
            kzz= sigma_z**2*u_stack_mag/(2*x)    #km**2/s    
            
        else :
            sigma_y=1
            sigma_z=1
            kxx=kyy= 0
            kzz= 0
        # sigma_y, sigma_z are in kms
            # kxx, kyy,kzz are also in kms
        return kxx,kyy,kzz
        
    get_diff_vect = np.vectorize(get_diff)
    
   
    kxx= get_diff_vect(x)[0];kxx= kxx[1:];kxx= np.reshape(kxx,c.shape)
    kyy= get_diff_vect(x)[1]; kyy= kyy[1:];kyy= np.reshape(kyy,c.shape)
    kzz= get_diff_vect(x)[2]; kzz=kzz[1:]; kzz= np.reshape(kzz,c.shape)
    
    
    return kxx, kyy, kzz
        
#defining the day or night
def daynight(time_hours):
    if 6< time_hours < 18 :
        time = "Day"
        Sunlight_Cloud_Cover=df1.Values[7]
    else:
        time = "Night"
        Sunlight_Cloud_Cover = df1.Values[8]
    return Sunlight_Cloud_Cover

# we are converting the source strength to conentration at source strngth. 
#This also allows us to have different source strength at different times.
def point_source_strength(time_hours):
    if time_factory_start < time_hours < time_factory_end :
        Q= df1.Values[1]/(delta_x*delta_y*delta_z) *10**-3 #microgram/m**3-s
    else:
        Q=0
    return Q

#generating the array of boxes
scale_factor= 14 #defining a scale factor to enlarge the frame
scene2 = display(title='Plume Dispersion',
     x=0, y=0, width=2500, height=400,
     center=(1,1,1), background=(0,0,0), userzoom= True)
     

#creating the frame
f= frame()    
list_of_boxes=[]
for i in range(n_x+2):
    for j in range(n_y+2):
        for k in range(n_z+2):
            boxes= box(frame=f,pos= vector(i*delta_x*scale_factor,j*delta_y* scale_factor,k*delta_z*scale_factor),length= delta_x*scale_factor , width= delta_z *scale_factor  , height= delta_y *scale_factor, opacity=0)
            list_of_boxes+= [boxes]
            
array_of_boxes= np.reshape(list_of_boxes,(n_x+2,n_y+2,n_z+2))
cylinder(frame=f, pos= vector(loc_point_source[0]*delta_x*scale_factor,loc_point_source[1]*delta_y* scale_factor,0) , axis= (0,0,1), radius= .5, height= H/1000, color= color.white, material= materials.rough, opacity=1 )

l= label(frame=f, pos= (0,4,l_z+2), text= "Time:")
factory_label= label(frame=f, pos= vector(loc_point_source[0]*delta_x*scale_factor,loc_point_source[1]*delta_y* scale_factor,0),xoffset= 0, zoffset=-2, yoffset=-1, text="Factory"  )
highway_label= label(frame=f, pos =vector(13,-1,-1), text= "Highway",zoffset=-1)
location_label= label(frame=f,pos= array_of_boxes[location_for_plot].pos, xoffset=1, yoffset=-1,zoffset=-1,text="My Home")

f.visible= True
f.rotate(angle= -scipy.pi/2, axis= (1,0,0), origin= (3,3,3))
f.pos= (-10,-5,5)

#function to update the array 
#loc_source represents the locations of the sources
def update(c, kxx, kyy, kzz):
    dQdt= np.dot(u_stack_vec,i_vec)*(c[:-2,1:-1,1:-1]-c[1:-1,1:-1,1:-1])+kxx[1:-1,1:-1,1:-1]/delta_x*(c[:-2,1:-1,1:-1]-2*c[1:-1,1:-1,1:-1]+c[2:,1:-1,1:-1])+np.dot(u_stack_vec,j_vec)*(c[1:-1,:-2,1:-1]-c[1:-1,1:-1,1:-1])+kyy[1:-1,1:-1,1:-1]/delta_y*(c[1:-1,:-2,1:-1]-2*c[1:-1,1:-1,1:-1]+c[1:-1,2:,1:-1])+ np.dot(u_stack_vec,k_vec)*(c[1:-1,1:-1,:-2]-c[1:-1,1:-1,1:-1])+kzz[1:-1,1:-1,1:-1]/delta_z*(c[1:-1,1:-1,:-2]-2*c[1:-1,1:-1,1:-1]+c[1:-1,1:-1,2:])
    c[1:-1,1:-1,1:-1]=c[1:-1,1:-1,1:-1]+dQdt * 2*(delta_x*delta_y+delta_y*delta_z+delta_z*delta_x)/(delta_x*delta_y*delta_z)* dt  # to convert hours to seconds #perfect units!
    return c
    
#defining the concentration array
c= np.zeros((n_x+2,n_y+2,n_z+2))

#making an array of x- distance points 
#making an array of x
x_array=np.reshape(np.argwhere(c>-1)[:,0],c.shape)
x=(x_array-loc_point_source[0])*delta_x #kms

#making a figure

plt.title("Pollution at my place")
plt.xlabel("Time(hrs)")
plt.ylabel("Concentration of pollutant (micrograms/ m^3)")
plt.xlim(0,24)
plt.xticks(np.linspace(0,24,13))
plt.grid(True)


for t_loop in range(int(time_steps_for_simulation+1)):
    t= t_loop*dt #updating time
    l.text= ("Time: %1.4f hours" %t  )   #displaying time
    Sunlight_Cloud_Cover = daynight (t) #obtaining conditions from interface
    Q= point_source_strength (t)    #obtaining Q depending on factories purge timings
    
    c[loc_point_source]= Q/(delta_x*delta_y*delta_z)*10**-3 # microgram/m**3-s
    c[4,2:5,0]= Q_line/(delta_x*delta_y*delta_z)*10**-3 #define Q_line from interface and the road loations from somewhere    
    
        
    #defining the kxx, kyy, kzz for the x values    
    kxx,kyy,kzz= get_diff_array(x, Sunlight_Cloud_Cover) #km**2/s
    
    # updating the array
    c= update(c, kxx,kyy,kzz)
    
    
    #resetting the Q-value accordingly
    c[loc_point_source]= Q/(delta_x*delta_y*delta_z)*10**-3 # microgram/m**3-s
    c[4,2:5,0]= Q_line/(delta_x*delta_y*delta_z)*10**-3 #define Q_line from interface and the road loations from somewhere
    
    #plt.contourf(c[:,loc_point_source[1],:].T,cmap='gray',origin='lower')
    plt.plot(t,c[location_for_plot],'.r')    
    plt.pause(0.0000000000000001)
    
       
    for i in np.arange(1,n_x+1):
        for j in np.arange(1,n_y+1):
            for k in np.arange(0,n_z+1):     
                array_of_boxes[i,j,k].color= (1,1-c[i,j,k]/600,0) # 600 just a random figure considering conentrations values. change if more convenient
                array_of_boxes[i,j,k].opacity= c[i,j,k]/600
    f.visible= True
    rate(100000000000000000)

#stopping the time and recorsing the elapsed time
stop= timeit.default_timer()
print stop-start,"seconds"
          
# better way to input road direction, calculate and input Q highway
#search for weave inline python for arrays
     
  # make the interface and upload the web app
     
#max conc, road location and Q_line are adhoc. everyhtihn else is iser defined      

         