##########################################TEST_CORDINATE################################################

# import numpy as np
# import math as m
# import pandas as pd
# # def Rx(theta):
# #   return np.matrix([[ 1, 0           , 0           ],
# #                    [ 0, m.cos(theta),-m.sin(theta)],
# #                    [ 0, m.sin(theta), m.cos(theta)]])
  
# # def Ry(theta):
# #   return np.matrix([[ m.cos(theta), 0, m.sin(theta)],
# #                    [ 0           , 1, 0           ],
# #                    [-m.sin(theta), 0, m.cos(theta)]])
  
# # def Rz(theta):
# #   return np.matrix([[ m.cos(theta), -m.sin(theta), 0 ],
# #                    [ m.sin(theta), m.cos(theta) , 0 ],
# #                    [ 0           , 0            , 1 ]])
# # phi = m.pi/2
# # theta = m.pi/4
# # psi = m.pi/2
# # print("phi =", phi)
# # print("theta  =", theta)
# # print("psi =", psi)
  
  
# # R = Rz(psi) * Ry(theta) * Rx(phi)
# from scipy.spatial.transform import Rotation as R
# A=[]
# df=pd.read_csv(r"C:\Users\ivsrs\Documents\AirSim\2021-05-07-22-37-05\2021-05-07-22-37-05_sub.csv");
# row,colum=df.shape
# #for i in range(row):
# i=0
# P=np.array([df.x[i],df.y[i],df.z[i]])
# qr= df.qw[i]
# qx=df.qx[i]
# qy=df.qy[i]
# qz=df.qz[i]
# img=df.img
# print(qx,qy,qz,qr)
# r = R.from_quat([qx,qy, qz, qr ])

# # print("P= {}".format(P))
# # print("qx=%s, qy=%s, qz=%s, qw=%s"%(qx,qy,qz,qr))

# r=r.as_matrix()
# print(r)

# # print("Toa do =%s"%(np.linalg.inv(r)@P.T))
# B=np.linalg.inv(r)@P.T
# A.append(B)
# A=np.array(A)
# print(A)
# # dict={'x':A.T[0],'y':A.T[1],'z':A.T[2],"img":img}
# # df = pd.DataFrame(dict)
# # df.to_csv(r'C:\Users\ivsrs\Documents\AirSim\2021-05-07-22-37-05\Cordinate.csv',index=False)
###################################################################################################################################################


import numpy as np
import math as m
import pandas as pd
# def Rx(theta):
#   return np.matrix([[ 1, 0           , 0           ],
#                    [ 0, m.cos(theta),-m.sin(theta)],
#                    [ 0, m.sin(theta), m.cos(theta)]])
  
# def Ry(theta):
#   return np.matrix([[ m.cos(theta), 0, m.sin(theta)],
#                    [ 0           , 1, 0           ],
#                    [-m.sin(theta), 0, m.cos(theta)]])
  
# def Rz(theta):
#   return np.matrix([[ m.cos(theta), -m.sin(theta), 0 ],
#                    [ m.sin(theta), m.cos(theta) , 0 ],
#                    [ 0           , 0            , 1 ]])
# phi = m.pi/2
# theta = m.pi/4
# psi = m.pi/2
# print("phi =", phi)
# print("theta  =", theta)
# print("psi =", psi)C:\Users\ivsrs\Documents\AirSim\2021-12-02-17-24-13\2021-12-02-17-24-13_sub.csv
  
  
# R = Rz(psi) * Ry(theta) * Rx(phi)
from scipy.spatial.transform import Rotation as R
A=[]
# df=pd.read_csv(r"C:\Users\ivsrs\Documents\AirSim\2021-12-02-17-24-13\2021-12-02-17-24-13_sub.csv");
df=pd.read_csv(r"/home/ivsr/Documents/AirSim/2022-12-21-18-37-47/airsim_rec.txt", sep="\t",header= 0);
# print(df)
row,colum=df.shape
for i in range(row):
  # calib x , y in real environment

  x = -df.POS_X[i]
  y = -df.POS_Y[i]
  # P=np.array([df.POS_X[i],df.POS_Y[i],df.POS_Z[i]])
  P=np.array([x,y,0.1 - df.POS_Z[i]])
  qr= df.Q_W[i]
  
  qx=df.Q_X[i]
  qy=df.Q_Y[i]
  qz=df.Q_Z[i]
  img=df.ImageFile

  r = R.from_quat([qx,qy, qz, qr])
  
  # print(np.linalg.inv(r))
# print("P= {}".format(P))
# print("qx=%s, qy=%s, qz=%s, qw=%s"%(qx,qy,qz,qr))

  r=r.as_matrix()
  print(r)
# print(r)

# print("Toa do =%s"%(np.linalg.inv(r)@P.T))
  print(P)
  B=np.linalg.inv(r)@P.T
  print(np.linalg.inv(r))
  print(B)
  A.append(B)
A=np.array(A)

# print(A)
dict={'x':A.T[0],'y':A.T[1],'z':A.T[2],"img":img}
# print(dict)
df = pd.DataFrame(dict)
df.to_csv(r'/home/ivsr/Documents/AirSim/2022-12-21-18-37-47/Cordinate_Noformat.csv',index=False)
df=pd.read_csv(r"/home/ivsr/Documents/AirSim/2022-12-21-18-37-47/Cordinate_Noformat.csv")
df.x=df.x.map('{:,.4f}'.format)
df.y=df.y.map('{:,.4f}'.format)
df.z=df.z.map('{:,.4f}'.format)
df.to_csv(r'/home/ivsr/Documents/AirSim/2022-12-21-18-37-47/2022-21-12_data.csv',index=False)

# df.to_csv(r'C:\Users\ivsrs\Documents\AirSim\2021-12-02-17-24-13\2021-12-02-17-24-13_data.csv',index=False)

  