# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 01:18:31 2019

@author: isgud
"""

import numpy as np
import matplotlib.pyplot as plt
from map_utils import mapCorrelation,bresenham2D

def initMap():  
    MAP = {}
    MAP['res']   = 0.05 #meters
    MAP['xmin']  = -35  #meters
    MAP['ymin']  = -35
    MAP['xmax']  =  35
    MAP['ymax']  =  35 
    MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
    MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1)) #cells
    MAP['map'] = np.log(0.5)*np.ones((MAP['sizex'],MAP['sizey']))
    MAP['track'] = np.log(0.5)*np.ones((MAP['sizex'],MAP['sizey']))
    return MAP

def drive_model (yaw, counts, tau):
    [FR,FL,RR,RL] = counts
    # differential-drive model
    v = ((FL + RL) / 2 * 0.0022 + (FR + RR) / 2 * 0.0022) / 2 /tau
    d = yaw * tau / 2
    x_t = v * tau * sinc(d) * np.cos(theta + d)
    y_t = v * tau * sinc(d) * np.sin(theta + d)
    u = np.array([x_t, y_t, yaw*tau])
    return u

def softmax(a):
    return np.exp(a-a.max()) / np.exp(a-a.max(axis=0)).sum()

def sinc(a):
    return np.sin(a)/(1e-9+a)

def update (t, P, MAP, N):
    # remove points
    angles = np.arange(lidar_angle_min,lidar_angle_max+lidar_angle_increment-1e-3,lidar_angle_increment)
    ranges = lidar_ranges[:,t]
    valid = np.logical_and((ranges < lidar_range_max), (ranges > lidar_range_min))
    angles = angles[valid]
    ranges = ranges[valid]
   
    # from lidar frame to world frame
    x_b = ranges * np.cos(angles)
    y_b = ranges * np.sin(angles)
    
    for i in range(N):
        [X,Y,theta] = P[i,:]
        x_w = x_b * np.cos(theta) - y_b * np.sin(theta) + X
        y_w = x_b * np.sin(theta) + y_b * np.cos(theta) + Y
        vp = np.stack((x_w,y_w))
        corr[i] = np.max(mapCorrelation(MAP['map'],x_im,y_im,vp,x_range,y_range))
   
    W = softmax(corr)
    ind = np.argmax(W)
    [X,Y,theta] = P[ind,:]
    state = [X,Y,theta]
    
    # resample
    if 1/(W**2).sum() < 0.1:
        W, P = resampling (W, P, N)
# =============================================================================
#     print(W)
# =============================================================================
    return P,state,W

def resampling (W, P, N):
	new_P = np.zeros([N,3])
	new_weight = np.ones(N) / N
	j = 0
	c = W[0]
	for k in range(N):
		u = np.random.uniform(0, 1/N) #uniform distribution
		beta = u + k/N 
		while beta > c :
			j = j +1
			c = c + W[j]
		new_P[k,:] = P[j,:] 
	return new_weight, new_P

def mapping (t, state):
# mapping
    # remove points
    angles = np.arange(lidar_angle_min,lidar_angle_max+lidar_angle_increment-1e-3,lidar_angle_increment)
    ranges = lidar_ranges[:,t]
    valid = np.logical_and((ranges < lidar_range_max), (ranges > lidar_range_min))
    angles = angles[valid]
    ranges = ranges[valid]
   
    # from lidar frame to world frame
    x_b = ranges * np.cos(angles)
    y_b = ranges * np.sin(angles)
    x_w = x_b * np.cos(theta) - y_b * np.sin(theta) + X
    y_w = x_b * np.sin(theta) + y_b * np.cos(theta) + Y
    x = cell(X, MAP) # location of particle
    y = cell(Y, MAP)
    x_s = cell(x_w, MAP) # locaiton of sensors beam
    y_s = cell(y_w, MAP)
    
    # build map
    for k in range(x_s.shape[0]):
        xb, yb = bresenham2D(x,y,x_s[k],y_s[k]) 
        indGood = np.logical_and((xb > 1), (yb > 1))
        MAP['map'][xb[indGood].astype(np.int16), yb[indGood].astype(np.int16)] += np.log(4) # MAP = 1
        
    indGood = np.logical_and((x_s > 1), (y_s > 1))
    MAP['map'][x_s[indGood], y_s[indGood]] -= np.log(4) # MAP = 0
    MAP['map'][MAP['map']>0] = 0
    return MAP

def cell (d, MAP):# convert from real position(meters) to map position(cells)
    return np.ceil((d - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1

###############################################################################
if __name__=='__main__':
    dataset = 23
      
    with np.load("Encoders%d.npz"%dataset) as data:
        encoder_counts = data["counts"] # 4 x n encoder counts # (4,4956)
        encoder_stamps = data["time_stamps"] # encoder time stamps # 4956
    
    with np.load("Hokuyo%d.npz"%dataset) as data:
        lidar_angle_min = data["angle_min"] # start angle of the scan [rad] # -2.35
        lidar_angle_max = data["angle_max"] # end angle of the scan [rad] # 2.35
        lidar_angle_increment = data["angle_increment"] # angular distance between measurements [rad] # 0.00436
        lidar_range_min = data["range_min"] # minimum range value [m] # 0.1
        lidar_range_max = data["range_max"] # maximum range value [m] # 30
        lidar_ranges = data["ranges"]       # range data [m] (Note: values < range_min or > range_max should be discarded) # (1081,4962)
        lidar_stamps = data["time_stamps"]  # acquisition times of the lidar scans # 4962
        
    with np.load("Imu%d.npz"%dataset) as data:
        imu_angular_velocity = data["angular_velocity"] # angular velocity in rad/sec # (3,12187)
        imu_linear_acceleration = data["linear_acceleration"] # Accelerations in gs (gravity acceleration scaling) # (3,12187)
        imu_stamps = data["time_stamps"]  # acquisition times of the imu measurements # 12187

    MAP=initMap()
    
    x_range = np.arange(-0.2, 0.2 + 0.05, 0.05)
    y_range = np.arange(-0.2, 0.2 + 0.05, 0.05)
    x_im = np.arange(MAP['xmin'], MAP['xmax'] + MAP['res'], MAP['res'])  # physical x-positions of each pixel of the map
    y_im = np.arange(MAP['ymin'], MAP['ymax'] + MAP['res'], MAP['res'])  # physical y-positions of each pixel of the map
    
    # initialize the particles
    N = 1
    # particle filter
    W = np.ones(N) / N # weight
    P = np.zeros([N,3]) # pose for particles
    corr = np.zeros(N)
    state = np.array([0,0,0])
    state_0 = state
    # find two close then get average
    yaw = 0
    # lidar to body
    delta_x = 0.29833
    delta_z = 0.51435

    # initialize the track
    traj = []
    count = [0,0,0]
    
    while (count[0]<encoder_stamps.shape[0]-1 and count[1]<imu_stamps.shape[0]-1 and count[2]<lidar_stamps.shape[0]-1):
        arg = np.argmin([encoder_stamps[count[0]],imu_stamps[count[1]],lidar_stamps[count[2]]])
        count[arg] = count[arg] + 1
        # encoder
        if arg == 0:
            # time difference
            tau = encoder_stamps[count[0]+1] - encoder_stamps[count[0]]     
            [X, Y, theta] = state
            u = drive_model(yaw,encoder_counts[:,count[0]+1],tau)
            state_0 = state
            state = state + u
            # trajectory
            xt = cell(state[0], MAP)
            yt = cell(state[1], MAP)
            xtp = cell(state_0[0], MAP)
            ytp = cell(state_0[1], MAP)
            x_i, y_i = bresenham2D(xtp,ytp,xt,yt).astype(int)
            MAP['track'][x_i,y_i] = -2000
            traj.append(state)
        elif arg == 1:
            [_,_,yaw1] = imu_angular_velocity[:,count[1]]
            [_,_,yaw2] = imu_angular_velocity[:,count[1]-1]
            yaw = (yaw1 + yaw2) / 2
        else:
            MAP = mapping(count[2],state)
            if count[2]%100 ==0:  # show progress
                print(count[2],'lidar stamps')
    plt.figure(figsize=(10, 10))
    trajectory = np.asarray(traj)
    plt.plot(trajectory[:,0],trajectory[:,1],'.k')
    plt.savefig('dead%d.png'%dataset)