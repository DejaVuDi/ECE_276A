# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 01:18:31 2019

@author: isgud
"""

import numpy as np
import matplotlib.pyplot as plt
from map_utils import mapCorrelation,bresenham2D
from PIL import Image

def initMap():  
    MAP = {}
    MAP['res']   = 0.05 #meters
    MAP['xmin']  = -35  #meters
    MAP['ymin']  = -35
    MAP['xmax']  =  35
    MAP['ymax']  =  35 
    MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
    MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
    MAP['map'] = np.log(0.5)*np.ones((MAP['sizex'],MAP['sizey']))
    MAP['track'] = np.log(0.5)*np.ones((MAP['sizex'],MAP['sizey']))
    MAP['texture'] = np.zeros([MAP['sizex'],MAP['sizey'],3])
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
	c = W[0] # weight_t1 is normalized
	for k in range(N):
		u = np.random.uniform(0, 1/N) #uniform distribution
		beta = u + k/N #scan each part in the circle
		while beta > c :
			j = j +1
			c = c + W[j] # increasing the decision section length
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

def plotmap(map):
    plt.figure(figsize=(10, 10))
    map01 = np.exp(map)
    map01[map01<0.2] = 0
    map01[map01>0.8] = 1
    plt.imshow(map01,cmap='gray')
    plt.clim(0,1)
    plt.savefig('check%d.png'%dataset)
    plt.show()

def savetexture(map,i):
    plt.figure(figsize=(10, 10))
    plt.imshow(map)
    #plt.colorbar()
    #plt.show()
    plt.savefig('texture_21/dataset %d'%dataset+'_'+str(i)+'_1.png')

###############################################################################
if __name__=='__main__':
    dataset = 21
      
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
    
    with np.load("Kinect%d.npz"%dataset) as data:
        disp_stamps = data["disparity_time_stamps"] # acquisition times of the disparity images # 2407
        rgb_stamps = data["rgb_time_stamps"] # acquisition times of the rgb images # 2289

    MAP=initMap()
    
    x_range = np.arange(-0.2, 0.2 + 0.05, 0.05)
    y_range = np.arange(-0.2, 0.2 + 0.05, 0.05)
    x_im = np.arange(MAP['xmin'], MAP['xmax'] + MAP['res'], MAP['res'])  # physical x-positions of each pixel of the map
    y_im = np.arange(MAP['ymin'], MAP['ymax'] + MAP['res'], MAP['res'])  # physical y-positions of each pixel of the map
    
    # initialize the particles
    N = 10
    # particle filter
    W = np.ones(N) / N # weight
    P = np.zeros([N,3]) # pose for particles
    corr = np.zeros(N)
    state = np.array([0,0,0])
    state_0 = state
    # find two clost then get average
    yaw = 0
    # lidar to body
    delta_x = 0.29833
    delta_z = 0.51435
    
    # initialize the map
    mt = MAP['map'] # mapping
    traj_map = np.zeros((MAP['sizex'],MAP['sizey'])) # track
    tmt = np.zeros((MAP['sizex'], MAP['sizey'], 3), dtype=np.uint8) # texture
    img_d=np.array([]) 
    # initialize the track
    trajx = []
    trajy = []
    count = [0,0,0,0,0]
    
    while (count[0]<encoder_stamps.shape[0]-1 and count[1]<imu_stamps.shape[0]-1 and count[2]<lidar_stamps.shape[0]-1 and count[3]<disp_stamps.shape[0]-1 and count[4]<rgb_stamps.shape[0]-1):
        arg = np.argmin([encoder_stamps[count[0]],imu_stamps[count[1]],lidar_stamps[count[2]],disp_stamps[count[3]],rgb_stamps[count[4]]])
        count[arg] = count[arg] + 1
        # encoder
        if arg == 0:
            # time difference
            tau = encoder_stamps[count[0]+1] - encoder_stamps[count[0]]     
            [X, Y, theta] = state
            u = drive_model(yaw,encoder_counts[:,count[0]+1],tau)
            state_0 = state
            state = state + u
            P = P + u + np.random.randn(N,3)*[1e-2,1e-2,1e-3]
            # trajectory
            xt = cell(state[0], MAP)
            yt = cell(state[1], MAP)
            xtp = cell(state_0[0], MAP)
            ytp = cell(state_0[1], MAP)
            x_i, y_i = bresenham2D(xtp,ytp,xt,yt).astype(int)
            MAP['track'][x_i,y_i] = -2000
        elif arg == 1:
            [_,_,yaw1] = imu_angular_velocity[:,count[1]]
            [_,_,yaw2] = imu_angular_velocity[:,count[1]-1]
            yaw = (yaw1 + yaw2) / 2
        elif arg == 2:
            P,state,W = update(count[2],P,MAP,N)
            MAP = mapping(count[2],state)
            if count[2]%100 ==0:  # show progress
                print(count[2],'lidar stamps')
        elif arg == 3:
            # read depth
            img_d = Image.open('dataRGBD/Disparity'+str(dataset)+'/disparity'+str(dataset)+'_'+str(count[3])+'.png')
            disparity_ = np.array(img_d.getdata(),np.uint16).reshape(img_d.size[1]*img_d.size[0])
        else:
            # read RGB
            rgb = plt.imread('dataRGBD/RGB'+str(dataset)+'/rgb'+str(dataset)+'_'+str(count[4])+'.png')
            [x_t,y_t,theta_t] = state
            m, n, _ = rgb.shape
            # resize
            disparity = disparity_
            rgb_vec = rgb.reshape((3, m*n))
            dd_ = 1.03/(-0.00304*disparity + 3.31)
            depth_mask = np.logical_and(dd_>0,dd_<5)
            grid = np.indices((m,n))
            u = grid[0]
            v = grid[1]
            u.resize((1,m*n))
            v.resize((1,m*n))
            img_ind = np.vstack((u,v))
            img_frame = np.vstack((u,v,np.ones(m*n)))

            # depth>0 and depth<5
            img_ind = img_ind[:,depth_mask]
            img_frame = img_frame[:,depth_mask]
            disparity = disparity[depth_mask]
            dd_ = dd_[depth_mask]
            
            K = np.matrix([[585.05108211, 0           , 242.94140713],
                           [0           , 585.05108211, 315.83800193],
                           [0           , 0           , 1           ]])
            matrix_pi = np.hstack((np.identity(3),np.zeros((3,1))))
            
            # tranform camera frame
            cam_frame = np.linalg.inv(K) * img_frame
            cam_frame = dd_ * np.asarray(cam_frame)
            cam_frame = np.vstack((cam_frame,np.ones(cam_frame.shape[1])))
            
            T_bo = np.matrix([[ 0.01965239,  0.35227423,  0.93569047,  0.33276   ],
                              [-0.99977951,  0.        ,  0.02099846,  0.        ],
                              [ 0.00739722, -0.93589682,  0.35219656,  0.38001   ],
                              [ 0.        ,  0.        ,  0.        ,  1.        ]])
    
            body_frame = T_bo * cam_frame
            
            # threshold height
            thresh_img = np.asarray(body_frame)
            mask = thresh_img[2,:]<1 # threshold
            thresh_img = body_frame[:, mask]
            thresh_ind = img_ind[:,mask]
            d_thres = disparity[mask]
            
            # transform from body to world frame
            T_wb = np.matrix([[np.cos(theta_t), -np.sin(theta_t),  0, x_t],
                              [np.sin(theta_t),  np.cos(theta_t),  0, y_t],
                              [0,                0,                1, 0],
                              [0,                0,                0, 1]])
            world_frame = T_wb * thresh_img
            
            # match color to depth
            rgbi = []
            rgbj = []
            for t in range(thresh_ind.shape[1]):
                [i, j] = thresh_ind[:,t]
                dd = (-0.00304*d_thres[t]+3.31)
                depth = 1.03/dd
                rgbi.append(np.round((i*526.37 + dd*(-4.5*1750.46) + 19276)/585.051).astype(int))
                rgbj.append(np.round((j*526.37 + 16662)/585.051).astype(int))

            rgbi = np.array(rgbi)
            rgbj = np.array(rgbj)
            mask_ind_i = np.logical_and((rgbi > 0),(rgbi <= m))
            mask_ind_j = np.logical_and((rgbj > 0),(rgbj <= n))
            mask_ind = np.logical_and(mask_ind_i,mask_ind_j)
            
            rgb_ind = np.vstack((rgbi, rgbj))
            thresh_rgb_ind = rgb_ind[:,mask_ind]
            world_frame_masked = world_frame[:,mask_ind]
            
            # map texture
            x_texture = cell(world_frame_masked[0,:],MAP)
            y_texture = cell(world_frame_masked[1,:],MAP)
            MAP['texture'][x_texture,y_texture,:] = rgb[thresh_rgb_ind[0,:], thresh_rgb_ind[1,:], :]
            
            # save map
            if count[4]%100 == 0: 
                print(count[4],'rgb stamps')
                savetexture(MAP['texture'],count[4])

    plotmap(MAP['track'] + MAP['map'] + 0.5)
    plt.figure(figsize=(10, 10))
    plt.imshow(MAP['texture'])