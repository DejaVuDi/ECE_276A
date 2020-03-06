import numpy as np
from utils import *
from scipy.linalg import expm
from numpy.linalg import inv,pinv


if __name__ == '__main__':
	filename = "./data/0027.npz"
	data = "0027"
	# filename = "./data/0042.npz"
	# data = "0042"
	t,features,linear_velocity,rotational_velocity,K,b,cam_T_imu = load_data(filename)

# (a) IMU Localization via EKF Prediction
	P = np.zeros((4,4))
	xi = np.ones((6,6))
	T = np.zeros((4,4,t.size-1))
	mu = np.identity(4)
	sigma = np.identity(6)

# (b) Landmark Mapping via EKF Update
	W = 1*np.identity(6)
	num_land = features.shape[1] # # of landmarks
	delta = np.random.normal(0,1,(3,1))
	D = np.array([[1,0,0],
				   [0,1,0],
				   [0,0,1],
				   [0,0,0]])
	z_0 = np.array([-1,-1,-1,-1]) # observation
	f_su = K[0,0]
	f_sv = K[1,1]
	c_u = K[0,2]
	c_v = K[1,2]
	M = np.array([[f_su,    0, c_u,       0],
				  [   0, f_sv, c_v,       0],
				  [f_su,    0, c_u, -f_su*b],
				  [   0, f_sv, c_v,       0]])
	s = 2
	map_sigma = s*np.identity(3*num_land)
	count = 0
	landmarks = np.zeros((4,num_land))
	z_t = -1*np.ones((4,features.shape[1]))
	
	for i in range(t.size-1):
	# (a) IMU Localization via EKF Prediction
		tau = t[:,i+1]-t[:,i]
		w = rotational_velocity[:,i]
		w_hat = np.array([[0,-w[2],w[1]],
						   [w[2],0,-w[0]],
						   [-w[1],w[0],0]])
		v = linear_velocity[:,i]
		v_hat = np.array([[0,-v[2],v[1]],
						   [v[2],0,-v[0]],
						   [-v[1],v[0],0]])
		v = v.reshape((-1,1))
		w = w.reshape((-1,1))
		u_t = np.vstack((v,w))
		u_vee = np.vstack((np.hstack((w_hat,v_hat)),np.hstack((np.zeros((3,3)),w_hat))))
		u_hat = np.vstack((np.hstack((w_hat,v)),np.zeros((1,4))))
		mu = expm(-tau[0]*u_hat)@mu
		sigma = expm(-tau[0]*u_vee)@sigma@(expm(-tau[0]*u_vee)).T+tau[0]**2*W
		P = inv(mu)
		T[:,:,i] = P

	# (b) Landmark Mapping via EKF Update
		z = features[:,:,i] # 4*206
		for j in range(num_land):
			if np.logical_and(((z[:,j] != z_0).all()),((z_t[:,j] == z_0).all())):
				z_t[:,count] = z[:,j]
				landmark_img = pinv(M)@z_t[:,count]
				landmark_img = landmark_img/landmark_img[3]
				landmark_world = P@inv(cam_T_imu)@landmark_img
				landmarks[:,count] = landmark_world
				count = count +1

	visualize_trajectory_2d(T,landmarks,path_name=data)