import numpy as np
from utils import *
from scipy.linalg import expm,block_diag
from numpy.linalg import inv,pinv

def invM(z,fsu,cu,fsv,cv,b):
    [ul,vl,ur,vr] = z
    return np.array([(ul-cu)/(ul-ur)*b,(vl-cv)/(ul-ur)*b,fsu*b/(ul-ur),1])

def hat(z):
    [v1,v2,v3,w1,w2,w3] = z
    return np.array([[0,-w3,w2,v1],
                     [w3,0,-w1,v2],
                     [-w2,w1,0,v3],
                     [0,0,0,0]])

def odot(z):
    dot = np.zeros((4*6,z.shape[1]))
    dot[[0,7,14],:] = z[3,:]
    dot[[4,5,9,11,15,16],:] = [z[2,:], -z[1,:], -z[2,:], z[0,:], z[1,:], -z[0,:]]
    return dot.reshape((4,6,z.shape[1]))

if __name__ == '__main__':
	filename = "./data/0027.npz"
	data = "0027"
	# filename = "./data/0042.npz"
	# data = "0042"
	t,features,linear_velocity,rotational_velocity,K,b,cam_T_imu = load_data(filename)

# (a) IMU Localization via EKF Prediction
	P = np.zeros((4,4))
	T = np.zeros((4,4,t.size-1))
	mu = np.identity(4) # for IMU

# (b) Landmark Mapping via EKF Update
	W = 1.5*np.identity(6)
	num_land = features.shape[1] # # of landmarks
	D = np.array([[1,0,0],
				   [0,1,0],
				   [0,0,1],
				   [0,0,0]])
	slam_D = np.kron(np.eye(num_land),D)
	z_0 = np.array([-1,-1,-1,-1]) # observation
	f_su = K[0,0]
	f_sv = K[1,1]
	c_u = K[0,2]
	c_v = K[1,2]
	M = np.array([[f_su,    0, c_u,       0],
				  [   0, f_sv, c_v,       0],
				  [f_su,    0, c_u, -f_su*b],
				  [   0, f_sv, c_v,       0]])

	sigma = np.identity(3*num_land+6)

	count = 0
	map_mu = -1*np.ones((4,num_land)) # for landmark

	temp = -1*np.ones((4,features.shape[1]))
	z_t = -1*np.ones((4,features.shape[1]))
	z_hat = -1*np.ones((4,features.shape[1]))

	map_V = 200*np.identity(4)
	
# (c) Visual-Inertial SLAM (Extra Credit)
	slam_V = 1*np.identity(4*num_land)

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
		sigma[3*num_land:3*num_land+6,3*num_land:3*num_land+6] = expm(-tau[0]*u_vee)@sigma[3*num_land:3*num_land+6,3*num_land:3*num_land+6]@(expm(-tau[0]*u_vee)).T+tau[0]**2*W

	# (b) Landmark Mapping via EKF Update & (c) Visual-Inertial SLAM (Extra Credit)
		z = features[:,:,i+1] # 4*206
		H_t=np.zeros((4,3))
		slam_H = np.zeros((4*num_land,3*num_land+6))
		
		for j in range(num_land):
			if np.logical_and(((z[:,j] != z_0).all()),((temp[:,j] == z_0).all())):
				# initialization
				temp[:,count] = z[:,j] # temp
				landmark_world = inv(mu)@inv(cam_T_imu)@invM(temp[:,count],f_su,c_u,f_sv,c_v,b)
				map_mu[:,count] = landmark_world
				count = count + 1
			elif np.logical_and((z[:,j] != z_0).all(),(temp[:,j] != z_0).all()):
				# update
				z_t[:,j] = z[:,j]
				q = cam_T_imu@mu@map_mu[:,j]
				z_hat[:,j] = M@(q/q[2])
				d_q = (1/q[2])*np.array([[1,0,-q[0]/q[2],0],
										 [0,1,-q[1]/q[2],0],
										 [0,0,0,0],
										 [0,0,-q[3]/q[2],1]])
				H_t = M@d_q@cam_T_imu@mu@D
				# K_t = map_sigma[j]@H_t.T@pinv(H_t@map_sigma[j]@H_t.T+map_V)
				# map_mu[:,j] = map_mu[:,j] + D@K_t@(z_t[:,j]-z_hat[:,j])
				# map_sigma[j] = (np.identity(3)-K_t@H_t)@map_sigma[j]
			slam_H[(4*j):(4*j+4),(3*j):(3*j+3)] = H_t

	# (c) Visual-Inertial SLAM (Extra Credit)	
		# z_1 = features[:,:,i+1]
		# slam_H = np.zeros((4*num_land,6))
		# H__t = np.zeros((4,6))
		slam = mu@map_mu
		slam_q = cam_T_imu@slam
		z_hat = M@(slam_q/slam_q[2,:])
		dot = odot(slam)
		for o in range(num_land):
			if (map_mu[:,o] == z_0).all():
				slam_H[(4*o):(4*o+4),(3*num_land):(3*num_land+6)] = np.zeros((4,6))
			else:
				another = mu@map_mu[:,o]
				one = cam_T_imu@mu@map_mu[:,o]		
				slam_d_q = (1/one[2])*np.array([[1,0,-one[0]/one[2],0],
												[0,1,-one[1]/one[2],0],
												[0,0,0,0],
												[0,0,-one[3]/one[2],1]])
				slam_H[(4*o):(4*o+4),(3*num_land):(3*num_land+6)] = M@slam_d_q@cam_T_imu@dot[:,:,o]
			# slam_H[(4*o):(4*o+4),(3*num_land):(3*num_land+6)] = H__t
		
		slam_K = sigma@slam_H.T@pinv(slam_H@sigma@slam_H.T+slam_V)
		mask = np.all((z+1)!=0,axis=0)
		K = slam_K@((((z-z_hat)*mask).swapaxes(0,1)).reshape((4*num_land,1)))
		mu = expm(hat(K[num_land*3:num_land*3+6,:]))@mu
		map_mu = map_mu + (slam_D@K[0:3*num_land,:]).reshape((4,num_land))

		sigma = (np.identity(3*num_land+6)-slam_K@slam_H)@sigma

		if i%100 == 0:
			print('process: T = '+str(i))

		P = inv(mu)
		T[:,:,i] = P

		if i == t.size-2:
			print('Finish')

	visualize_trajectory_2d(T,map_mu,path_name=data)