# this was written by my friend Hsuan Chen Liu. I liked his implementation much 
# better but didn't have time to implement it myself before the deadline. keeping 
# it here for future reference

import numpy as np
from scipy.interpolate import RectBivariateSpline

from scipy.ndimage.interpolation import affine_transform


def InverseCompositionAffine(It, It1):
	# Input: 
	#	It: template image
	#	It1: Current image

	# Output:
	#	M: the Affine warp matrix [2x3 numpy array]

    # put your implementation here
	M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0],[0,0,1]])
	p0 = np.zeros(6)
	p = p0.astype(np.double) #p+=delta_p,will be different from p=p+delta_p if p is not set to double here
	#p[5] = 2.5	#test

	#get image It1 coordinates, flatten and make homogeneous
	H = It1.shape[0]
	W = It1.shape[1]
	x = np.linspace(0,W-1,W)
	y = np.linspace(0,H-1,H)
	xv, yv = np.meshgrid(x, y)
	xv = xv.ravel()
	yv = yv.ravel()
	#print(xv.shape)
	# It1gradx = np.gradient(It1,axis=1)
	# It1grady = np.gradient(It1,axis=0)

	#STEP3: Evaluate gradient of the template T(x)
	Tgradx = np.gradient(It,axis=1).ravel()
	Tgrady = np.gradient(It,axis=0).ravel()
	Tgrad = np.stack((Tgradx,Tgrady)).T
	#print(Tgrad.shape)

	#STEP4: Evaluate the Jacobian dW/dp at (x;0)
	#STEP5: Compute the steepest descent images
	steepDes = np.empty([Tgrad.shape[0],6])
	steepDes[:,0] = Tgrad[:,0] * xv
	steepDes[:,1] = Tgrad[:,0] * yv
	steepDes[:,2] = Tgrad[:,0] * 1
	steepDes[:,3] = Tgrad[:,1] * xv
	steepDes[:,4] = Tgrad[:,1] * yv
	steepDes[:,5] = Tgrad[:,1] * 1 
	#print('steepDes',steepDes.shape)
	
	#STEP6: Compute the Hessian
	Hessian = np.matmul(steepDes.T,steepDes)
	#print('Hessian',Hessian.shape)
		

	e = 0.5 #threshold
	delta_p = [e,e,e,e,e,e]	#set to a value greater than e first so loop will start
	#print(np.linalg.norm(delta_p))
	while np.linalg.norm(delta_p) > e:
		#STEP1: warp coordinates of rectangle in I with W(x;p)
		#Warp = np.array([[1,0,-p[0]],[0,1,-p[1]]])		#negative translation because shifting image to rectangle position instead of shifting rectangle
		#M = np.array([[1-p[0],-p[1],-p[2]],[-p[3],1-p[4],-p[5]],[0,0,1]])
		#M = np.array([[1+p[4],p[3],p[5]],[p[1],1+p[0],p[2]],[0,0,1]]) 	#flipped columns and rows because affine_transform uses flipped xy ordering
		It1w = affine_transform(It1,M)
		mask = np.ones_like(It1)
		maskw = affine_transform(mask,M)	#not using here
		#print('It1w',It1w.shape)
		#print(maskw)
		#b = (It * maskw - It1w).ravel()
		b = (It1w- It).ravel()
		#print(b.shape)
		# gradxw = affine_transform(It1gradx,M).ravel()
		# gradyw = affine_transform(It1grady,M).ravel()
		# gradw = np.stack((gradxw,gradyw)).T
		#print('gradw',gradw.shape)

		#Jacobian = np.array([[xx[i],yy[i],1,0,0,0],[0,0,0,xx[i],yy[i],1]])
		#steepDes = np.empty([gradw.shape[0],6])
		# print(xv.shape)
		# steepDes[:,0] = np.vstack((xv,np.zeros(gradw.shape[0])))
		# print(steepDes)
		# print(gradw[:,0].shape)

		# steepDes[:,0] = gradw[:,0] * xv
		# steepDes[:,1] = gradw[:,0] * yv
		# steepDes[:,2] = gradw[:,0] * 1
		# steepDes[:,3] = gradw[:,1] * xv
		# steepDes[:,4] = gradw[:,1] * yv
		# steepDes[:,5] = gradw[:,1] * 1
		# Hessian = np.matmul(steepDes.T,steepDes)
		#print('Hessian',Hessian.shape)
		#print('steepDes',steepDes)
		
		#STEP7:
		#print('b',b.shape)
		productArray = steepDes.T * b
		#print(productArray.shape)
		product = np.sum(productArray,axis=1).reshape(6,1)
		#print(product.shape)

		#STEP8: compute delta_p
		delta_p = np.matmul(np.linalg.inv(Hessian),product)
		delta_p = delta_p.reshape(6,)

		#print('delta',delta_p)

		#STEP9: update parameters
		#p = p + delta_p
		#delta_p = 0.01 #test-to run only for one loop
		#print('p',p)
		delta_M = np.array([[1+delta_p[4],delta_p[3],delta_p[5]],[delta_p[1],1+delta_p[0],delta_p[2]],[0,0,1]])
		M = np.matmul(M,np.linalg.inv(delta_M))
		#M = np.array([[1+p[4],p[3],p[5]],[p[1],1+p[0],p[2]],[0,0,1]])
	return M


# #for testing
if __name__ == '__main__':
	#im = np.load('../data/carseq.npy')
	im = np.load('../data/aerialseq.npy')

	It = im[:,:,0]
	It1 = im[:,:,1]
	#print('It1',It1.shape)
	#rect = np.array([59,116,145,151]).reshape(4,1)
	#s = time.time()
	M = InverseCompositionAffine(It,It1)
	print('M',M)
	#print('time',time.time()-s)