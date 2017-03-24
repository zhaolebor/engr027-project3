#!/usr/bin/env python

import cv2
import numpy
import sys
import os

###############################################################################
def fixKeyCode(code):
    return numpy.uint8(code).view(numpy.int8)
###############################################################################

# Get command line arguments or print usage and exit
if len(sys.argv) > 2:
    proj_file = sys.argv[1]
    cam_file = sys.argv[2]
else:
    progname = os.path.basename(sys.argv[0])
    print >> sys.stderr, 'usage: '+progname+' PROJIMAGE CAMIMAGE'
    sys.exit(1)

# Load in our images as grayscale (1 channel) images
proj_image = cv2.imread(proj_file, cv2.IMREAD_GRAYSCALE)
cam_image = cv2.imread(cam_file, cv2.IMREAD_GRAYSCALE)

# Make sure they are the same size.
assert(proj_image.shape == cam_image.shape)

# Set up parameters for stereo matching (see OpenCV docs at
# http://goo.gl/U5iW51 for details).
min_disparity = 0
max_disparity = 16
window_size = 11
param_P1 = 0
param_P2 = 20000

# Create a stereo matcher object
matcher = cv2.StereoSGBM_create(min_disparity,
                                max_disparity,
                                window_size,
                                param_P1,
                                param_P2)

# Compute a disparity image. The actual disparity image is in
# fixed-point format and needs to be divided by 16 to convert to
# actual disparities.
disparity = matcher.compute(cam_image, proj_image) / 16.0

f = 600
u0 = 320
v0 = 240
b = 0.05
z_max = 8
K = numpy.array([[f,0,u0],[0,f,v0],[0,0,1]])
Kinv = numpy.linalg.inv(K)
w = cam_image.shape[1]
h = cam_image.shape[0]
delta_min = b*f/z_max

# explicit for loop version
points = []
"""
for i in range(h):
    for j in range(w):
        q = numpy.array([j, i, 1])
        new_point = numpy.dot(Kinv,q)
        if disparity[i][j] > delta_min:
            z = b*f/disparity[i][j]
            actual_point = new_point/new_point[2]*z
            points.append(actual_point)
        if (i==120 and j==40) or (i==300 and j==40) or (i==240 and j==320):
            print z
            print i,j, actual_point
"""
# vectorized version

#create the grid
Xrange = numpy.linspace(0, w-1, w).astype('float32')
Yrange = numpy.linspace(0,h-1,h).astype('float32')

X,Y = numpy.meshgrid(Xrange, Yrange)

# initialize an empty array for Z
Z = numpy.ones_like(X)


#put it together
xyz = numpy.hstack( ( X.reshape((-1,1)),
                      Y.reshape((-1,1)),
                      Z.reshape((-1,1))))

xyz = numpy.transpose(xyz)
new_xyz = numpy.dot(Kinv, xyz)
X = new_xyz[0]
Y = new_xyz[1]

mask = numpy.greater(disparity, delta_min)
Z[mask] = b*f/disparity[mask]
Z = Z.reshape((-1,1))
X[mask] = X*Z
Y[mask] = Y*Z

final_xyz = numpy.hstack( ( X[mask].reshape((-1,1)),
                            Y[mask].reshape((-1,1)),
                            Z[mask].reshape((-1,1)) ) )

print final_xyz, final_xyz.shape




#print xyz







#numpy.savez('starter.npz',points)

# Pop up the disparity image.
cv2.imshow('Disparity', disparity/disparity.max())
while fixKeyCode(cv2.waitKey(5)) < 0:
    pass
