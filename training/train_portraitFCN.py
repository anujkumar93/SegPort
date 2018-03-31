from __future__ import division
import os
caffe_root = '../caffe-portraitseg/'
import sys
sys.path.insert(0,caffe_root + 'python')
import caffe
import numpy as np

# make a bilinear interpolation kernel
# credit @longjon
def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)

# set parameters s.t. deconvolutional layers compute bilinear interpolation
# N.B. this is for deconvolution without groups
def interp_surgery(net, layers):
    for l in layers:
        m, k, h, w = net.params[l][0].data.shape
        if m != k:
            print 'input + output channels need to be the same'
            raise
        if h != w:
            print 'filters need to be square'
            raise
        filt = upsample_filt(h)
        net.params[l][0].data[range(m), range(k), :, :] = filt

# init
caffe.set_mode_gpu()
caffe.set_device(2)

MODEL_FILE = './FCN8s_models/fcn-8s-pascal-deploy.prototxt'
PRETRAINED = './FCN8s_models/fcn-8s-pascal.caffemodel'

net = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)

solverpath = './model_files/solver_portraitFCN.prototxt'
solver = caffe.SGDSolver(solverpath)

# do net surgery to set the deconvolution weights for bilinear interpolation
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
interp_surgery(solver.net, interp_layers)

# copy base weights for fine-tuning
#solver.net.copy_from(base_weights)
solver.net.params['conv1_1'][0].data[:,0:3:1,:,:] = net.params['conv1_1'][0].data[:,:,:,:]

layerkeys = ['conv1_2', 'conv2_1', 'conv2_2','conv3_1','conv3_2','conv3_3','conv4_1','conv4_2','conv4_3','conv5_1','conv5_2','conv5_3', 'conv1_2','fc6','fc7']
for key in layerkeys:
    solver.net.params[key][0].data[...] = net.params[key][0].data[...]


# also copy other weights from the net
solver.net.params['score-fr'][0].data[:,:,:,:] = net.params['score-fr'][0].data[0:15:15,:,:,:]

#score2
solver.net.params['score2'][0].data[:,:,:,:] = net.params['score2'][0].data[0:15:15,0:15:15,:,:]
#score-pool4
solver.net.params['score-pool4'][0].data[:,:,:,:] = net.params['score-pool4'][0].data[0:15:15,:,:,:]
#score4
solver.net.params['score4'][0].data[:,:,:,:] = net.params['score4'][0].data[0:15:15,0:15:15,:,:]
#score-pool3
solver.net.params['score-pool3'][0].data[:,:,:,:] = net.params['score-pool3'][0].data[0:15:15,:,:,:]
#upsample
solver.net.params['upsample'][0].data[:,:,:,:] = net.params['upsample'][0].data[0:15:15,0:15:15,:,:]


solver.step(80000)
