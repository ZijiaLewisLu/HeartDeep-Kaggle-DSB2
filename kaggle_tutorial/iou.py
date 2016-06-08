import dicom, lmdb, cv2, re, sys
import os, fnmatch, shutil, subprocess
import numpy as np
np.random.seed(1234)
import caffe
import score

caffe.set_mode_gpu() # or caffe.set_mode_cpu() for machines without a GPU
try:
    del solver # it is a good idea to delete the solver object to free up memory before instantiating another one
    solver = caffe.SGDSolver('fcn_solver.prototxt')
except NameError:
    solver = caffe.SGDSolver('fcn_solver.prototxt')


img_train = solver.net.blobs['data'].data[0,0,...]
img_test = solver.test_nets[0].blobs['data'].data[0,0,...]

form = "IOU/{0}"

score.seg_tests(solver, form , range(100) , layer='score', gt= "label")
