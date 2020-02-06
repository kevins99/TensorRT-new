import time
from typing import List

import torch
import numpy as np

from model.SSH import SSH
from model.network import load_check_point
from model.utils.config import cfg
from model.nms.nms_wrapper import nms
from model.utils.test_utils import _get_image_blob, _compute_scaling_factor

path = 'check_point.zip'
net = SSH(vgg16_image_net=False)
net.eval()
net.cuda()
check_point = load_check_point(path)
net.load_state_dict(check_point['model_state_dict'])

import cv2
# img=torch.tensor(np.transpose(np.array([cv2.imread("small.jpeg")]),(0,3,1,2)),dtype=torch.float32)
img=cv2.imread("0_Parade_marchingband_1_20.jpg")
print(img.shape)

# img=np.random.randn(480,640,3)


def preprocessImage(im):
        """
        Perform basic processing on input image such as 
        scaling the image to appropriate sizes

        Arguments:
            im {[numpy.ndarray]} -- The image/frame to be processed by SSH detector

        Returns:
            im_info [torch.Tensor] -- array containing shapes of image_blobs
            im_data [torch.Tensor] -- the actual image data extracted from the image_blob
            im_scale [int] -- The scale factor
        """
        im_scale = _compute_scaling_factor(
            im.shape, cfg.TEST.SCALES[0], cfg.TEST.MAX_SIZE)
        im_blob = _get_image_blob(im, [im_scale])[0]
        im_info = np.array(
            [[im_blob['data'].shape[2], im_blob['data'].shape[3], im_scale]])
        im_data = im_blob['data']

        im_info = torch.from_numpy(im_info).cuda()
        im_data = torch.from_numpy(im_data).cuda()
        return im_info, im_data, im_scale
def preprocessImage1(im):
        """
        Perform basic processing on input image such as 
        scaling the image to appropriate sizes

        Arguments:
            im {[numpy.ndarray]} -- The image/frame to be processed by SSH detector

        Returns:
            im_info [torch.Tensor] -- array containing shapes of image_blobs
            im_data [torch.Tensor] -- the actual image data extracted from the image_blob
            im_scale [int] -- The scale factor
        """
        im_scale = _compute_scaling_factor(
            im.shape, cfg.TEST.SCALES[0], cfg.TEST.MAX_SIZE)
        im_blob = _get_image_blob(im, [im_scale])[0]
        im_info = np.array(
            [[im_blob['data'].shape[2], im_blob['data'].shape[3], im_scale]])
        im_data = im_blob['data']

        im_info = torch.from_numpy(im_info).cuda()
        im_data = torch.from_numpy(im_data)
        return im_info, im_data, im_scale

_,imu,_ = preprocessImage1(img)
_,im,_=preprocessImage(img)
print(im.shape)

torch.onnx.export(net,im,"ssh.onnx",verbose=False)


import os
import sys
import cv2
import time
import ctypes
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt




TRT_LOGGER = trt.Logger(trt.Logger.WARNING)       
runtime = trt.Runtime(TRT_LOGGER)

model_path = 'ssh.onnx'


def print_network(network):
    for i in range(network.num_layers):
        layer = network.get_layer(i)

        print("\nLAYER {}".format(i))
        print("===========================================")
        layer_input = layer.get_input(0)
        if layer_input:
            print("\tInput Name:  {}".format(layer_input.name))
            print("\tInput Shape: {}".format(layer_input.shape))

        layer_output = layer.get_output(0)
        if layer_output:
            print("\tOutput Name:  {}".format(layer_output.name))
            print("\tOutput Shape: {}".format(layer_output.shape))
        print("===========================================")
def build_engine(model_path):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(flags = 1) as network, \
    trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_workspace_size = 1<<32
        builder.max_batch_size = 1
        builder.fp16_mode = 1

        
        with open(model_path, 'rb') as f:
            value = parser.parse(f.read())
            print("Parser: ", value)
           
        engine = builder.build_cuda_engine(network)
        # print_network(network)

        print(engine)
        return engine

engine = build_engine(model_path)

# buf = engine.serialize()
# with open("ssh.engine", 'wb') as f:
#     f.write(buf)


# create buffer






context = engine.create_execution_context()

stream = cuda.Stream()
# print(engine.get_binding_shape(0),engine.get_binding_shape(1),engine.get_binding_shape(2))
print(imu[0].shape,"imu shape")
h_input = cuda.pagelocked_empty(trt.volume((3,480,640)), dtype = np.float32)
h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=np.float32)
# Allocate device memory for inputs and outputs.
d_input = cuda.mem_alloc(h_input.nbytes)
d_output = cuda.mem_alloc(h_output.nbytes)
bindings = [int(d_input), int(d_output)]
# Create a stream in which to copy inputs/outputs and run inference.
stream = cuda.Stream()


print(np.ascontiguousarray(np.array(imu[0])).shape)

with engine.create_execution_context() as context:
        # Transfer input data to the GPU.
        cuda.memcpy_htod_async(d_input,np.ascontiguousarray(np.array(imu[0])), stream)
        # Run inference.
        context.execute_async(bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        # Synchronize the stream
        stream.synchronize()
        # Return the host output. 
        print(h_output.shape)
        print(np.argmax(h_output))
        print(h_output[np.argmax(h_output)])

# h_output=h_output.reshape(1,18000,5)
# for j in h_output:
#     for i in j:
#         if(i[0]>0 or i[4]>0):
#             print(i)
# print(h_output[0:35])
        

# import sys
# sys.exit(0)

# host_inputs  = []
# cuda_inputs  = []
# host_outputs = []
# cuda_outputs = []
# bindings = []
# stream = cuda.Stream()

# for binding in engine:
#     size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
#     host_mem = cuda.pagelocked_empty(size, np.float32)
#     cuda_mem = cuda.mem_alloc(host_mem.nbytes)

#     bindings.append(int(cuda_mem))
#     if engine.binding_is_input(binding):
#         print("INPUT")
#         host_inputs.append(host_mem)
#         cuda_inputs.append(cuda_mem)
#     else:
#         print("OUTPUT")
#         host_outputs.append(host_mem)
#         cuda_outputs.append(cuda_mem)
# context = engine.create_execution_context()

# np.copyto(host_inputs[0], im.reshape(-1))


# cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
# context.execute_async(bindings=bindings, stream_handle=stream.handle)
# cuda.memcpy_dtoh_async(host_outputs[1], cuda_outputs[1], stream)
# cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
# stream.synchronize()
# print("execute times "+str(time.time()-start_time))

# output = host_outputs[0]
# height, width, channels = ori.shape
# for i in range(int(len(output)/model.layout)):
#     prefix = i*model.layout
#     index = int(output[prefix+0])
#     label = int(output[prefix+1])
#     conf  = output[prefix+2]
#     xmin  = int(output[prefix+3]*width)
#     ymin  = int(output[prefix+4]*height)
#     xmax  = int(output[prefix+5]*width)
#     ymax  = int(output[prefix+6]*height)

#     if conf > 0.7:
#         print("Detected {} with confidence {}".format(COCO_LABELS[label], "{0:.0%}".format(conf)))
#         cv2.rectangle(ori, (xmin,ymin), (xmax, ymax), (0,0,255),3)
#         cv2.putText(ori, COCO_LABELS[label],(xmin+10,ymin+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

# cv2.imwrite("result.jpg", ori)
# cv2.imshow("result", ori)
# cv2.waitKey(0)