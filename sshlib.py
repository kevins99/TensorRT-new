import time
from typing import List

import torch
import numpy as np

from model.SSH import SSH
from model.network import load_check_point
from model.utils.config import cfg
from model.nms.nms_wrapper import nms
from model.utils.test_utils import _get_image_blob, _compute_scaling_factor


def Timer(callback):
    """
    Utility function to time different parts of the inference process
    Use it as a decorator on all functions that are to be timed
    Arguments:
        callback {function} -- the function being decorated, i.e., the function that is to be timed

    Returns:
        wrapper function which adds the functionality of timing to original callback
    """
    def wrapper(*args, **kwargs):
        start = time.time()
        result = callback(*args, **kwargs)
        end = time.time()
        print(f"{callback.__name__}: {end - start}")
        return result

    return wrapper


class FaceDetector:
    """
    This is the main class to be instantiated to use the SSH FaceDetector 
    to detect faces in a given frame 

    Attributes:
        thresh = threshhold value for IOU

    Methods: 
        - load_weights
        - preprocessImage
        - predict
        - non_max_suppression
    Returns:
        An object of to be used for FaceDetection
    """

    def __init__(self, thresh=0.5):
        """
        constructor for FaceDector class

        Keyword Arguments:
            thresh {float} -- threshhold value for IOU (default: {0.5})
        """
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.thresh = thresh

        self.net: torch.nn.Module = SSH(vgg16_image_net=False)
        self.net.eval()
        self.net.cuda()

    # @Timer
    def load_weights(self, path="./check_point.zip"):
        """
        Load weights of pretrained model.

        Keyword Arguments:
            path {str} -- path to saved model (default: {"./check_point/check_point.zip"})
        """
        check_point = load_check_point(path)
        self.net.load_state_dict(check_point['model_state_dict'])

    # @Timer
    def preprocessImage(self, im):
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

        im_info = torch.from_numpy(im_info).to(self.device)
        im_data = torch.from_numpy(im_data).to(self.device)
        return im_info, im_data, im_scale

    # @Timer
    def predict(self, im_data, im_scale):
        """
        Runs the pretrained network in eval mode and precits bounding boxes for given frame/image

        Arguments:
            im_info [torch.Tensor] -- array containing shapes of image_blobs
            im_data [torch.Tensor] -- the actual image data extracted from the image_blob
            im_scale [int] -- The scale factor
        Returns:
            list of ROIs output by network, that have been rescaled to align with the dimensions of original image
        """
        with torch.no_grad():
            # print(im_info[0,2])
            ssh_rois = self.net(im_data)
            indices = (ssh_rois[:, :, 4] > self.thresh)
            ssh_roi_keep = ssh_rois[:, indices[0], :]
            ssh_roi_keep[:, :, 0:4] /= im_scale
           # print(ssh_roi_keep)
        return ssh_roi_keep

    # @Timer
    def non_max_suppression(self, ssh_rois: List[torch.Tensor]):
        """
        perform NMS on ROIs given by SSH network 

        Arguments:
            ssh_rois {List[torch.Tensor]} -- list ROIs given by SSH network     
        Returns:
            bounding_boxes {[numpy.ndarray]} -- final list of bounding boxes for all detected faces
        """

        # NOTE :- The ROI operations are currently being perfomred on CPU, instead of cuda Tensors.
        # I've tried moving them to gpu but it doesn't work, atleast on my machine, despite there being a gpu version
        # of nms_code (./model/nms/gpu_nms.pyx)
        # NMS part of the code is barely taking any time as is, so i've left it this way for now

        bounding_boxes = []
        for single_roi in ssh_rois:
            single_roi = single_roi.cpu().numpy()
            nms_keep = nms(single_roi, cfg.TEST.RPN_NMS_THRESH)
            cls_single = single_roi[nms_keep, :]
            bounding_boxes.append(cls_single)
        return bounding_boxes
