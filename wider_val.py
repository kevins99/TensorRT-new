# %% 
import os
import cv2
from sshlib import FaceDetector
from tqdm import tqdm 

facedetector = FaceDetector()
facedetector.load_weights()
# %%
os.chdir('/home/dlbox/Desktop/obj_detection/varun/pytorch/SSH-pytorch-master/kushal/WIDER_val/images')
for dirs in tqdm(os.listdir()):
    for img_file in os.listdir(os.path.join(os.getcwd(),dirs)):
        # print(os.path.join(os.getcwd(),dirs,img_file))
        img = cv2.imread(os.path.join(os.getcwd(),dirs,img_file))
        im_info, im_data, im_scale = facedetector.preprocessImage(img)
        ssh_rois = facedetector.predict(im_info, im_data, im_scale)
        bounding_boxes = facedetector.non_max_suppression(ssh_rois)
        # print(len(bounding_boxes[0]))
        # for bb in bounding_boxes[0]:
        #     cv2.rectangle(img,(bb[0],bb[1]),(bb[2],bb[3]),(0,0,255),1,cv2.LINE_AA)
        # cv2.imshow('test',img)
        # cv2.waitKey(0)
        detection_file_name = '/home/dlbox/Desktop/obj_detection/varun/pytorch/SSH-pytorch-master/kushal/detections2/'+ img_file[:-3] + "txt"
        with open(detection_file_name, 'w+') as det_file:
            for bb in bounding_boxes[0]:
                bb = list(bb)
                bb.insert(0,bb.pop())
                result = f'person {bb[0]:.2f} {bb[1]:.0f} {bb[2]:.0f} {bb[3]-bb[1]:.0f} {bb[4]-bb[2]:.0f}'
                det_file.writelines(result + '\n')

# %%
# %%
