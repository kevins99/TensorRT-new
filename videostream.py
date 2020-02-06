import time
import argparse
import cv2
import torch
from sshlib import FaceDetector
import numpy as np


def parser():
    parser = argparse.ArgumentParser('SSH Train module')
    parser.add_argument('--model_path', dest='model_path', default='check_point.zip', type=str,
                        help='Saved model path')
    parser.add_argument('--thresh', dest='thresh', default=0.5, type=float,
                        help='Detections with a probability less than this threshold are ignored')
    return parser.parse_args()


avg = 0
c = 0
if __name__ == "__main__":
    args = parser()
    capture = cv2.VideoCapture(0)
    face_detector = FaceDetector()
    face_detector.load_weights(path=args.model_path)
    while True:
        torch.cuda.empty_cache()
        # For some reason the GPU is caching the final result of every frame after each pass
        # hence its necesarry to empty the cache before running a pass through the network

        ret, frame = capture.read()
        start = time.time()
        im_info, im_data, im_scale = face_detector.preprocessImage(frame)
        print((im_info), type(im_data), type(im_scale))
        ssh_rois = face_detector.predict(im_data, 1)
        print(ssh_rois)
        bounding_boxes = face_detector.non_max_suppression(ssh_rois)
        for bounding_box in bounding_boxes[0]:
            cv2.rectangle(frame, (bounding_box[0], bounding_box[1]), (bounding_box[2], bounding_box[3]), color=(
                255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        end = time.time()
        fps = 1/(end-start)
        # cv2.putText(frame, fps, (0, 130), cv2.FONT_HERSHEY_PLAIN,
        #             1, (200, 255, 255), 2, cv2.LINE_AA)
        avg += fps

        print(f"FPS:{avg/(c+1)}")
        c += 1
        cv2.imshow("Detected faces", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
