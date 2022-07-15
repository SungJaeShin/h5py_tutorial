#!/usr/bin/env python
from asyncio import SubprocessTransport
import rospy
import numpy as np
import cv2 as cv
import h5py
import os
from PIL import Image as pil_Image

# For Debugging
import pdb

# Image Stitching related 
import imutils
from imutils import paths

# Multi-thread and Mutex related
import threading
import time
import random

# Getting lots of information related with sensor_msgs
from queue import Queue
import copy

# ROS Messeage Related
from sensor_msgs.msg import Image
from std_msgs.msg import Header

from cv_bridge import CvBridge, CvBridgeError

buffer_mutex = threading.Lock()

# True -> Save as h5 extension !!
USE_H5 = True 
FIRST = True
h5_filename = "/home/sj/workspace/paper_ws/icra2023/src/ROS_NetVLAD_Panorama/images/subpanoDB.h5"
# h5 = h5py.File(h5_filename, 'w')
h5 = h5py.File(h5_filename, 'a') # It can read and write at same time

class PanoNetVLAD():
    def __init__(self):
        # Sensor_msgs Queue
        self.panoBuf = Queue()

        # Convert CV
        self.bridge = CvBridge()

        # Subscriber
        self.sub_pano = rospy.Subscriber("/Panorama_Image", Image, self.pano_callback)

        # Multi-thread 
        self.thread = threading.Thread(target=self.makePanoDataset, daemon=True)
        self.thread.start()

    # Convert sensor_msgs to cv::Mat image         
    def convertSensor2Mat(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        
        return cv_image    

    def pano_callback(self, data):
        global buffer_mutex
        buffer_mutex.acquire()
        self.panoBuf.put(data)
        buffer_mutex.release()

    def prune_blank(self, img, thresh=0.05):
        (H, W, C) = img.shape
        up_cut = 0
        down_cut = H

        # Prune top
        for h_top in range(H//2):
            h_bot = H - h_top - 1
            row_top = img[h_top, :, :]
            row_bot = img[h_bot, :, :]
            black_pixel_count_top = np.sum(np.sum(row_top != [[0, 0, 0]], axis=1) == 0)
            black_pixel_count_bot = np.sum(np.sum(row_bot != [[0, 0, 0]], axis=1) == 0)

            if black_pixel_count_top / W > thresh:
                up_cut = h_top

            if black_pixel_count_bot / W > thresh:
                down_cut = h_bot

        out = img[up_cut:down_cut, :, :]

        return out

    def makeSubPano(self, img, windows=10):
        # Image = (height, width, channel)
        panoWidth = img.shape[1]
        stride = int((panoWidth - 640) / (windows - 1))

        subPano = []
        for n in range(windows):
            img_subPano = img[:, stride * n: stride * n + 640, :]
            subPano.append(img_subPano)

        return subPano

    def saveSubPano(self, subPano, sequence):
        # Convert back from numpy to PIL 
        subPano = [pil_Image.fromarray(np.uint8(f)) for f in subPano]
        for i, f in enumerate(subPano):
            f.save("./images/subpano/subpano_" + str(sequence) + "_" + str(i) + ".png")

    def saveSubPanoH5(self, subPano, sequence):
        ##### [CASE 1] Make Several Dataset which get 'N' subPano Images #####
        # idx = "/subpano/" + str(sequence)
        # h5.create_dataset(idx, data=subPano)
        ######################################################################

        ##### [CASE 2] Make Only One Dataset that get all subPano Images #####
        global FIRST
        if FIRST == True:
            # pdb.set_trace()
            h5.create_dataset('subpano', data=subPano, maxshape=(None, None, None, None), chunks=True)
            FIRST = False
            return

        print("First is False: ", FIRST)
        pdb.set_trace()
        
        
        ########## Saving cached feature indexes for debugging ##########
        # h5_filename = "/media/TrainDataset/lfincache.h5"
        # hf = h5py.File(h5_filename, 'w')
        # hf.create_dataset('qFeat', data=qFeat)
        # hf.create_dataset("dbFeat", data=dbFeat)
        # hf.close()
        # pdb.set_trace()
        # hf = h5py.File(h5_filename, 'r')
        # qFeat = np.array(hf.get('qFeat'))
        # dbFeat = np.array(hf.get('dbFeat'))
        #################################################################

    def makePanoDataset(self):
        # print("Active Thread: ", threading.active_count())
        # print("Thread Count: ", threading.current_thread())

        while True:
            buffer_mutex.acquire()

            if self.panoBuf.qsize() != 0: 
                # Get First Panorama Image from panoBuf
                # print("panoBuf Size: ", self.panoBuf.qsize())
                panoFirst = self.panoBuf.get()

                # Get First Panorama Image Size 
                panoHeight = panoFirst.height
                panoWidth = panoFirst.width
                
                # Filter Real Pano Images
                if panoHeight == 0 or panoWidth == 0:
                    print("Not Pano Images !!")
                    buffer_mutex.release()
                    continue

                print("pano Height: %d, pano Widht: %d", panoHeight, panoWidth)

                # Initialization of First Panorama Information 
                panoHeader = panoFirst.header
                panoTime = panoFirst.header.stamp.to_sec()
                panoSeq = panoFirst.header.seq
                panoMat = self.convertSensor2Mat(panoFirst)
                
                panoPath = "/home/sj/workspace/paper_ws/icra2023/src/ROS_NetVLAD_Panorama/images/panorama/pano_" + str(panoSeq) + ".png"
                cv.imwrite(panoPath, panoMat)

                # Prune Black Space of Panorama Image
                panoMat = self.prune_blank(panoMat)
                prunePanoPath = "/home/sj/workspace/paper_ws/icra2023/src/ROS_NetVLAD_Panorama/images/prune_pano/prune_pano_" + str(panoSeq) + ".png"
                cv.imwrite(prunePanoPath, panoMat)

                # Make Sliding Image window of Panorama 
                # Number of Sliding Image is 10 (Default "n = 10")
                subPanoMat = self.makeSubPano(panoMat, 10)

                # Save SubPanorama Image
                if USE_H5 == True:
                    self.saveSubPanoH5(subPanoMat, panoSeq)
                else:
                    self.saveSubPano(subPanoMat, panoSeq)

                del panoFirst, panoHeight, panoWidth, panoHeader, panoTime, panoSeq, panoMat
    
            buffer_mutex.release()

if __name__ == '__main__':
    rospy.init_node('NetVLAD_Realsense')

    PanoNetVLAD()

    rospy.spin()
