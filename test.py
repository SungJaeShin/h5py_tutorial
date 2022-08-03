#!/usr/bin/env python
from asyncio import SubprocessTransport
import rospy
import numpy as np
import cv2 as cv
import h5py
import os
import torch
from PIL import Image as pil_Image
import faiss

# NetVLAD Related
from utils.arguments import argument_parser
from utils.tools import import_yaml
from utils.transforms import *
from models.models import NetVLAD

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
DB_METHOD = 1 # 1: save several dataset / 2: save one dataset (resize and fix subPano image: [300, 640, 3])
h5_filename = "/home/sj/workspace/paper_ws/icra2023/src/ROS_NetVLAD_Panorama/images/subpanoDB.h5"
# h5 = h5py.File(h5_filename, 'w')
h5 = h5py.File(h5_filename, 'a') # It can read and write at same time

VLAD_FIRST = True
h5VLAD_filename = "/home/sj/workspace/paper_ws/icra2023/src/ROS_NetVLAD_Panorama/images/subpanoVLAD.h5"
h5VLAD = h5py.File(h5VLAD_filename, 'a') # It can read and write at same time

class PanoNetVLAD():
    def __init__(self, config, model, device, transforms=None):
        # Model
        self.device = device
        self.config = config
        self.model = model.to(self.device)
        self.transforms = transforms

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
        
        out_height = str(out.shape[0])
        print("\033[1;32m Subpano Height: ", out_height, "\033[0m");

        # For Saving H5 file
        if DB_METHOD == 2:
            out = cv.resize(out, dsize=(out.shape[1], 300), interpolation=cv.INTER_CUBIC) # [width, height, channel]
            h5_out_height = str(out.shape[0])
            print("\033[1;32m For H5 Subpano Height: ", h5_out_height, "\033[0m \n");

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
        if DB_METHOD == 1:
            idx = "/subpano/" + str(sequence)
            h5.create_dataset(idx, data=subPano)
        ######################################################################

        ##### [CASE 2] Make Only One Dataset that get all subPano Images #####
        if DB_METHOD == 2:
            global FIRST
            if FIRST == True:
                # pdb.set_trace()
                h5.create_dataset('subpano', data=subPano, maxshape=(None, None, None, None), chunks=True)
                FIRST = False
                return

            len1 = h5['subpano'].shape[0]
            len2 = len(subPano)
            total_len = np.array([len1+len2, 300, 640, 3])
            h5['subpano'].resize(total_len)
            h5['subpano'][len1:] = subPano
        ######################################################################

    def saveSubVLADH5(self, subVLAD, panoSeq):
        subVLAD = subVLAD.detach().cpu().numpy()

        global VLAD_FIRST
        if VLAD_FIRST == True:
            # pdb.set_trace()
            h5VLAD.create_dataset('subpano', data=subVLAD, maxshape=(None, None), chunks=True)
            VLAD_FIRST = False
            return

        len1 = h5VLAD['subpano'].shape[0]
        len2 = len(subVLAD)
        total_len = np.array([len1+len2, 32768])
        h5VLAD['subpano'].resize(total_len)
        h5VLAD['subpano'][len1:] = subVLAD

    def readSubPanoDB(self):
        ##### [CASE 1] Make Several Dataset which get 'N' subPano Images #####
        if DB_METHOD == 1:
            cur_shape = len(h5['subpano'])
            print("\033[1;33m [CASE 1] Current length DB: ", cur_shape, "\033[0m")

        ##### [CASE 2] Make Only One Dataset that get all subPano Images #####
        if DB_METHOD == 2:
            cur_shape = h5['subpano'].shape
            print("\033[1;33m [CASE 2] Current Shape DB: ", cur_shape, "\033[0m")

        curVLAD_shape = h5VLAD['subpano'].shape
        print("\033[1;33m [VLAD Case] Current Shape DB: ", curVLAD_shape, "\033[0m")

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
                if panoHeight < 10 or panoWidth < 10:
                    print("Not Pano Images !!")
                    buffer_mutex.release()
                    continue
   
                print("pano Height: %d, pano Widht: %d", panoHeight, panoWidth)

                # Initialization of First Panorama Information 
                panoHeader = panoFirst.header
                panoTime = panoFirst.header.stamp.to_sec()
                panoSeq = panoFirst.header.seq
                panoMat = self.convertSensor2Mat(panoFirst)
                
                print("\033[1;31m Current Sequence: ", panoSeq, "\033[0m")

                # Prune Black Space of Panorama Image
                panoMat = self.prune_blank(panoMat)

                out_panoMat = panoMat.shape[0]
                print("\033[1;32m out_panoMat Height: ", out_panoMat, "\033[0m");
                if out_panoMat < 10:
                    print("Very Small size after pruning because of black background")
                    buffer_mutex.release()
                    continue
     
                # Make Sliding Image window of Panorama 
                # Number of Sliding Image is 10 (Default "n = 10")
                subPanoMat = self.makeSubPano(panoMat, 10)

                # Convert subPanoMat to Pil
                subPanoPil = [pil_Image.fromarray(np.uint8(f)) for f in subPanoMat]
                if self.transforms is not None:
                    subPanoPil = [self.transforms(f) for f in subPanoPil]

                # Make VLAD Vector using NetVLAD Model
                subVLAD = self.model(torch.stack(subPanoPil).to(device))

                # Save SubPanorama Image
                if USE_H5 == True:
                    self.saveSubPanoH5(subPanoMat, panoSeq)
                else:
                    self.saveSubPano(subPanoMat, panoSeq)

                # Save Subpanorama VLAD
                self.saveSubVLADH5(subVLAD, panoSeq)

                del panoFirst, panoHeight, panoWidth, panoHeader, panoTime, panoSeq, panoMat, subPanoPil, subVLAD

                self.readSubPanoDB()
    
            buffer_mutex.release()

if __name__ == '__main__':
    rospy.init_node('NetVLAD_Realsense')

    device = 'cuda'
    if device == 'cuda':
        print("Using CUDA!")

    # Get configuration file of NetVLAD
    opt = argument_parser()
    config = import_yaml(opt.config)

    # # Set visible device
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"]= str(config['hardware']['gpu_number'])

    # Time Duration
    start = time.time() 

    # Loading model
    model = NetVLAD(config)
    model.load_state_dict(torch.load(opt.ptmodel)['state_dict'])
    print("Pretained model loaded from: ", opt.ptmodel)

    print("Time duration of getting Pretrained model: ", (time.time()-start))

    PanoNetVLAD(config, model, device, transforms=T_KAIST)

    rospy.spin()
