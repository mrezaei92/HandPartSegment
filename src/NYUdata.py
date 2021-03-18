
"""
Copyright 2015 Markus Oberweger, ICG,
Graz University of Technology <oberweger@icg.tugraz.at>

This file is part of DeepPrior.

DeepPrior is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

DeepPrior is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with DeepPrior.  If not, see <http://www.gnu.org/licenses/>.
"""
import math
import cv2
import torch
from torch.utils.data.dataset import Dataset
# General
from PIL import Image
import numpy as np
import os.path
import scipy.io
import copy
from utils import *

#################
#### Functions ######################

## Transforms

def convert_uvd_to_xyz( uvd ):
    # both xyz and the resturned uvd will be np.array of size(num_joints,3)
    xRes = 640;
    yRes = 480;
    xzFactor = 1.08836710; #xzFactor=640/coeffX
    yzFactor = 0.817612648;

    normalizedX = np.double(uvd[:,0]) / xRes - 0.5;
    normalizedY = 0.5 - np.double(uvd[:,1]) / yRes;

    xyz = np.zeros(uvd.shape);
    xyz[:,2] = np.double(uvd[:,2]);
    xyz[:,0] = normalizedX * xyz[:,2] * xzFactor;
    xyz[:,1] = normalizedY * xyz[:,2] * yzFactor;
    return xyz

def point3DToImg_NYU(sample, fx=588.036865, fy=587.075073, ux=320, uy=240):
    ret = np.zeros((3,), np.float32)
    #convert to metric using f, see Thomson et.al.
    if sample[2] == 0.:
        ret[0] = ux
        ret[1] = uy
        return ret
    ret[0] = sample[0]/sample[2]*fx+ux
    ret[1] = uy-sample[1]/sample[2]*fy
    ret[2] = sample[2]
    return ret

def pointImgTo3D_NYU(sample, fx=588.036865, fy=587.075073, ux=320, uy=240):
    ret = np.zeros((3,), np.float32)
    # convert to metric using f, see Thomson et al.
    ret[0] = (sample[0] - ux) * sample[2] / fx
    ret[1] = (uy - sample[1]) * sample[2] / fy
    ret[2] = sample[2]
    return ret

def transformPoint2D(pt, M):
    """
    Transform point in 2D coordinates
    :param pt: point coordinates
    :param M: transformation matrix
    :return: transformed point
    """
    pt2 = np.asmatrix(M.reshape((3, 3))) * np.matrix([pt[0], pt[1], 1]).T
    return np.array([pt2[0] / pt2[2], pt2[1] / pt2[2]])

def rotateImageAndGt(imgDepth, gtUvd, gt3d, angle,jointIdRotCenter, pointsImgTo3DFunction,
                     fx=588.036865, fy=587.075073, cx=320, cy=240, bgValue=10000):
    """
    :param angle:   rotation angle
    :param pointsImgTo3DFunction:   function which transforms a set of points 
        from image coordinates to 3D coordinates
        like transformations.pointsImgTo3D() (from the same file).
        (To enable specific projections like for the NYU dataset)
    """
    # Rotate image around given joint
    jtId = jointIdRotCenter
    center = (gtUvd[jtId][0], gtUvd[jtId][1])
    rotationMat = cv2.getRotationMatrix2D(center, angle, 1.0)
    sizeRotImg = (imgDepth.shape[1], imgDepth.shape[0])
    imgRotated = cv2.warpAffine(src=imgDepth, M=rotationMat, 
                                dsize=sizeRotImg, flags=cv2.INTER_NEAREST, 
                                borderMode=cv2.BORDER_CONSTANT, 
                                borderValue=bgValue)
    
    # Rotate GT
    gtUvd_ = gtUvd.copy()
    gtUvdRotated = np.ones((gtUvd_.shape[0], 3), dtype=gtUvd.dtype)
    gtUvdRotated[:,0:2] = gtUvd_[:,0:2]
    gtUvRotated = np.dot(rotationMat, gtUvdRotated.T)
    gtUvdRotated[:,0:2] = gtUvRotated.T
    gtUvdRotated[:,2] = gtUvd_[:,2]
    # normalized joints in 3D coordinates
    gt3dRotated = convert_uvd_to_xyz(gtUvdRotated)
    
    return imgRotated, gtUvdRotated, gt3dRotated


def cropArea3D(imgDepth, com, fx=588.036865, fy=587.075073, minRatioInside=0.75, 
                   size=(250, 250, 250), dsize=(128, 128)):
        """
        Crop area of hand in 3D volumina, scales inverse to the distance of hand to camera
        :param com: center of mass, in image coordinates (x,y,z), z in mm
        :param size: (x,y,z) extent of the source crop volume in mm
        :param dsize: (x,y) extent of the destination size
        :return: cropped hand image, transformation matrix for joints, CoM in image coordinates
        """
        RESIZE_BILINEAR = 0
        RESIZE_CV2_NN = 1
        RESIZE_CV2_LINEAR = 2
        CROP_BG_VALUE = 0.0
        resizeMethod = RESIZE_CV2_NN
        # calculate boundaries
        zstart = com[2] - size[2] / 2.
        zend = com[2] + size[2] / 2.
        xstart = int(math.floor((com[0] * com[2] / fx - size[0] / 2.) / com[2]*fx))
        xend = int(math.floor((com[0] * com[2] / fx + size[0] / 2.) / com[2]*fx))
        ystart = int(math.floor((com[1] * com[2] / fy - size[1] / 2.) / com[2]*fy))
        yend = int(math.floor((com[1] * com[2] / fy + size[1] / 2.) / com[2]*fy))
        
        # Check if part within image is large enough; otherwise stop
        xstartin = max(xstart,0)
        xendin = min(xend, imgDepth.shape[1])
        ystartin = max(ystart,0)
        yendin = min(yend, imgDepth.shape[0])        
        ratioInside = float((xendin - xstartin) * (yendin - ystartin)) / float((xend - xstart) * (yend - ystart))
        if (ratioInside < minRatioInside) \
                and ((com[0] < 0) \
                    or (com[0] >= imgDepth.shape[1]) \
                    or (com[1] < 0) or (com[1] >= imgDepth.shape[0])):
            print("Hand largely outside image (ratio (inside) = {})".format(ratioInside))
            raise UserWarning('Hand not inside image')

        # crop patch from source
        cropped = imgDepth[max(ystart, 0):min(yend, imgDepth.shape[0]), 
                           max(xstart, 0):min(xend, imgDepth.shape[1])].copy()
        # add pixels that are out of the image in order to keep aspect ratio
        cropped = np.pad(cropped, ((abs(ystart)-max(ystart, 0), abs(yend)-min(yend, imgDepth.shape[0])), 
                                      (abs(xstart)-max(xstart, 0), abs(xend)-min(xend, imgDepth.shape[1]))), 
                            mode='constant', constant_values=int(CROP_BG_VALUE))
        msk1 = np.bitwise_and(cropped < zstart, cropped != 0)
        msk2 = np.bitwise_and(cropped > zend, cropped != 0)
        # Backface is at 0, it is set later; 
        # setting anything outside cube to same value now (was set to zstart earlier)
        cropped[msk1] = CROP_BG_VALUE
        cropped[msk2] = CROP_BG_VALUE
        
        wb = (xend - xstart)
        hb = (yend - ystart)
        trans = np.asmatrix(np.eye(3, dtype=float))
        trans[0, 2] = -xstart
        trans[1, 2] = -ystart
        # Compute size of image patch for isotropic scaling 
        # where the larger side is the side length of the fixed size image patch (preserving aspect ratio)
        if wb > hb:
            sz = (dsize[0], int(round(hb * dsize[0] / float(wb))))
        else:
            sz = (int(round(wb * dsize[1] / float(hb))), dsize[1])

        # Compute scale factor from cropped ROI in image to fixed size image patch; 
        # set up matrix with same scale in x and y (preserving aspect ratio)
        roi = cropped
        if roi.shape[0] > roi.shape[1]: # Note, roi.shape is (y,x) and sz is (x,y)
            scale = np.asmatrix(np.eye(3, dtype=float) * sz[1] / float(roi.shape[0]))
        else:
            scale = np.asmatrix(np.eye(3, dtype=float) * sz[0] / float(roi.shape[1]))
        scale[2, 2] = 1

        # depth resize
        if resizeMethod == RESIZE_CV2_NN:
            rz = cv2.resize(cropped, sz, interpolation=cv2.INTER_NEAREST)
        elif resizeMethod == RESIZE_BILINEAR:
            rz = HandDetector.bilinearResize(cropped, sz, CROP_BG_VALUE)
        elif resizeMethod == RESIZE_CV2_LINEAR:
            rz = cv2.resize(cropped, sz, interpolation=cv2.INTER_LINEAR)
        else:
            raise NotImplementedError("Unknown resize method!")

        # Sanity check
        numValidPixels = np.sum(rz != CROP_BG_VALUE)
        if (numValidPixels < 40) or (numValidPixels < (np.prod(dsize) * 0.01)):
            print("Too small number of foreground/hand pixels: {}/{} ({}))".format(
                numValidPixels, np.prod(dsize), dsize))
            raise UserWarning("No valid hand. Foreground region too small.")

        # Place the resized patch (with preserved aspect ratio) 
        # in the center of a fixed size patch (padded with default background values)
        ret = np.ones(dsize, np.float32) * CROP_BG_VALUE  # use background as filler
        xstart = int(math.floor(dsize[0] / 2 - rz.shape[1] / 2))
        xend = int(xstart + rz.shape[1])
        ystart = int(math.floor(dsize[1] / 2 - rz.shape[0] / 2))
        yend = int(ystart + rz.shape[0])
        ret[ystart:yend, xstart:xend] = rz
        off = np.asmatrix(np.eye(3, dtype=float))
        off[0, 2] = xstart
        off[1, 2] = ystart

        return ret, off * scale * trans, com #, off
    
#######################################
nyuRestrictedJointsEval = [0, 3, 6, 9, 12, 15, 18, 21, 24, 25, 27, 30, 31, 32]
class NyuHandPoseDataset(Dataset):
    def __init__(self, basepath="/media/mrezaei/Samsung_T3/NYU",
                 train=True, 
                 cropSize=(128, 128),doJitterRotation=False,doAddWhiteNoise=False,rotationAngleRange=[-45.0, 45.0],
                 sigmaNoise=10,transform=None, targetTransform=None, 
                 randomSeed = 123456789,cropSize3D=(250,250,250),camID=1,do_norm_zero_one=False): 
      
        #  do_norm_zero_one if is False, the depth values will be squashed to the interval [-1,+1]      
        self.min_depth_cam = 50.
        self.max_depth_cam = 1500.
    
        self.do_norm_zero_one=do_norm_zero_one
        self.doJitterRotation=doJitterRotation
        self.rotationAngleRange=rotationAngleRange
        self.basepath = basepath
        self.restrictedJointsEval = nyuRestrictedJointsEval
        self.cropSize3D = cropSize3D
        # For comparisons check results with adapted cube size
        
        self.testseq2_start_id = 2441
        self.cropSize = cropSize
        self.doAddWhiteNoise = doAddWhiteNoise
        self.sigmaNoise = sigmaNoise
        self.camID=camID;
        self.doNormZeroOne = do_norm_zero_one  # [-1,1] or [0,1]
        
        self.transform = transform
        self.targetTransform = targetTransform
        
        self.doTrain = train
        self.seqName = ""
        if self.doTrain:
            self.seqName = "train"
        else:
            self.seqName = "test"
            
        self.numJoints = len(nyuRestrictedJointsEval)
        
        # Load labels
        labels = '{}/{}/joint_data.mat'.format(basepath, self.seqName)
        self.labelMat = scipy.io.loadmat(labels)
        
        # Get number of samples from annotations (test: 8252; train: 72757)
        numAllSamples = self.labelMat['joint_xyz'][self.camID-1].shape[0]
        self.numSamples = numAllSamples
                
        # Precomputations for normalization of 3D point
        self.precompute_normalization_factors()
        
        print("NYU Dataset init done.")
        
        
    def __len__(self):
        return self.numSamples
    
    def precompute_normalization_factors(self):
        min_depth_in = self.min_depth_cam
        max_depth_in = self.max_depth_cam
        
        if self.do_norm_zero_one:
            depth_range_out = 1.
            self.norm_min_out = 0.
        else:
            depth_range_out = 2.
            self.norm_min_out = -1.
            
        depth_range_in = float(max_depth_in - min_depth_in)
        
        self.norm_max_out = 1.
        self.norm_min_in = min_depth_in
        self.norm_scale_3Dpt_2_norm = depth_range_out / depth_range_in
        
        
    def __getitem__(self, index):
 
        data = loadSingleSampleNyu(basepath=self.basepath, seqName=self.seqName, index=index,doLoadRealSample=True,
                                   camId=self.camID,cropSize3D=self.cropSize3D, cropSize=self.cropSize,
                                   doAddWhiteNoise=self.doAddWhiteNoise,sigmaNoise=self.sigmaNoise,
                                   doJitterRotation=self.doJitterRotation,
                                   rotationAngleRange=self.rotationAngleRange,labelMat=self.labelMat, minRatioInside=0.6)
                                    
        if self.doNormZeroOne:
            img, target = normalizeZeroOne(data)
        else:
            img, target = normalizeMinusOneOne(data)
            
        # Image need to be HxWxC and will be divided by transform (ToTensor()), which is assumed here!
        uvd=(data["gt2Dcrop"]);uvd[:,2]=uvd[:,2]-uvd[-1,2]
        #landmarks=uvd[:,0:2]; #landmarks will be of shape (num_joints,2)
      
        img = np.expand_dims(img, axis=0)
        img=torch.from_numpy(img)

        #heatmap_2D=torch.from_numpy(Y).double()

        #target = torch.from_numpy(target.astype('float32'))
        #M=data["M"]
        #com = torch.from_numpy(data["com3D"])
        landmarks=torch.from_numpy(uvd)
        
        return img, landmarks
        
#### ######### Helper functions #########################################################33
def loadDepthMap(filename):
    """
    Read a depth-map
    :param filename: file name to load
    :return: image data of depth image
    """
    with open(filename) as f:
        img = Image.open(filename)
        # top 8 bits of depth are packed into green channel and lower 8 bits into blue
        assert len(img.getbands()) == 3
        r, g, b = img.split()
        r = np.asarray(r,np.int32)
        g = np.asarray(g,np.int32)
        b = np.asarray(b,np.int32)
        dpt = np.bitwise_or(np.left_shift(g,8),b)
        imgdata = np.asarray(dpt,np.float32)

    return imgdata

def loadSingleSampleNyu(basepath, seqName, index, doLoadRealSample=True,camId=1, cropSize3D=(250,250,250), 
                        cropSize=(128,128),doAddWhiteNoise=False,sigmaNoise=10.,
                       doJitterRotation=False,rotationAngleRange=[-45.0, 45.0],
                        labelMat=None, minRatioInside=0.3):
    
    
    idComGT = 13
    # Load the dataset
    objdir = '{}/{}/'.format(basepath,seqName)

    if labelMat == None:
        labelsAdress = '{}/{}/joint_data.mat'.format(basepath, seqName)
        labelMat = scipy.io.loadmat(labelsAdress)
        
    joints3D = labelMat['joint_xyz'][camId-1]
    joints2D = labelMat['joint_uvd'][camId-1]
   
    eval_idxs = nyuRestrictedJointsEval

    numJoints = len(eval_idxs)
    
    data = []
    line = index
    
    
    # Assemble original filename
    prefix = "depth" if doLoadRealSample else "synthdepth"
    dptFileName = '{0:s}/{1:s}_{2:1d}_{3:07d}.png'.format(objdir, prefix, camId, line+1)
    
    dpt = loadDepthMap(dptFileName)
    
    # Add noise?
    if doAddWhiteNoise:
        img_white_noise_scale = np.random.randn(dpt.shape[0], dpt.shape[1])
        dpt = dpt + sigmaNoise * img_white_noise_scale
    
    # joints in image coordinates
    gt2Dorignal = np.zeros((numJoints, 3), np.float32)
    jt = 0
    for ii in range(joints2D.shape[1]):
        if ii not in eval_idxs:
            continue
        gt2Dorignal[jt,0] = joints2D[line,ii,0]
        gt2Dorignal[jt,1] = joints2D[line,ii,1]
        gt2Dorignal[jt,2] = joints2D[line,ii,2]
        jt += 1

    # normalized joints in 3D coordinates
    gt3Dorignal = np.zeros((numJoints,3),np.float32)
    jt = 0
    for jj in range(joints3D.shape[1]):
        if jj not in eval_idxs:
            continue
        gt3Dorignal[jt,0] = joints3D[line,jj,0]
        gt3Dorignal[jt,1] = joints3D[line,jj,1]
        gt3Dorignal[jt,2] = joints3D[line,jj,2]
        jt += 1
        
    if doJitterRotation:
        rotation_angle_scale = np.random.randn()
        rot = rotation_angle_scale * (rotationAngleRange[1] - rotationAngleRange[0]) + rotationAngleRange[0]
        dpt, gt2Dorignal, gt3Dorignal = rotateImageAndGt(dpt, gt2Dorignal, gt3Dorignal, rot, 
                                                 jointIdRotCenter=idComGT, 
                                                 pointsImgTo3DFunction=convert_uvd_to_xyz,
                                                 bgValue=10000)

        
    comGT = copy.deepcopy(gt2Dorignal[idComGT])  # use GT position for comparison
        
    
    # Jitter scale (cube size)?
    cubesize = cropSize3D
   
    
    dpt, M, com = cropArea3D(imgDepth=dpt,com=comGT,minRatioInside=minRatioInside,size=cubesize, dsize=cropSize)
                                    
    com3D = pointImgTo3D_NYU(com)
    gt3Dcrop = gt3Dorignal - com3D     # normalize to com
    gt2Dcrop = np.zeros((gt2Dorignal.shape[0], 3), np.float32)
    for joint in range(gt2Dorignal.shape[0]):
        t=transformPoint2D(gt2Dorignal[joint], M)
        gt2Dcrop[joint, 0] = t[0]
        gt2Dcrop[joint, 1] = t[1]
        gt2Dcrop[joint, 2] = gt2Dorignal[joint, 2]
 
    D={};D["M"]=M;D["com3D"]=com3D;D["cubesize"]=cubesize
    D["dpt"]=dpt.astype(np.float32);D["gt2Dorignal"]=gt2Dorignal
    D["gt2Dcrop"]=gt2Dcrop;D["gt3Dorignal"]=gt3Dorignal;D["gt3Dcrop"]=gt3Dcrop;
    return D


def normalizeZeroOne(sample):
    imgD = np.asarray(sample["dpt"].copy(), 'float32')
    imgD[imgD == 0] = sample.com[2] + (sample['cubesize'][2] / 2.)
    imgD -= (sample["com3D"][2] - (sample['cubesize'][2] / 2.))
    imgD /= sample['cubesize'][2]
    
    target = np.clip(np.asarray(sample["gt3Dcrop"], dtype='float32') / sample['cubesize'][2], -0.5, 0.5) + 0.5
                
    return imgD, target
    
    
def normalizeMinusOneOne(sample):
    imgD = np.asarray(sample["dpt"].copy(), 'float32')
    imgD[imgD == 0] = sample["com3D"][2] + (sample['cubesize'][2] / 2.)
    imgD -= sample["com3D"][2]
    imgD /= (sample['cubesize'][2] / 2.)
    
    target = np.clip(np.asarray(sample["gt3Dcrop"], dtype='float32')/ (sample['cubesize'][2] / 2.), -1, 1)
    return imgD, target

def denormalizeJointPositions(jointPos,cubesize=250,deNormZeroOne=False,com3D=None):
    # jointPos is an array of shape(batch,num_key,3)
    offset = 0
    com3D=np.array(com3D)
    scaleFactor = cubesize / 2.0
    if deNormZeroOne:
        offset = -0.5
        scaleFactor = cubesize
        
    r=((jointPos + offset) * scaleFactor)
    if com3D is not None:
        r=r+np.expand_dims(com3D,axis=1)
    return r

































