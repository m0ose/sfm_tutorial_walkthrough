import skimage
import matplotlib
from kornia.feature import LoFTR
import kornia.feature as KF
import torch as torch
import cv2
import numpy as np


def _matchFeatures(img1, img2, matcher):

    # convert to grayscale if the input image is a color image
    if len(img1.shape) > 2:
        # img1 = skimage.color.rgb2gray(img1)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        print('img1 to grayscale')
    if len(img2.shape) > 2:
        # img2 = skimage.color.rgb2gray(img2)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        print('img2 to grayscale')
    # resize images while maintaining aspect ratio
    newDim= (640,640)
    img1Shape = img1.shape
    img2Shape = img2.shape

    img1_raw = cv2.resize(img1, newDim)
    img2_raw = cv2.resize(img2, newDim)

    print(img1_raw.shape, img1.shape)

    img1_tensor = torch.from_numpy(img1_raw)[None][None].cpu() / 255.
    img2_tensor = torch.from_numpy(img2_raw)[None][None].cpu() / 255.
    batch = {'image0': img1_tensor, 'image1': img2_tensor}

    # Inference with LoFTR and get prediction
    with torch.no_grad():
        correspondences = matcher(batch)
        print(correspondences.keys())
        mkpts0 = correspondences['keypoints0'].cpu().numpy()
        mkpts1 = correspondences['keypoints1'].cpu().numpy()
        mconf = correspondences['confidence'].cpu().numpy()
        S1 = np.eye(2)
        S1[0,0] = img1Shape[1] / newDim[0]
        S1[1,1] = img1Shape[0] / newDim[1]
        S2 = np.eye(2)
        S2[0,0] = img2Shape[1] / newDim[0]
        S2[1,1] = img2Shape[0] / newDim[1]
        mkpts0 = S1 @ mkpts0.T
        mkpts1 = S2 @ mkpts1.T

    return mkpts0.T, mkpts1.T, mconf

def matchFeaturesLOFTR(img1, img2, pretrainedName='outdoor'):
    matcher = LoFTR(pretrained=pretrainedName)
    matcher = matcher.eval().cpu()
    return _matchFeatures(img1, img2, matcher)

def matchFeaturesHardnet(img1, img2):
    matcher = KF.LocalFeatureMatcher(KF.GFTTAffNetHardNet(3000), KF.DescriptorMatcher('smnn', 0.9))
    matcher = matcher.eval().cpu()
    return _matchFeatures(img1, img2, matcher)

def drawMatches(img1, img2, keypoints1, keypoints2, confidences):
    colorMap = matplotlib.cm.get_cmap('Spectral')
    newImage = np.hstack((img1, img2))
    if len(newImage.shape) == 2:
        newImage = skimage.color.gray2rgb(newImage)
    for i in range(len(keypoints1)):
        k1 = keypoints1[i]
        k2a = keypoints2[i]
        k2b = (k2a[0] + img1.shape[1], k2a[1])
        color = colorMap(confidences[i])
        colorb = (color[0]*255, color[1]*255, color[2]*255)
        cv2.line(newImage, (int(k1[0]), int(k1[1])), (int(k2b[0]), int(k2b[1])), colorb, 1)
        cv2.circle(newImage, (int(k1[0]), int(k1[1])), 5, colorb, 1)
        cv2.circle(newImage, (int(k2b[0]), int(k2b[1])), 5, colorb, 1)
    return newImage

def drawMatchesVertical(img1, img2, keypoints1, keypoints2, confidences):
    colorMap = matplotlib.cm.get_cmap('Spectral')
    newImage = np.vstack((img1, img2))
    if len(newImage.shape) == 2:
        newImage = skimage.color.gray2rgb(newImage)
    for i in range(len(keypoints1)):
        k1 = keypoints1[i]
        k2a = keypoints2[i]
        k2b = (k2a[0], k2a[1] + img1.shape[0])
        color = colorMap(confidences[i])
        colorb = (color[0]*255, color[1]*255, color[2]*255)
        cv2.line(newImage, (int(k1[0]), int(k1[1])), (int(k2b[0]), int(k2b[1])), colorb, 1)
        cv2.circle(newImage, (int(k1[0]), int(k1[1])), 5, colorb, 1)
        cv2.circle(newImage, (int(k2b[0]), int(k2b[1])), 5, colorb, 1)
    return newImage
