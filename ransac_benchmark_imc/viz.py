import numpy as np
import matplotlib.pyplot as plt
import cv2

def decolorize(img):
    return  cv2.cvtColor(cv2.cvtColor(img,cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c,ch = img1.shape
    img1o = deepcopy(img1)
    img2o = deepcopy(img2)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1o = cv2.line(img1o, (x0,y0), (x1,y1), color,1)
        img1o = cv2.circle(img1o,tuple(pt1.squeeze().astype(np.int32)),5,color,-1)
        img2o = cv2.circle(img2o,tuple(pt2.squeeze().astype(np.int32)),5,color,-1)
    return img1o,img2o


def draw_everything(img1, img2, pts1_good, pts2_good, F_gt):
    lines1gt = cv2.computeCorrespondEpilines(pts2_good.reshape(-1,1,2), 2, F_gt)
    lines1gt = lines1gt.reshape(-1,3)
    img5gt,img6gt = drawlines(img1,img2,lines1gt,pts1_good,pts2_good)
    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2gt = cv2.computeCorrespondEpilines(pts1_good.reshape(-1,1,2), 1,F_gt)
    lines2gt = lines2gt.reshape(-1,3)
    img3gt,img4gt = drawlines(img2,img1,lines2gt,pts2_good,pts1_good)
    plt.figure(figsize = (12,12))
    plt.subplot(121)
    plt.imshow(img5gt)
    plt.subplot(122)
    plt.imshow(img3gt)
    return
