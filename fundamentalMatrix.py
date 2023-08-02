import numpy as np
import cv2

def normalize_points(points):
    # Normalize the points to have zero mean and average distance of sqrt(2)
    mean = np.mean(points, axis=0)
    points_normalized = points - mean
    scale = np.sqrt(2) / np.mean(np.linalg.norm(points_normalized, axis=1))
    T = np.array([[scale, 0, -scale * mean[0]],
                  [0, scale, -scale * mean[1]],
                  [0, 0, 1]])
    points_homogeneous = np.column_stack((points, np.ones(len(points))))
    points_normalized = (T @ points_homogeneous.T).T
    return points_normalized, T


#
# estimate 8-point fundamental matrix. One Time. 
# points1 and points2 are 2D points in homogeneous coordinates
# returns F, a 3x3 fundamental matrix
#   Made with significant help from chatGPT
#
def estimate_normalized_8_point_fundamental_matrix(points1, points2):
    if len(points1) != len(points2) or len(points1) < 8:
        raise ValueError("Number of points must be at least 8 and must match in both sets")

    # Normalize the points
    points1_norm, T1 = normalize_points(points1)
    points2_norm, T2 = normalize_points(points2)
    #
    A = []
    for i in range(len(points1_norm)):
        x1, y1 = points1_norm[i][:2]
        x2, y2 = points2_norm[i][:2]
        A.append([x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1])
    A = np.array(A)
    # Compute the singular value decomposition (SVD) of A
    _, _, V = np.linalg.svd(A)
    # Extract the rightmost column of V (corresponding to the smallest singular value)
    F = V[-1].reshape(3, 3)
    # Enforce rank 2 constraint by performing SVD on F and zeroing out the smallest singular value
    U, S, Vt = np.linalg.svd(F)
    S[-1] = 0
    F = U @ np.diag(S) @ Vt
    # Denormalize the fundamental matrix
    F = T2.T @ F @ T1

    return True, F


#
# estimate 8-point fundamental matrix many times using RANSAC algorithm
# points1 and points2 are 2D points in homogeneous coordinates
# It can be used with different F estimation functions (F_function)
# returns F, a 3x3 fundamental matrix
#
def ransac_fundamental_matrix(pts0, pts1, threshold = 0.5, iterations = 2000, F_function = estimate_normalized_8_point_fundamental_matrix):
    pts03 = np.ones((pts0.shape[0], 3))
    pts03[:, :2] = pts0
    pts13 = np.ones((pts1.shape[0], 3))
    pts13[:, :2] = pts1
    F = None
    mask = np.array([])
    bestRisidual = 999999999
    for i in range(iterations):
        # pick between 8 and 24 random points
        pointCount = min(pts0.shape[0], np.random.randint(8, 24))
        indices = np.random.choice(pts0.shape[0], pointCount, replace=False)  
        p0 = pts0[indices]
        p1 = pts1[indices]
        ret, Fpth = F_function(p0, p1)
        # Below is a faster way to write x1.T * F * x0 for every point pair in the list
        x = (Fpth @ pts03.T).T
        scores = (pts13[:,0] * x[:,0]) + (pts13[:,1] * x[:,1]) + (pts13[:,2] * x[:,2])
        # compute inliers
        inlierIndices = np.abs(scores) < threshold
        inliers0 = pts03[inlierIndices]
        # compute risidual by summing up scores
        risidual = scores[inlierIndices].T @ scores[inlierIndices]
        if inliers0.shape[0] > mask.shape[0]:
            mask = inlierIndices
            F = Fpth
            bestRisidual = 999999999
        elif inliers0.shape[0] == mask.shape[0] and risidual < bestRisidual:
            mask = inlierIndices
            F = Fpth 
            bestRisidual= risidual
            # print(risidual, scores.shape)
    print("inliers: ",mask.shape[0], " / " , pts03.shape[0])
    return F, mask

def homogenizePoints(pts2xn):
    pts3xn = np.ones((pts2xn.shape[0], 3))
    pts3xn[:, :2] = pts2xn
    return pts3xn

def drawLines(img, lines, pts):
    print(img.shape)
    imgb = img.copy()
    if len(img.shape) == 2:
        imgb = cv2.cvtColor(imgb, cv2.COLOR_GRAY2BGR)
    w = img.shape[1]
    h = img.shape[0]
    for r, pt in zip(lines, pts):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0 = 0
        y0 = -(r[0]*x0 + r[2])/r[1]
        x1 = w
        y1 = -(r[0]*x1 + r[2])/r[1]
        cv2.line(imgb, (int(x0),int(y0)), (int(x1),int(y1)), color,1)
        cv2.circle(imgb, (int(pt[0]),int(pt[1])), 5, color)
    return imgb


