import numpy as np
import sys, os
PATH_TO_COLMAP="/home/adminlocal/PhD/cpp/colmap"
sys.path.append(os.path.join(PATH_TO_COLMAP,"scripts","python"))
from read_write_model import qvec2rotmat


class ColmapImage:

    # code taken from COLMAP repo: scripts/python/visualize_model.py

    def __init__(self,img,cam):

        R = qvec2rotmat(img.qvec)

        # translation
        t = img.tvec

        # invert
        self.t = -R.T @ t
        self.R = R.T

        T = np.column_stack((R, t))
        self.T = np.vstack((T, (0, 0, 0, 1)))

        if cam.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
            fx = fy = cam.params[0]
            cx = cam.params[1]
            cy = cam.params[2]
        elif cam.model in ("PINHOLE", "OPENCV", "OPENCV_FISHEYE"):
            fx = cam.params[0]
            fy = cam.params[1]
            cx = cam.params[2]
            cy = cam.params[3]
        else:
            raise Exception("Camera model not supported")

        # intrinsics
        K = np.identity(3)
        K[0, 0] = fx
        K[1, 1] = fy
        K[0, 2] = cx
        K[1, 2] = cy

        self.K = K
        self.Kinv = np.linalg.inv(K)