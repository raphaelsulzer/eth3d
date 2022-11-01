import xml.etree.ElementTree as ET
import numpy as np
import open3d as o3d
import os

class ETH3D:
    def __init__(self,path,scene):

        self.path=path
        self.scene=scene

        self.setPaths()

    def setPaths(self,lidar="dslr_scan_eval"):
        self.lidar_path = lidar

    def loadGT(self,type="eval"):
        # Pass the path of the xml document
        tree = ET.parse(os.path.join(self.path, self.scene, self.lidar_path, 'scan_alignment.mlp'))
        # get the parent tag
        root = tree.getroot()

        points = []
        sensor_pos = []
        sensor_idx = []
        sensors = []
        pcd_combined = o3d.geometry.PointCloud()
        for i, child in enumerate(root[0]):
            f = child.attrib["filename"]

            pcd = o3d.io.read_point_cloud(os.path.join(self.path, self.scene, self.lidar_path, f))

            pcd = pcd.voxel_down_sample(0.005)


            mat = np.fromstring(root[0][i][0].text, dtype=np.float, sep="\n")

            mat = mat.reshape((4, 4))

            points.append(np.asarray(pcd.points))
            sp = mat[:3, 3]
            sensors.append(sp)
            sp = np.expand_dims(sp, axis=0)
            sp = np.repeat(sp, len(pcd.points), axis=0)
            sensor_pos.append(sp)

            si = np.expand_dims(i, axis=0)
            si = np.repeat(si, len(pcd.points), axis=0)
            sensor_idx.append(si)

            pcd_combined += (pcd.transform(mat))

        self.points = np.concatenate(points, axis=0)
        self.sensor_pos = np.concatenate(sensor_pos, axis=0)
        self.sensor_idx = np.concatenate(sensor_idx, axis=0)
        self.sensors = np.array(sensors)
        # self.lidar_pc = pcd_combined


    def exportGT(self,outpath=""):

        if not outpath:
            outpath = os.path.join(self.path,self.scene,self.lidar_path,"lidar.ply")

        o3d.io.write_point_cloud(outpath,self.lidar_pc)



