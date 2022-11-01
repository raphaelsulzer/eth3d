import xml.etree.ElementTree as ET
import numpy as np
import open3d as o3d
import os

# dir = "/home/adminlocal/PhD/data/ETH3D/"
dir = "/mnt/raphael/ETH3D/"


classes = os.listdir(dir)

for c in classes:

    print(c)

    dataset_dir = os.path.join(dir,c)

    # Pass the path of the xml document
    tree = ET.parse(os.path.join(dataset_dir,'dslr_scan_eval','scan_alignment.mlp'))
    # get the parent tag
    root = tree.getroot()


    points = []
    sensor_pos = []
    pcd_combined = o3d.geometry.PointCloud()
    for i,child in enumerate(root[0]):
        f = child.attrib["filename"]

        pcd = o3d.io.read_point_cloud(os.path.join(dataset_dir,'dslr_scan_eval',f))

        mat = np.fromstring(root[0][i][0].text, dtype=np.float, sep="\n")

        mat = mat.reshape((4,4))

        points.append(np.asarray(pcd.points))
        sp = mat[:3,3]
        sp = np.expand_dims(sp,axis=0)
        sp = np.repeat(sp,len(pcd.points),axis=0)
        sensor_pos.append(sp)


        pcd_combined+=(pcd.transform(mat))


    # points = np.concatenate(points,axis=0)
    # sensor_pos = np.concatenate(sensor_pos,axis=0)
    # np.savez(os.path.join(dataset_dir,'convonet',"lidar.npz"),points=points,sensor_position=sensor_pos)


    points = np.concatenate(points,axis=0)
    sensor_pos = np.concatenate(sensor_pos,axis=0)
    ind = np.random.randint(points.shape[0], size=int(points.shape[0]/10))
    points = points[ind,:]
    sensor_pos = sensor_pos[ind,:]

    np.savez(os.path.join(dataset_dir,'convonet',"lidar_10.npz"),points=points,sensor_position=sensor_pos)

    a=5

    # # save the combined ground truth
    # o3d.io.write_point_cloud(os.path.join(dataset_dir,"gt.ply"),pcd_combined)
    #
    # obb=pcd_combined.get_oriented_bounding_box()
    #
    # data = np.load(os.path.join(dataset_dir,'scan',c+'.npz'))
    #
    # pcd = o3d.geometry.PointCloud()
    #
    # pcd.points = o3d.utility.Vector3dVector(data["points"])
    # pcd.normals = o3d.utility.Vector3dVector(data["normals"])
    # pcd.colors = o3d.utility.Vector3dVector(data["sensor_position"])
    #
    # pcd = pcd.crop(obb)
    #
    # os.makedirs(os.path.join(dataset_dir,'convonet'),exist_ok=True)
    # np.savez(os.path.join(dataset_dir,'convonet',"pointcloud.npz"),points=np.asarray(pcd.points),normals=np.asarray(pcd.normals),sensor_position=np.asarray(pcd.colors))
    #
    # # save the cropped scan as ply with color
    # pcd = o3d.io.read_point_cloud(os.path.join(dataset_dir,"openMVS","densify_file.ply"))
    # pcd = pcd.crop(obb)
    #
    # o3d.io.write_point_cloud(os.path.join(dataset_dir,"scan",c+"_cropped.ply"),pcd)
    #
    # a=5



