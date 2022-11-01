import vedo
import numpy as np
from vedo import Plotter, Picture, dataurl
import os, sys
from scipy.spatial.transform import Rotation
from colmap_image import ColmapImage
from eth import ETH3D

PATH_TO_COLMAP="/home/adminlocal/PhD/cpp/colmap"
sys.path.append(os.path.join(PATH_TO_COLMAP,"scripts","python"))
from read_write_model import read_model, qvec2rotmat



cam_dict = {}
cam = dict(pos=(-12.06, -2.529, -1.610),
           focalPoint=(-1.018, -2.306, -3.224),
           viewup=(0.1446, 9.981e-4, 0.9895),
           distance=11.16,
           clippingRange=(0.08168, 81.68))
cam_dict["terrace"] = cam


class Visualizer:


    def __init__(self, path="/home/adminlocal/PhD/data/ETH3D",scene="terrace",size=(960, 658)):

        self.size = size
        self.scene = scene
        self.path = path

        self.outpath = "./out"

    def plotMeshes(self):

        ## LiDAR
        plt = vedo.Plotter(axes=0)

        mesh = vedo.Mesh(os.path.join(self.path,self.scene,"occlusion","surface_mesh.ply"),c=[180,180,180])
        mesh=mesh.computeNormals().phong().lighting("default")
        plt+=mesh
        # mesh = vedo.Mesh(os.path.join(path,"occlusion","splats.ply"),c="r")
        # mesh=mesh.computeNormals().phong().lighting("default")
        # plt+=mesh

        plt.show(camera=cam_dict[self.scene],interactive=True,size=self.size)
        plt.screenshot(os.path.join(self.outpath,"terrace_lidar_mesh.png"))
        plt.close()

        ## MVS
        plt = vedo.Plotter(axes=0)

        mesh = vedo.Mesh(os.path.join(self.path,self.scene,"openMVS","terrace_cl_05_textured.ply"),c=[180,180,180])
        mesh.texture(os.path.join(self.path,self.scene,"openMVS","terrace_cl_05_textured.png"))
        mesh=mesh.computeNormals().phong().lighting("default")
        plt+=mesh

        plt.show(camera=cam_dict[self.scene],interactive=False,size=self.size)
        plt.screenshot(os.path.join(self.outpath,"terrace_mvs_mesh.png"))
        plt.close()

    def plotMVS(self):

        # read colmap project
        cameras, images, points3D = read_model(os.path.join(self.path,self.scene, "dslr_calibration_undistorted"))

        # images_sample = dict(random.sample(images.items(), 4))
        images_sample = dict()
        images_sample[12] = images[12]
        # images_sample[8]=images[8]
        images_sample[9] = images[9]
        pc = vedo.load(os.path.join(self.path, self.scene,"openMVS", "densify_file.ply"))
        pc.pointSize(4)
        plt = vedo.Plotter(axes=0)
        plt += pc.lighting("default")

        interactive = False

        plt.show(camera=cam_dict[self.scene], interactive=interactive, size=self.size)
        plt.screenshot(os.path.join(self.outpath, "terrace_mvs.png"))

        spos = {}
        for k, img in images_sample.items():
            pic = Picture(os.path.join(self.path,self.scene, "images", img.name))
            pic.rotateX(180)

            cam = cameras[img.camera_id]
            cimg = ColmapImage(img, cam)

            spos[k] = cimg.t
            rm = Rotation.from_matrix(cimg.R)
            rm = rm.as_euler('xyz', degrees=True)

            scale = 0.0001
            pic.scale(scale)

            pic.addPos(-cam.width * scale / 2, cam.height * scale / 2,
                       cimg.K[0, 0] * scale)  # move image from lower left to center

            pic.rotateX(rm[0]), pic.rotateY(rm[1]), pic.rotateZ(rm[2])
            pic.addPos(cimg.t[0], cimg.t[1], cimg.t[2])

            pt = vedo.Point(cimg.t, c='orange', r=12)  # orientation center
            plt += pt.lighting("default")
            plt += pic

        sxyz_sample = []
        xyz = []
        xyz_sample = []
        rgb = []
        for point3D in points3D.values():
            image_id = point3D.image_ids[0]
            if image_id in spos:
                sxyz_sample.append(spos[point3D.image_ids[0]])
                xyz_sample.append(point3D.xyz)
            xyz.append(point3D.xyz)
            rgb.append(point3D.rgb / 255)

        ## add also the sparse point cloud
        # pts=vedo.Points(xyz,c=rgb,r=10)
        # plt+=pts

        lines = vedo.shapes.Lines(xyz_sample, sxyz_sample, c='y', lw=1, res=24, alpha=0.7)
        plt += lines.lighting("default")

        plt.show(camera=cam_dict[self.scene], interactive=interactive, size=self.size)
        plt.screenshot(os.path.join(self.outpath, "terrace_mvs_sensor.png"))
        plt.close()


    def plotLiDAR(self):


        ds = ETH3D(self.path, self.scene)
        ds.setPaths(lidar="scan_raw")
        ds.loadGT()

        plt = vedo.Plotter(axes=0, offscreen=False)

        si = 0

        ## chose only one sensor
        points = ds.points
        sensor_pos = ds.sensor_pos

        ind = np.random.randint(points.shape[0], size=int(points.shape[0] / 10))
        points = points[ind, :]
        sensor_pos = sensor_pos[ind, :]

        sub_points = points[points[:, 0] > ds.sensors[si, 0] * 0.3, :]
        sub_sensor_pos = sensor_pos[points[:, 0] > ds.sensors[si, 0] * 0.3, :]
        ind = np.random.randint(sub_points.shape[0], size=int(sub_points.shape[0] / 200))
        sub_points = sub_points[ind, :]
        sub_sensor_pos = sub_sensor_pos[ind, :]

        interactive = True

        ## plot the points
        pts = vedo.Points(points, c=[180, 180, 180], r=4)
        plt += pts.lighting("default")

        plt.show(camera=cam_dict[self.scene], interactive=interactive, size=self.size)
        plt.screenshot(os.path.join(self.outpath, "terrace_lidar.png"))

        ## plot the sensor
        sp = vedo.Points(ds.sensors, c='orange', r=12)  # orientation center
        plt += sp.lighting("default")

        ## plot lines to sensor
        lines = vedo.shapes.Lines(sub_points, sub_sensor_pos, c='y', lw=1, res=24, alpha=0.7)
        plt += lines.lighting("default")

        plt.show(camera=cam_dict[self.scene], interactive=interactive, size=self.size)
        plt.screenshot(os.path.join(self.outpath, "terrace_lidar_sensor.png"))
        plt.close()


if __name__ == "__main__":

    vc=Visualizer()

    vc.plotMeshes()
    # vc.plotMVS()
    # vc.plotLiDAR()
