
from thirdparty.colmap.scripts.python.read_write_model import read_model
from dataset.sfm_solution import SFMSolution

import numpy as np
import os
from PIL import Image


class ColmapSolution(SFMSolution):

    def __init__(self, path_root, sol_nbr) -> None:
        super().__init__()

        self.path_root = path_root

        self.cameras, self.images, self.points = read_model(
            os.path.join(path_root, "sparse", str(sol_nbr)))

        self.grids = self.generate_grids()

    def generate_grids(self):

        grids = {}

        for key, camera in self.cameras.items():

            x = np.linspace(0, camera.width-1, camera.width)

            y = np.linspace(0, camera.height-1, camera.height)

            xv, yv = np.meshgrid(x, y)

            grids[key] = np.stack((xv, yv), axis=0)

        return grids

    def unit_rescale(self):
        pass

    def calculate_rays(self):

        total_rays = []

        total_tn = []
        total_tf = []
        total_color = []

        for _, image in self.images.items():

            img_path = os.path.join(self.path_root, "images", image.name)

            with Image.open(img_path) as im:

                im = np.asarray(im).astype(np.float32)/255.0

            im = np.transpose(im, [2, 0, 1])

            camera = self.cameras[image.camera_id]

            params = np.expand_dims(camera.params, -1)
            params = np.expand_dims(params, -1)

            focal, principal = np.split(params, 2)

            R = image.qvec2rotmat()

            t = image.tvec

            o = -R.transpose()@t

            d = (self.grids[image.camera_id]-principal)/focal

            d = np.concatenate((d, np.ones_like(d[0:1])), axis=0)

            C, H, W = d.shape

            d = (R.transpose() @ d.reshape(C, H*W))

            d = d / np.linalg.norm(d, axis=0)

            o = np.repeat(np.expand_dims(o, -1), H*W, axis=-1)

            rays = np.concatenate((o, d), axis=0)

            total_rays.append(rays.astype(np.float32))

            point3d = np.stack(
                [self.points[p_id].xyz for p_id in image.point3D_ids if p_id != -1], axis=-1)

            point3d = R @ point3d + np.expand_dims(t, -1)

            depth = point3d[2, :]

            tn = np.min(depth)/2
            tf = np.max(depth)*2

            total_tn.append(np.repeat(np.expand_dims(
                np.array(tn), -1), H*W, axis=-1))
            total_tf.append(np.repeat(np.expand_dims(
                np.array(tf), -1), H*W, axis=-1))

            total_color.append(im.reshape(3, H*W))

        total_rays = np.concatenate(total_rays, axis=-1)
        total_tn = np.expand_dims(np.concatenate(total_tn, axis=-1), 0)
        total_tf = np.expand_dims(np.concatenate(total_tf, axis=-1), 0)
        total_color = np.concatenate(total_color, axis=-1)

        return np.concatenate((total_rays, total_tn, total_tf, total_color), axis=0).astype(np.float32)
