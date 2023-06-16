
from thirdparty.colmap.scripts.python.read_write_model import read_model
from dataset.sfm_solution import SFMSolution

import numpy as np
import os
from PIL import Image
from torch import nn
import torch


class ColmapSolution(SFMSolution):

    def __init__(self, path_root, sol_nbr, size = None) -> None:
        super().__init__()

        self.path_root = path_root
        self.size = size

        self.cameras, self.images, self.points = read_model(
            os.path.join(path_root, "sparse", str(sol_nbr)))

    def generate_grid(self, H, W):

        x = np.linspace(0, W-1, W)

        y = np.linspace(0, H-1, H)

        xv, yv = np.meshgrid(x, y)

        return np.stack((xv, yv), axis=0)

    def unit_rescale(self):
        pass

    def extract_images(self):

        images = []

        for _, image in self.images.items():

            img_path = os.path.join(self.path_root, "images", image.name)

            with Image.open(img_path) as im:

                im = np.asarray(im).astype(np.float32)/255.0

            im = np.transpose(im, [2, 0, 1])

            camera = self.cameras[image.camera_id]

            R = image.qvec2rotmat()

            t = image.tvec

            T = np.eye(4)

            T[0:3, 0:3] = R
            T[0:3, 3] = t

            point3d = np.stack(
                [self.points[p_id].xyz for p_id in image.point3D_ids if p_id != -1], axis=-1)

            point3d = R @ point3d + np.expand_dims(t, -1)

            depth = point3d[2, :]

            tn = np.min(depth)/2
            tf = np.max(depth)*2

            intrinsics = camera.params

            if self.size is not None:

                im, intrinsics = self.rescale(im, intrinsics)

            data = {"image": im, "T": T,
                    "intrinsics": intrinsics, "tn": tn, "tf": tf}

            for key, val in data.items():

                data[key] = val.astype(np.float32)

            images.append(data)

        return images

    def rescale(self, im, intrinsics):

        fx, fy, cx, cy = intrinsics

        C, H, W = im.shape

        im = nn.functional.interpolate(
            torch.tensor(im).unsqueeze(0), self.size, align_corners=True, mode="bilinear").squeeze(0).numpy()

        C, Ho, Wo = im.shape

        ax = Wo/W
        ay = Ho/H

        intrinsics = np.array([ax*fx, ay*fy, ax*cx, ay*cy])

        return im, intrinsics

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

            intrinsics = camera.params

            if self.size is not None:

                im, intrinsics = self.rescale(im, intrinsics)

            params = np.expand_dims(intrinsics, -1)
            params = np.expand_dims(params, -1)

            focal, principal = np.split(params, 2)

            R = image.qvec2rotmat()

            t = image.tvec

            o = -R.transpose()@t

            grid = self.generate_grid(im.shape[1], im.shape[2])

            d = (grid-principal)/focal

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

            depth = np.linalg.norm(point3d, axis=0)  # point3d[2, :]

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
