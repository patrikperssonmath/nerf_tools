
from thirdparty.colmap.scripts.python.read_write_model import rotmat2qvec, Camera, read_model, Point3D

from thirdparty.colmap.scripts.python.read_write_model import Image as colmap_image

from dataset.sfm_solution import SFMSolution

import numpy as np
import os
from PIL import Image
from torch import nn
import torch


def to_homogen(x):

    return np.concatenate((x, np.ones_like(x[0:1])), axis=0)


def to_K_matrix(intrinsics):

    K = np.eye(4, 4)

    fx, fy, cx, cy = intrinsics

    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = cx
    K[1, 2] = cy

    return K


def image_to_projection_matrix(image: colmap_image):

    R = image.qvec2rotmat()

    t = image.tvec

    P = np.eye(3, 4)
    P[0:3, 0:3] = R
    P[0:3, 3] = t

    return P


def image_to_camera_center(image: colmap_image):

    R = image.qvec2rotmat()

    t = image.tvec

    P = np.eye(4, 4)
    P[0:3, 0:3] = R
    P[0:3, 3] = t

    return np.linalg.inv(P)[0:3, 3]


def projection_matrix_to_image(P, image: colmap_image):

    q = rotmat2qvec(P[0:3, 0:3])
    t = P[0:3, 3]

    return colmap_image(id=image.id, qvec=q, tvec=t,
                        camera_id=image.camera_id, name=image.name,
                        xys=image.xys, point3D_ids=image.point3D_ids)


def iqr(x):

    q3, q1 = np.percentile(x, [75, 25], axis=-1)

    iqr = q3 - q1

    L = q1-1.5*iqr
    U = q3+1.5*iqr

    return L, U


class Solution:

    def __init__(self, path) -> None:
        self.cameras, self.images, self.points = read_model(path)

    def unit_transformations(self):

        points = [point.xyz for k, point in self.points.items()]
        camera_o = [image_to_camera_center(image)
                    for k, image in self.images.items()]

        points.extend(camera_o)

        x = np.array(points)
        x = np.transpose(x, (1, 0))

        x_mean = np.mean(x, axis=-1, keepdims=True)

        x_cen = x - x_mean

        s = np.max(np.abs(x_cen), keepdims=True)

        #s = iqr(np.abs(x_cen).reshape(-1))[1]

        T = np.eye(4, 4)

        T[0:3, 0:3] = s*np.eye(3, 3)
        T[0:3, 3:] = x_mean

        Tinv = np.linalg.inv(T)

        T = 1/s*T

        return T, Tinv

    def unit_rescale(self):

        T, Tinv = self.unit_transformations()

        for key, image in self.images.items():

            P = image_to_projection_matrix(image)

            P = P @ T

            self.images[key] = projection_matrix_to_image(P, image)

        for key, point in self.points.items():

            x = to_homogen(point.xyz)
            x = Tinv@x

            self.points[key] = Point3D(id=point.id, xyz=x[0:3], rgb=point.rgb,
                                       error=point.error, image_ids=point.image_ids,
                                       point2D_idxs=point.point2D_idxs)

    def rescale(self, size):

        Ho, Wo = size

        for key, camera in self.cameras.items():
            W = camera.width
            H = camera.height

            fx, fy, cx, cy = camera.params
            ax = Wo/W
            ay = Ho/H

            intrinsics = np.array([ax*fx, ay*fy, ax*cx, ay*cy])

            self.cameras[key] = Camera(id=camera.id, model=camera.model,
                                       width=Wo, height=Ho,
                                       params=intrinsics)

    def get_points(self, image):

        point3d = to_homogen(np.stack(
            [self.points[p_id].xyz for p_id in image.point3D_ids if p_id != -1], axis=-1))

        return point3d

    def get_camera(self, image) -> Camera:

        return self.cameras[image.camera_id]


class ColmapSolution(SFMSolution):

    def __init__(self, path_root, sol_nbr, size=None) -> None:
        super().__init__()

        self.path_root = path_root

        self.solution = Solution(os.path.join(
            path_root, "sparse", str(sol_nbr)))

        self.solution.unit_rescale()

        self.solution.rescale(size)

    def generate_grid(self, H, W):

        x = np.linspace(0, W-1, W)

        y = np.linspace(0, H-1, H)

        xv, yv = np.meshgrid(x, y)

        return np.stack((xv, yv), axis=0)

    def load_image(self, image: colmap_image):

        img_path = os.path.join(self.path_root, "images", image.name)

        with Image.open(img_path) as im:

            im = np.asarray(im).astype(np.float32)/255.0

        im = np.transpose(im, [2, 0, 1])

        camera = self.solution.get_camera(image)

        size = [camera.height, camera.width]

        im = nn.functional.interpolate(torch.tensor(im).unsqueeze(0), size,
                                       align_corners=True, mode="bilinear").squeeze(0).numpy()

        return im

    def extract_images(self):

        images = []

        for _, image in self.solution.images.items():

            im = self.load_image(image)

            P = image_to_projection_matrix(image)
            point3d = self.solution.get_points(image)
            camera = self.solution.get_camera(image)

            point3d = P@point3d

            depth = np.linalg.norm(point3d[0:3], axis=0)

            tn = np.min(depth)/2
            tf = np.max(depth)*2

            T = np.eye(4, 4)
            T[0:3, 0:4] = P

            data = {"image": im, "T": T,
                    "intrinsics": camera.params,
                    "tn": tn, "tf": tf}

            for key, val in data.items():

                data[key] = val.astype(np.float32)

            images.append(data)

        return images

    def calculate_rays(self):

        total_rays = []

        total_tn = []
        total_tf = []
        total_color = []

        for _, image in self.solution.images.items():

            im = self.load_image(image)

            P = image_to_projection_matrix(image)
            point3d = self.solution.get_points(image)
            camera = self.solution.get_camera(image)

            K = to_K_matrix(camera.params)

            point3d = P@point3d

            depth = np.linalg.norm(point3d[0:3], axis=0)

            tn = np.min(depth)/2
            tf = np.max(depth)*2

            T = np.eye(4, 4)
            T[0:3, 0:4] = P

            Tinv = np.linalg.inv(T)
            Kinv = np.linalg.inv(K)

            M = Tinv[0:3, 0:3]@Kinv[0:3, 0:3]

            H, W = im.shape[1:]
            grid = self.generate_grid(H, W)
            grid = np.concatenate((grid, np.ones_like(grid[0:1])), axis=0)

            o = Tinv[0:3, 3]

            d = M@grid.reshape(3, H*W)

            d = d[0:3]/np.linalg.norm(d[0:3], axis=0)

            o = np.repeat(o[:, None], H*W, axis=-1)

            rays = np.concatenate((o, d), axis=0)

            total_rays.append(rays.astype(np.float32))
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
