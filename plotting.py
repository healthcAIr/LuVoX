import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure


def plot_3d(image, desc, prefixpath="", threshold=-300, show=False,
            StudyInstanceUID='StudyInstanceUID', SeriesInstanceUID='SeriesInstanceUID'):
    # Position the scan upright,
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2, 1, 0)

    verts, faces, normals, values = measure.marching_cubes(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    fpath = '{}-{}-{}.png'.format(StudyInstanceUID, SeriesInstanceUID, desc)
    if len(prefixpath):
        fpath = os.path.sep.join([prefixpath, fpath])
    if show:
        plt.show()
    else:
        plt.savefig(fpath, dpi=150)
    return fpath


def plot_2d(voxels):
    mid_X = voxels[int(voxels.shape[0] // 2), :, :]
    mid_Y = np.flipud(voxels[:, int(voxels.shape[1] // 2), :])
    mid_Z = np.flipud(voxels[:, :, int(voxels.shape[2] // 2)])
    return {
        'X': mid_X,
        'Y': mid_Y,
        'Z': mid_Z
    }


def create_overlays(clipped_voxels, lung_mask, alpha=0.2, imagage_count=5):
    overlays = []
    for idx, (hu, mask) in enumerate(zip(clipped_voxels, lung_mask)):
        b_mask = mask.astype(np.bool)
        if b_mask.sum():
            img = cv2.cvtColor(hu, cv2.COLOR_GRAY2BGR)
            img = img.copy()
            overlay = img.copy()
            overlay[b_mask] = (0.0, 1.0, 0.0)
            # apply the overlay
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
            overlays.append(img)

    ls = list(np.linspace(0, len(overlays), num=imagage_count + 2, dtype=np.int32, endpoint=True)[1:-1])
    overlays = [overlays[idx] for idx in ls]
    return overlays
