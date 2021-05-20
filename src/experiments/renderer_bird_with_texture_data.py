
from __future__ import division
import os
import argparse
import glob

import torch
import torch.nn as nn
import numpy as np
from skimage.io import imread, imsave,imshow
import tqdm
import imageio
from skimage import img_as_ubyte
import math
# from .load_texture_by_obj import load_texture_by_obj
import neural_renderer as nr
from matplotlib import pyplot as plt
current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, 'data')
def load_texture_by_obj(filename_obj, normalization=True, texture_size=4, load_texture=False,
             texture_wrapping='REPEAT', use_bilinear=True):
    """
    Load Wavefront .obj file.
    This function only supports vertices (v x x x) and faces (f x x x).
    """

    # load vertices
    vertices = []
    with open(filename_obj) as f:
        lines = f.readlines()

    for line in lines:
        if len(line.split()) == 0:
            continue
        if line.split()[0] == 'vt':
            vertices.append([float(v) for v in line.split()[1:4]])

    for i in range(len(vertices)):
        vertices[i].append(0.0)
    vertices = torch.from_numpy(np.vstack(vertices).astype(np.float32)).cuda()

    # load faces
    faces = []
    for line in lines:
        if len(line.split()) == 0:
            continue
        if line.split()[0] == 'f':
            vs = line.split()[1:]
            nv = len(vs)
            v0 = int(vs[0].split('/')[1])
            for i in range(nv - 2):
                v1 = int(vs[i + 1].split('/')[1])
                v2 = int(vs[i + 2].split('/')[1])
                faces.append((v0, v1, v2))
    faces = torch.from_numpy(np.vstack(faces).astype(np.int32)).cuda() - 1

    return vertices, faces
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-io', '--filename_obj', type=str, default=os.path.join('BirdReOrined_UnwrapBlender.obj'))
    parser.add_argument('-ir', '--filename_ref', type=str, default=os.path.join(data_dir, 'birdie2.png'))


    parser.add_argument('-or', '--filename_output', type=str, default=os.path.join(data_dir, 'example_result.gif'))
    parser.add_argument('-g', '--gpu', type=int, default=0)
    args = parser.parse_args()


    texture_vertices,texture_faces=load_texture_by_obj(args.filename_obj)
    texture_vertices=texture_vertices[None, :, :]
    texture_faces=texture_faces[None, :, :]
    texture_data=open("texture_data.txt","r")
    cuda0 = torch.device('cuda:0')
    model_textures=torch.zeros([1,1106,6,6,6,3],dtype=torch.float32,device=cuda0)
    for faces in range(1106):
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    point_colors=texture_data.readline().split()
                    for color in range(3):
                        model_textures[0][faces][i][j][k][color]=float(point_colors[color])
    # print(model_textures.shape)
    # print(model_textures[0][0][0][0])
    # exit()
    renderer_texture=nr.Renderer(image_size=256,camera_mode="look",fill_back=False)
    renderer_texture.perspective = True
    renderer_texture.background_color=[1,1,1]
    renderer_texture.light_intensity_directional = 0.0
    renderer_texture.light_intensity_ambient = 1.0
    renderer_texture.camera_direction=[0,0,1]
    renderer_texture.eye=[0.5,0.5,-1*(3**0.5)/2]
    images, _, _ = renderer_texture(texture_vertices, texture_faces, torch.tanh(model_textures))
    uv_image = images.detach().cpu().numpy()[0].transpose((1, 2, 0))


    imsave('data/texture_data_output.png', img_as_ubyte(uv_image))
    imshow(img_as_ubyte(uv_image))
    plt.show()
    #----------------------------------------------
    # for num, azimuth in enumerate(loop):
    #     loop.set_description('Drawing')
    #     model.renderer.eye = nr.get_points_from_angles(2.732, 0, azimuth)
    #     images, _, _ = model.renderer(model.vertices, model.faces, torch.tanh(model.textures))
    #     image = images.detach().cpu().numpy()[0].transpose((1, 2, 0))
    #     imsave('/tmp/_tmp_%04d.png' % num, img_as_ubyte(image))
    # make_gif(args.filename_output)


if __name__ == '__main__':
    main()