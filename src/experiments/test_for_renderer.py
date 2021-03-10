"""
Example 3. Optimizing textures.
"""
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

import neural_renderer as nr
from matplotlib import pyplot as plt
current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, 'data')

class Model(nn.Module):
    def __init__(self, filename_obj, filename_ref):
        super(Model, self).__init__()
        vertices, faces = nr.load_obj(filename_obj)
        # print(faces[-1])
        # exit()
        ##--------------------------
        # print(vertices)

        #############
        # delta v
        #目前先嘗試把delta v 拿掉，原因是，blender產生出來的vertices順序會打亂。
        delta_v_read = open("bird3_delta_v.txt", "r")
        for i in range(len(vertices)):
            # continue
            vector=delta_v_read.readline().split()
            vertices[i][0]=vertices[i][0]+float(vector[0])
            vertices[i][1]=vertices[i][1]+float(vector[1])
            vertices[i][2]=vertices[i][2]+float(vector[2])
        # print(vertices.shape)
        ############
        ###########

        #scale original is 0.4
        # 有一度把scale 拿掉，因為blender產生的vertices與原本mesh npy檔案的不一樣
        vertices=vertices*0.4


        ##########
        ##########
        #rotataion
        cos=math.cos(4*(math.pi)/3)
        sin=math.sin(4*(math.pi)/3)
        # print(vertices[0])
        cos_2=math.cos(3*(math.pi)/2)
        sin_2=math.sin(3*(math.pi)/2)
        for i in range(len(vertices)):
            #---------------------Y軸旋轉
            vertices_0=cos_2*vertices[i][0]+sin_2*vertices[i][2]
            vertices_1=vertices[i][1]
            vertices_2=vertices[i][0]*sin_2*(-1)+cos_2*vertices[i][2]
            vertices[i][0]=vertices_0
            vertices[i][1]=vertices_1
            vertices[i][2]=vertices_2

            #---------------------Z軸旋轉
            vertices_0=cos*vertices[i][0]-sin*vertices[i][1]
            vertices_1=sin*vertices[i][0]+cos*vertices[i][1]
            vertices_2=vertices[i][2]
            vertices[i][0]=vertices_0
            vertices[i][1]=vertices_1
            vertices[i][2]=vertices_2
            #-----------------------
            #------------------
            vertices[i][1]-=0.015
            vertices[i][0]-=0.18

            #------------------
        #################################
        # for i in range(len(vertices)):
        #     vertices[i][1]+=0.
        # exit()
        self.register_buffer('vertices', vertices[None, :, :])
        self.register_buffer('faces', faces[None, :, :])

        # create textures
        # texture size=1的時候每個三角形都各自有一個顏色
        #texture size=2的時候就很神奇了，三角形像是有內插的顏色一樣，這一點我不太理解
        texture_size = 4
        textures = torch.zeros(1, self.faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32)
        self.textures = nn.Parameter(textures)
        #silhouette-----------------------
        mask = open("bird3_mask.txt", "r")
        filename_ref_data=imread(filename_ref)
        # filename_ref_data_silhouette=filename_ref_data
        for i in range(256):
            mask_element = mask.readline().split()
            for j in range(256):
                # break
                filename_ref_data[i][j]=filename_ref_data[i][j]*float(mask_element[j])
        imshow(filename_ref_data)
        imsave('data/birdie3_silhouette.png', img_as_ubyte(filename_ref_data))
        # plt.show()
        # exit()
        image_ref = torch.from_numpy(filename_ref_data.astype('float32') / 255.).permute(2,0,1)[None, ::]
        image_ref_2=imread("birdie2_silhouette_switch.png")

        image_ref_2=torch.from_numpy(image_ref_2.astype('float32')/255.).permute(2,0,1)[None, ::]
        self.register_buffer('image_ref', image_ref)
        self.register_buffer("image_ref_2",image_ref_2)

        # setup renderer
        renderer = nr.Renderer(camera_mode='look_at')
        renderer.perspective = False
        renderer.light_intensity_directional = 0.0
        renderer.light_intensity_ambient = 1.0
        self.renderer = renderer


    def forward(self):
        self.renderer.eye = nr.get_points_from_angles(2.732, 0, np.random.uniform(0, 360))
        # self.renderer.eye=
        self.renderer.eye = nr.get_points_from_angles(2.732, 180,0)
        # image_ref = self.image_ref.detach().cpu().numpy()[0].transpose((1, 2, 0))



        image, _, _ = self.renderer(self.vertices, self.faces, torch.tanh(self.textures))
        loss_one_side = torch.sum((image - self.image_ref) ** 2)

        self.renderer.eye = nr.get_points_from_angles(2.732, 0,0)
        image, _, _ = self.renderer(self.vertices, self.faces, torch.tanh(self.textures))
        loss_symmetric = torch.sum((image - self.image_ref_2) ** 2)

        loss=(loss_one_side+loss_symmetric)/2

        return loss


def make_gif(filename):
    with imageio.get_writer(filename, mode='I') as writer:
        for filename in sorted(glob.glob('/tmp/_tmp_*.png')):
            writer.append_data(imageio.imread(filename))
            os.remove(filename)
    writer.close()
def load_obj_get_texture_vertices_and_face(filename_obj, normalization=True, texture_size=4, load_texture=False,
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

    # print(len(vertices[0]))
    for i in range(len(vertices)):
        vertices[i].append(0.0)
    # print(len(vertices[0]))
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

    # print(faces[0])
    return vertices, faces



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-io', '--filename_obj', type=str, default=os.path.join(data_dir, 'teapot.obj'))
    parser.add_argument('-ir', '--filename_ref', type=str, default=os.path.join(data_dir, 'example3_ref.png'))


    parser.add_argument('-or', '--filename_output', type=str, default=os.path.join(data_dir, 'example_result.gif'))
    parser.add_argument('-g', '--gpu', type=int, default=0)
    args = parser.parse_args()

    model = Model(args.filename_obj, args.filename_ref)
    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1, betas=(0.5,0.999))
    loop = tqdm.tqdm(range(300))
    for _ in loop:
        loop.set_description('Optimizing')
        optimizer.zero_grad()
        loss = model()
        loss.backward()
        optimizer.step()

    # draw object
    # loop = tqdm.tqdm(range(0, 360, 4))
    model.renderer.eye = nr.get_points_from_angles(2.732, 180, 0)

    images, _, _ = model.renderer(model.vertices, model.faces, torch.tanh(model.textures))
    image = images.detach().cpu().numpy()[0].transpose((1, 2, 0))
    imsave('data/outputs.png', img_as_ubyte(image))

    #---------------------------test for read texture
    texture_vertices,texture_faces=load_obj_get_texture_vertices_and_face(args.filename_obj)
    texture_vertices=texture_vertices[None, :, :]
    texture_faces=texture_faces[None, :, :]
    for i in range(len(texture_vertices[0])):
        texture_vertices[0][i][0]*=2
        texture_vertices[0][i][1]*=2

    for i in range(len(texture_vertices[0])):
        texture_vertices[0][i][0]-=1
        texture_vertices[0][i][1]-=1

    # texture_vertices=texture_vertices*1.5
    # temp_x=0.0
    # temp_y=0.0
    # min_x=1.0
    # min_y=1.0
    #
    # print(texture_vertices.size())
    # for i in range(len(texture_vertices[0])):
    #     if(temp_x<texture_vertices[0][i][0]):
    #         temp_x=texture_vertices[0][i][0]
    #     if(temp_y<texture_vertices[0][i][1]):
    #         temp_y=texture_vertices[0][i][1]
    #     if(min_x>texture_vertices[0][i][0]):
    #         min_x=texture_vertices[0][i][0]
    #     if(min_y>texture_vertices[0][i][1]):
    #         min_y=texture_vertices[0][i][1]
    # print(temp_x)
    # print(temp_y)
    # print(min_x)
    # print(min_y)
    # exit()
    model.renderer.eye = nr.get_points_from_angles(2.732, 0, 0)
    images, _, _ = model.renderer(texture_vertices, texture_faces, torch.tanh(model.textures))
    image = images.detach().cpu().numpy()[0].transpose((1, 2, 0))
    #-----test white texture png
    #-----
    # for i in range(len(image)):
    #     for j in range(len(image[i])):
    #         if(image[i][j][0]==0.0 and image[i][j][1]==0.0 and image[i][j][2]==0.0):
    #             image[i][j][0]=1.0
    #             image[i][j][1]=1.0
    #             image[i][j][2]=1.0
    # exit()
    imsave('data/texture.png', img_as_ubyte(image))
    imshow(img_as_ubyte(image))
    plt.show()

    # exit()
    for num, azimuth in enumerate(loop):
        loop.set_description('Drawing')
        model.renderer.eye = nr.get_points_from_angles(2.732, 0, azimuth)

        # model.textures = torch.zeros(1, self.faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32)
        # model.textures=torch.rand(1,555,6,6,6,3,dtype=torch.float32)
        # textures = torch.zeros(1, 555, 4, 4, 4, 3, dtype=torch.float32)
        # model.textures = nn.Parameter(textures)



        images, _, _ = model.renderer(model.vertices, model.faces, torch.tanh(model.textures))
        image = images.detach().cpu().numpy()[0].transpose((1, 2, 0))
        imsave('/tmp/_tmp_%04d.png' % num, img_as_ubyte(image))
    make_gif(args.filename_output)


if __name__ == '__main__':
    main()
