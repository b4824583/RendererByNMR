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
# from .load_texture_by_obj import load_texture_by_obj
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
class Model(nn.Module):
    def __init__(self, filename_obj, filename_ref,useSilhouette=False,useDeltaAndScale=True):
        super(Model, self).__init__()
        vertices, faces = nr.load_obj(filename_obj)
        ##--------------------------
        # print(vertices)

        #############
        # delta v
        #??????????????????delta v ?????????????????????blender???????????????vertices??????????????????

        if useDeltaAndScale:
            delta_v_read = open("bird2_delta_v.txt", "r")
            # print(len(vertices))
            # print(vertices.shape()[0])
            for i in range(len(vertices)):
                # continue
                # print(vertices[i])
                vector=delta_v_read.readline().split()
                vertices[i][0]=vertices[i][0]+float(vector[0])
                vertices[i][1]=vertices[i][1]+float(vector[1])
                vertices[i][2]=vertices[i][2]+float(vector[2])
            ############
            ###########

            #scale original is 0.4
            # ????????????scale ???????????????blender?????????vertices?????????mesh npy??????????????????
            vertices=vertices*0.4
        #write new vertices
        f=open("deformBird.obj","w")
        f.write("mtllib spot_triangulated.mtl\n")
        f.write("o BirdReOrined_UnwrapBlender\n")
        for i in range(len(vertices)):
            f.write("v "+str(vertices[i][0].detach().cpu().numpy())+" "+str(vertices[i][1].detach().cpu().numpy())+" "+str(vertices[i][2].detach().cpu().numpy())+"\n")
        f.close()
        ##########
        ##########
        #rotataion
        cos=math.cos(4*(math.pi)/3)
        sin=math.sin(4*(math.pi)/3)
        cos_2=math.cos(3*(math.pi)/2)
        sin_2=math.sin(3*(math.pi)/2)
        for i in range(len(vertices)):
            #---------------------Y?????????
            vertices_0=cos_2*vertices[i][0]+sin_2*vertices[i][2]
            vertices_1=vertices[i][1]
            vertices_2=vertices[i][0]*sin_2*(-1)+cos_2*vertices[i][2]
            vertices[i][0]=vertices_0
            vertices[i][1]=vertices_1
            vertices[i][2]=vertices_2

            #---------------------Z?????????
            vertices_0=cos*vertices[i][0]-sin*vertices[i][1]
            vertices_1=sin*vertices[i][0]+cos*vertices[i][1]
            vertices_2=vertices[i][2]
            vertices[i][0]=vertices_0
            vertices[i][1]=vertices_1
            vertices[i][2]=vertices_2
            #-----------------------
            #------------------
            #---------------??????translation
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
        # texture size=1????????????????????????????????????????????????
        #texture size=2?????????????????????????????????????????????????????????????????????????????????????????????
        texture_size = 6
        textures = torch.zeros(1, self.faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32)
        self.textures = nn.Parameter(textures)
        #silhouette-----------------------
        mask = open("bird2_mask.txt", "r")
        filename_ref_data=imread(filename_ref)
        #---------------------????????????????????????????????????256*256??????????????????????????????
        # print(mask.shape())
        if useSilhouette:
            # filename_ref_data_silhouette=filename_ref_data
            for i in range(256):
                mask_element = mask.readline().split()
                for j in range(256):
                    # break
                    filename_ref_data[i][j]=filename_ref_data[i][j]*float(mask_element[j])
            imshow(filename_ref_data)
            imsave('data/birdie2_silhouette.png', img_as_ubyte(filename_ref_data))
        image_ref = torch.from_numpy(filename_ref_data.astype('float32') / 255.).permute(2,0,1)[None, ::]
        image_ref_flip=np.fliplr(filename_ref_data)
        # image_ref_flip=imread("birdie2_silhouette_switch.png")

        image_ref_flip=torch.from_numpy(image_ref_flip.astype('float32')/255.).permute(2,0,1)[None, ::]
        self.register_buffer('image_ref', image_ref)
        self.register_buffer("image_ref_2",image_ref_flip)

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-io', '--filename_obj', type=str, default=os.path.join('BirdReOrined_UnwrapBlender.obj'))
    parser.add_argument('-ir', '--filename_ref', type=str, default=os.path.join('birdie2.png'))


    parser.add_argument('-or', '--filename_output', type=str, default=os.path.join(data_dir, 'example_result.gif'))
    parser.add_argument('-g', '--gpu', type=int, default=0)
    args = parser.parse_args()

    model = Model(args.filename_obj, args.filename_ref,useSilhouette=True,useDeltaAndScale=True)
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
    # print(nr.get_points_from_angles(2.732,180,0))
    #??????????????????????????????????????????????????????????????????????????????????????????????????????????????????
    #????????????
    model.renderer.eye = nr.get_points_from_angles(2.732, 180, 0)
    # print(nr.get_points_from_angles(2.732,0,0))
    images, _, _ = model.renderer(model.vertices, model.faces, torch.tanh(model.textures))
    image = images.detach().cpu().numpy()[0].transpose((1, 2, 0))
    imsave('data/outputs.png', img_as_ubyte(image))
    # exit()
    #---------------------------test for read texture
    renderer_texture=nr.Renderer(image_size=256,camera_mode="look")
    renderer_texture.perspective = True
    renderer_texture.background_color=[1,1,1]
    renderer_texture.light_intensity_directional = 0.0
    renderer_texture.light_intensity_ambient = 1.0
    texture_vertices,texture_faces=load_texture_by_obj(args.filename_obj)
    texture_vertices=texture_vertices[None, :, :]
    texture_faces=texture_faces[None, :, :]
    texture_data=open("texture_data.txt","w")
    print(model.textures.shape)
    # exit()
    for face_num in(range(model.textures.shape[1])):
        for i in range(model.textures.shape[2]):
            for j in range(model.textures.shape[3]):
                for k in range(model.textures.shape[4]):
                    for color in range(3):
                        texture_data.write(str(model.textures[0][face_num][i][j][k][color].detach().cpu().numpy()))
                        if(color<2):
                            texture_data.write(" ")
                        else:
                            texture_data.write("\n")
    # print(model.textures[0])
    texture_data.close()
    # exit()
    renderer_texture.camera_direction=[0,0,1]
    renderer_texture.eye=[0.5,0.5,-1*(3**0.5)/2]
    images, _, _ = renderer_texture(texture_vertices, texture_faces, torch.tanh(model.textures))
    uv_image = images.detach().cpu().numpy()[0].transpose((1, 2, 0))


    imsave('data/texture.png', img_as_ubyte(uv_image))
    imshow(img_as_ubyte(uv_image))
    plt.show()
    #----------------------------------------------
    for num, azimuth in enumerate(loop):
        loop.set_description('Drawing')
        model.renderer.eye = nr.get_points_from_angles(2.732, 0, azimuth)
        images, _, _ = model.renderer(model.vertices, model.faces, torch.tanh(model.textures))
        image = images.detach().cpu().numpy()[0].transpose((1, 2, 0))
        imsave('/tmp/_tmp_%04d.png' % num, img_as_ubyte(image))
    make_gif(args.filename_output)


if __name__ == '__main__':
    main()
