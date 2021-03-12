
"""
Example 1. Drawing a teapot from multiple viewpoints.
"""
import os
import argparse

import torch
import numpy as np
import tqdm
import imageio
from skimage.io import imread, imsave,imshow
import neural_renderer as nr
from skimage import img_as_ubyte
current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, 'data')
from matplotlib import pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--filename_input', type=str, default=os.path.join(data_dir, 'teapot.obj'))
    parser.add_argument('-o', '--filename_output', type=str, default=os.path.join(data_dir, 'example1.gif'))
    parser.add_argument('-g', '--gpu', type=int, default=0)
    args = parser.parse_args()

    # other settings
    camera_distance = 2.732
    texture_size = 2

    # load .obj
    vertices, faces = nr.load_obj(args.filename_input)
    vertices = vertices[None, :, :]  # [num_vertices, XYZ] -> [batch_size=1, num_vertices, XYZ]
    faces = faces[None, :, :]  # [num_faces, 3] -> [batch_size=1, num_faces, 3]

    # create texture [batch_size=1, num_faces, texture_size, texture_size, texture_size, RGB]
    textures = torch.ones(1, faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32).cuda()

    # to gpu

    # create renderer
    # renderer = nr.Renderer(camera_mode='look_at')
    K=np.array([[[150,0,0],
               [0,150,0],
               [0,0,1]]],dtype=np.float32)
    R=np.array([[[1,0,0],
                [0,1,0],
                [0,0,1]]],dtype=np.float32)
    # print(K.shape)
    # exit()
    t=np.array([[0,0,0]],dtype=np.float32)
    renderer=nr.Renderer(camera_mode="projection",K=K,R=R,t=t)
    # draw object
    # loop = tqdm.tqdm(range(0, 360, 4))
    elevation = 0
    azimuth=180

    # writer = imageio.get_writer(args.filename_output, mode='I')
    # renderer.eye=  nr.get_points_from_angles(camera_distance,elevation,azimuth)
    # renderer.eye=
    # renderer.eye = nr.get_points_from_angles(camera_distance, elevation, azimuth)
    images, _, _ = renderer(vertices, faces, textures)  # [batch_size, RGB, image_size, image_size]
    image = images.detach().cpu().numpy()[0].transpose((1, 2, 0))  # [image_size, image_size, RGB]
    # print(image)
    # print(image[0][0])
    imsave('data/teapot.png', img_as_ubyte(image))
    # imshow(img_as_ubyte(image))
    # plt.show()


    renderer2 = nr.Renderer(camera_mode='look')
    renderer2.camera_direction=[0,0,1]
    renderer2.eye=[0.5,0.5,-1*(3**0.5)]
    # writer = imageio.get_writer(args.filename_output, mode='I')
    # renderer.eye=  nr.get_points_from_angles(camera_distance,elevation,azimuth)
    # renderer.eye=
    # renderer.eye = nr.get_points_from_angles(camera_distance, elevation, azimuth)
    images, _, _ = renderer2(vertices, faces, textures)  # [batch_size, RGB, image_size, image_size]
    image = images.detach().cpu().numpy()[0].transpose((1, 2, 0))  # [image_size, image_size, RGB]
    # imsave('data/teapot.png', img_as_ubyte(image))
    imshow(img_as_ubyte(image))
    plt.show()


    # writer.append_data((255*image).astype(np.uint8))

    # for num, azimuth in enumerate(loop):
    #     loop.set_description('Drawing')
    #     renderer.eye = nr.get_points_from_angles(camera_distance, elevation, azimuth)
    #     images, _, _ = renderer(vertices, faces, textures)  # [batch_size, RGB, image_size, image_size]
    #     image = images.detach().cpu().numpy()[0].transpose((1, 2, 0))  # [image_size, image_size, RGB]
    #     writer.append_data((255*image).astype(np.uint8))
    # writer.close()

if __name__ == '__main__':
    main()
