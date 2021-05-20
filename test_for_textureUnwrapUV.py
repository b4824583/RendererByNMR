import numpy as np
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../')
from src.utils import mesh
from src.nnutils import geom_utils
import torch
import pymesh
from src.nnutils import train_utils
from absl import app, flags
from src.utils import visutil
  #######################
        ### Setup Mean Shape
        #######################
from src.data import cub as cub_data
from src.utils import image as image_utils
import torchvision
from src.data import imagenet as imagenet_data
from src.data import json_dataset as json_data
from src.data import p3d as p3d_data
flags.DEFINE_string('dataset', 'cub', 'yt (YouTube), or cub, or yt_filt (Youtube, refined)')
# Weights:
flags.DEFINE_float('rend_mask_loss_wt', 0, 'rendered mask loss weight')
flags.DEFINE_float('deform_loss_wt', 0, 'reg to deformation')
flags.DEFINE_float('laplacian_loss_wt', 0, 'weights to laplacian smoothness prior')
flags.DEFINE_float('meanV_laplacian_loss_wt', 0, 'weights to laplacian smoothness prior on mean shape')
flags.DEFINE_float('deltaV_laplacian_loss_wt', 0, 'weights to laplacian smoothness prior on delta shape')
flags.DEFINE_float('graphlap_loss_wt', 0, 'weights to graph laplacian smoothness prior')
flags.DEFINE_float('edge_loss_wt', 0, 'weights to edge length prior')
flags.DEFINE_float('texture_loss_wt', 0, 'weights to tex loss')
flags.DEFINE_float('camera_loss_wt', 0, 'weights to camera loss')

flags.DEFINE_boolean('perspective', False, 'whether to use strong perrspective projection')
flags.DEFINE_string('shape_path', '', 'Path to initial mean shape')
flags.DEFINE_integer('num_multipose', -1, 'num_multipose_az * num_multipose_el')
flags.DEFINE_integer('num_multipose_az', 8, 'Number of camera pose hypothesis bins (along azimuth)')
flags.DEFINE_integer('num_multipose_el', 5, 'Number of camera pose hypothesis bins (along elevation)')
flags.DEFINE_boolean('use_gt_camera', False, 'Use ground truth camera pose')
flags.DEFINE_boolean('viz_rend_video', True, 'Render video to visualize mesh')
flags.DEFINE_integer('viz_rend_steps', 36, 'Number of angles to visualize mesh from')

flags.DEFINE_float('initial_quat_bias_deg', 90, 'Rotation bias in deg. 90 for head-view, 45 for breast-view')

flags.DEFINE_float('scale_bias', 0.8, 'Scale bias for camera pose')
flags.DEFINE_boolean('optimizeCameraCont', True, 'Optimize Camera Continuously')
flags.DEFINE_float('optimizeLR', 0.0001, 'Learning rate for camera pose optimization')
flags.DEFINE_float('optimizeMomentum', 0, 'Momentum for camera pose optimization')
flags.DEFINE_enum('optimizeAlgo', 'adam', ['sgd','adam'], 'Algo for camera pose optimization')
flags.DEFINE_float('quatScorePeakiness', 20, 'quat score = e^(-peakiness * loss)')
flags.DEFINE_string('cameraPoseDict', '', 'Path to pre-computed camera pose dict for entire dataset')

flags.DEFINE_float('softras_sigma', 1e-5, 'Softras sigma (transparency)')
flags.DEFINE_float('softras_gamma', 1e-4, 'Softras gamma (blurriness)')

flags.DEFINE_boolean('texture_flipCam', False, 'Render flipped mesh and supervise using flipped image')

flags.DEFINE_boolean('pred_pose_supervise', True, 'Supervise predicted pose using best camera (in multiplex, or gt)')
flags.DEFINE_boolean('optimizeCamera_reloadCamsFromDict', False, 'Ignore camera pose from saved dictionary')

flags.DEFINE_boolean('laplacianDeltaV', False, 'Smooth DeltaV instead of V in laplacian_loss')

flags.DEFINE_float('optimizeAzRange', 30, 'Optimize Azimuth range (degrees')
flags.DEFINE_float('optimizeElRange', 30, 'Optimize Elevation range (degrees')
flags.DEFINE_float('optimizeCrRange', 60, 'Optimize CycloRotation range (degrees')
flags.DEFINE_boolean('textureUnwrapUV', True, 'UV map onto 2d image directly, not a sphere')
flags.DEFINE_boolean('pred_shape', True, 'Predict Shape')
flags.DEFINE_boolean('pred_texture', True, 'Predict Texture')
flags.DEFINE_integer('tex_size', 6, 'Texture resolution per face')
flags.DEFINE_boolean('texture_uvshift', True, 'Shift uv-map along x to symmetrize it')

# flags.DEFINE_string('name','cub_train_cam8x5_test','')
flags.DEFINE_string('flagfile','configs/cub-train.cfg','')
flags.DEFINE_integer('num_epoches',21,'')
# flags.DEFINE_string('cameraPoseDict','../cachedir/logs/cub_init_campose8x5/stats/campose_0.npz','')
# opts.name = "cub_train_cam8x5_test"
# opts.flagfile = "../configs/cub-train.cfg"
# opts.num_epoches = 21
# opts.cameraPoseDict = "../cachedir/logs/cub_init_campose8x5/stats/campose_0.npz"

opts = flags.FLAGS
class ShapeTrainer(train_utils.Trainer):
    def define_model(self):
        mean_shape = mesh.fetch_mean_shape(opts.shape_path, mean_centre_vertices=True)
        verts = mean_shape['verts']
        faces = mean_shape['faces']
        verts_uv = mean_shape['verts_uv']
        faces_uv = mean_shape['faces_uv']
        # # Visualize uvmap
    #        misc_utils.plot_triangles(faces_uv)

        self.verts_uv = torch.from_numpy(verts_uv).float() # V,2
        self.verts = torch.from_numpy(verts).float() # V,3
        self.faces = torch.from_numpy(faces).long()  # F,2
        self.faces_uv = torch.from_numpy(faces_uv).float()  # F,3,2
        #使用obj file的時候不會提取那個資料的verts_uv，所以不會觸發這一行的錯誤，這一點滿奇特的
        assert(verts_uv.shape[0] == verts.shape[0])
        assert(verts_uv.shape[1] == 2)
        assert(verts.shape[1] == 3)
        assert(faces.shape[1] == 3)
        assert(faces_uv.shape == (faces.shape)+(2,))
        #這邊convert_uv_to_3d_coordinates這一行把uv的點變回球體，但調鬼的是
        #這件事情其實跟一開始把它從2d壓扁成3d是差不多的
        # Store UV sperical texture map
        verts_sph = geom_utils.convert_uv_to_3d_coordinates(verts_uv)
        if not opts.textureUnwrapUV:
            uv_sampler = mesh.compute_uvsampler_softras(verts_sph, faces, tex_size=opts.tex_size, shift_uv=opts.texture_uvshift)
        else:
            print("unsampler_softras_unwrapUV")
            uv_sampler = mesh.compute_uvsampler_softras_unwrapUV(faces_uv, faces, tex_size=opts.tex_size, shift_uv=opts.texture_uvshift)
        uv_texture = visutil.uv2bgr(uv_sampler) # F,T,T,3
        uv_texture = np.repeat(uv_texture[:,:,:,None,:], opts.tex_size, axis=3) # F,T,T,T,2
        self.uv_texture = torch.tensor(uv_texture).float().cuda()/255.

        if not opts.textureUnwrapUV:
            uv_sampler_nmr = mesh.compute_uvsampler(verts_sph, faces, tex_size=opts.tex_size, shift_uv=opts.texture_uvshift)
        else:
            print("unsampler_unwrapUV")
            uv_sampler_nmr = mesh.compute_uvsampler_unwrapUV(faces_uv, faces, tex_size=opts.tex_size, shift_uv=opts.texture_uvshift)
        # exit()
    def init_dataset(self):
        opts = self.opts
        if opts.dataset == 'cub':
            dataloader_fn = cub_data.data_loader
        elif opts.dataset == 'imnet':
            dataloader_fn = imagenet_data.imnet_dataloader
        elif opts.dataset == 'p3d':
            dataloader_fn = p3d_data.data_loader
        elif opts.dataset == 'json':
            dataloader_fn = json_data.data_loader
        else:
            raise ValueError('Unknown dataset %d!' % opts.dataset)

        self.dataloader = dataloader_fn(opts)
        self.resnet_transform = torchvision.transforms.Normalize(
            mean=torch.tensor(image_utils.BGR_MEAN, dtype=torch.float),
            std=torch.tensor(image_utils.BGR_STD, dtype=torch.float)
        )
def main(_):
    # opts = flags.FLAGS
    # opts = flags.FLAGS


    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed(opts.seed)

    if opts.use_gt_camera:
        opts.num_multipose_az = 1
        opts.num_multipose_el = 1
    opts.num_multipose = opts.num_multipose_az * opts.num_multipose_el
    # assert(uv_Fxtxtx2.max()<=1+1e-12)
    # assert(uv_Fxtxtx2.min()>=1+1e-12)
    # print(1+1e-12)
    # print(-1+1e-12)
    # uv0 = faces_uv[:, 2]
    # uv01 = faces_uv[:, 1] - faces_uv[:, 2]
    # uv02 = faces_uv[:, 0] - faces_uv[:, 2]
    # uv_Fx2xtxt = np.inner(np.dstack([uv01, uv02]), coords_txtx2) + uv0.reshape(faces.shape[0], faces_uv.shape[-1], 1, 1)
    # uv_Fxtxtx2 = np.transpose(uv_Fx2xtxt, (0, 2, 3, 1))
    # exit()
    trainer = ShapeTrainer(opts)
    trainer.init_training()

if __name__ == '__main__':
    opts.name="cub_train_cam8x5_test"
    # opts.flagfile="../configs/cub-train.cfg"
    # opts.num_epoches=21
    opts.cameraPoseDict="cachedir/logs/cub_init_campose8x5/stats/campose_0.npz"

    app.run(main)