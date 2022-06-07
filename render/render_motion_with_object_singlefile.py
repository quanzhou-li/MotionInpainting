import os, sys
import cv2
import numpy as np
import smplx
import argparse

import torch
from torch.utils.data import DataLoader

sys.path.append('.')
sys.path.append('..')

from tools.objectmodel import ObjectModel
from tools.meshviewer import Mesh, MeshViewer, colors
from tools.utils import to_cpu, params2torch
from tools.utils import euler
from tools.utils import rotmat2aa
from tools.utils import CRot2rotmat
from models.inr_gan import *
from tools.cfg_parser import Config
from data.dataloader import LoadData
from firelab.config import Config as FireCFG

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_npz(sequence, allow_pickle=True):
    seq_data = np.load(sequence, allow_pickle=allow_pickle)
    data = {}
    for k in seq_data.files:
        data[k] = seq_data[k]
    return data


def load_torch(dataset_dir='datasets_parsed_motion_imgs', ds_name='train_data', batch_size=8):
    ds = LoadData(dataset_dir, ds_name=ds_name)
    ds = DataLoader(ds, batch_size=batch_size, num_workers=8, shuffle=True, drop_last=True)
    return ds


def smooth_sequence(T, sequence):
    smoothed = sequence.copy()
    smoothed[0] = (smoothed[0]+smoothed[1]) / 2
    for i in range(1, T-1):
        smoothed[i] = (smoothed[i-1]+smoothed[i]+smoothed[i+1]) / 3
    smoothed[T-1] = (smoothed[T-2]+smoothed[T-1]) / 2
    return smoothed


def render_img(cfg):
    config_inr = FireCFG.load(cfg.inr_config)
    inr = INRGenerator(config_inr).to(device)
    inr.load_state_dict(torch.load(cfg.model_path, map_location=torch.device('cpu')))
    inr.eval()

    mv = MeshViewer(offscreen=True)

    # set the camera pose
    camera_pose = np.eye(4)
    camera_pose[:3, :3] = euler([80, 30, 0], 'xzx')
    camera_pose[:3, 3] = np.array([1.3, -2.3, 1.5])
    mv.update_camera_pose(camera_pose)

    data = parse_npz(cfg.data_path)
    batch_size = 1
    data['motion_img'] = torch.FloatTensor(data['motion_img'][:, :].reshape(1, 342, 64))
    bs, height, width = data['motion_img'].shape
    dist = torch.distributions.normal.Normal(
        loc=torch.tensor(np.zeros([batch_size, 512]), requires_grad=False),
        scale=torch.tensor(np.ones([batch_size, 512]), requires_grad=False)
    )
    z_s = dist.rsample().float()
    fframes = data['motion_img'][:, :, 0]
    lframes = data['motion_img'][:, :, -1]
    res = inr.decode(z_s, fframes, lframes, width, height)
    T = width
    fullpose_6D = res['imgs'].view(bs, height, width)[:, :330, :][0].t()  # [n_frames, n_pose_D]
    root = res['imgs'].view(bs, height, width)[:, 330:333, :][0].t()
    if cfg.replace_fl:
        fullpose_6D[0, :] = fframes[0, :330]
        fullpose_6D[-1, :] = lframes[0, :330]
        root[0, :] = fframes[0, 330:333]
        root[-1, :] = lframes[0, 330:333]
    fullpose_rotmat = torch.zeros((T, 55, 3, 3))
    for i in range(T):
        fullpose_rotmat[i] = CRot2rotmat(torch.tensor(fullpose_6D[[i]]))
    fullpose_rotmat = fullpose_rotmat.reshape(T, 1, 55, 9)
    fullpose = rotmat2aa(fullpose_rotmat).reshape(T, 165)

    sbj_parms = {
        'global_orient': fullpose[:, :3].float(),
        'body_pose': fullpose[:, 3:66].float(),
        'jaw_pose': fullpose[:, 66:69].float(),
        'leye_pose': fullpose[:, 69:72].float(),
        'reye_pose': fullpose[:, 72:75].float(),
        'left_hand_pose': fullpose[:, 75:120].float(),
        'right_hand_pose': fullpose[:, 120:165].float(),
        'transl': root.float(),
    }

    LossL2 = torch.nn.MSELoss(reduction='mean')
    loss = 100 * LossL2(data['motion_img'], res['imgs'].view(bs, height, width))
    print(loss)

    sbj_mesh = os.path.join(cfg.tool_meshes, cfg.vtemp)
    sbj_vtemp = np.array(Mesh(filename=sbj_mesh).vertices)

    sbj_m = smplx.create(model_path=cfg.smplx_path,
                         model_type='smplx',
                         gender=cfg.gender,
                         use_pca=False,
                         # num_pca_comps=test_data['n_comps'],
                         v_template=sbj_vtemp,
                         batch_size=T)
    verts_sbj = to_cpu(sbj_m(**sbj_parms).vertices)

    obj_mesh = os.path.join(cfg.tool_meshes, str(data['object_mesh']))
    obj_mesh = Mesh(filename=obj_mesh)
    obj_vtemp = np.array(obj_mesh.vertices)
    obj_m = ObjectModel(v_template=obj_vtemp,
                        batch_size=T)
    obj_transl = res['imgs'].view(bs, height, width)[:, 339:, :][0].t()
    obj_orient_6D = res['imgs'].view(bs, height, width)[:, 333:339, :][0].t()  # [n_frames, n_pose_D]
    if cfg.replace_fl:
        obj_orient_6D[0, :] = fframes[0, 333:339]
        obj_orient_6D[-1, :] = lframes[0, 333:339]
        obj_transl[0, :] = fframes[0, 339:]
        obj_transl[-1, :] = lframes[0, 339:]
    obj_orient_rotmat = CRot2rotmat(obj_orient_6D)
    obj_orient_rotmat = obj_orient_rotmat.reshape(T, 1, 3, 3)
    obj_orient = rotmat2aa(obj_orient_rotmat).reshape(T, 3)

    obj_params = {
        'global_orient': obj_orient.float().cpu().detach().numpy(),
        'transl': obj_transl.float().cpu().detach().numpy()
    }
    obj_parms = params2torch(obj_params)
    verts_obj = to_cpu(obj_m(**obj_parms).vertices)

    suj_id = 's1'
    if not os.path.exists(os.path.join(cfg.renderings, suj_id)):
        os.makedirs(os.path.join(cfg.renderings, suj_id))

    for i in range(T):
        s_mesh = Mesh(vertices=verts_sbj[i], faces=sbj_m.faces, vc=colors['pink'], smooth=True)
        o_mesh = Mesh(vertices=verts_obj[i], faces=obj_mesh.faces, vc=colors['yellow'])

        mv.set_static_meshes([s_mesh, o_mesh])

        color, depth = mv.viewer.render(mv.scene)
        img = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        img_save_path = os.path.join(cfg.renderings, suj_id)
        img_name = 'cup_' + str(i) + '.jpg'
        cv2.imwrite(os.path.join(img_save_path, img_name), img)

    # mv.close_viewer()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Render GNet Poses')
    parser.add_argument('--model-path', required=True, type=str,
                        help='Path to the trained model')
    parser.add_argument('--data-path', default='datasets_parsed_motion_inpaint_64frames/', type=str,
                        help='Path to the data to be tested')
    parser.add_argument('--renderings', default='renderings_motion_imgs', type=str,
                        help='Path to the directory saving the renderings')
    parser.add_argument('--inr-config', default='config/inr-gan.yml', type=str)
    parser.add_argument('--replace-fl', default=False, type=lambda arg: arg.lower() in ['true', '1'],
                        help='If to replace the first and last frames of the predicted results')

    args = parser.parse_args()
    data_path = args.data_path
    model_path = args.model_path
    renderings = args.renderings
    inr_config = args.inr_config
    replace_fl = args.replace_fl

    cfg = {
        'tool_meshes': 'toolMeshes',
        'smplx_path': 'smplx_models',
        'model_path': model_path,
        'inr_config': inr_config,
        'vtemp': 'tools/subject_meshes/male/s1.ply',
        'gender': 'male',
        'data_path': data_path,
        'renderings': renderings,
        'replace_fl': replace_fl,
    }

    cfg = Config(**cfg)

    if not os.path.exists(cfg.renderings):
        os.makedirs(cfg.renderings)

    render_img(cfg)
