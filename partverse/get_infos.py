import os, glob, json
import sys
sys.path.append('.')
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
import argparse

import trimesh
import numpy as np
import torch
import torchvision

import flex_render
import kaolin as kal

anno_infos_root='/nvme/lihe/dataset/gobjaverse/tmp/parts'
filter_ins_file = 'debug/partverse_valid_ins2.txt'
with open(filter_ins_file, 'r') as f:
    filter_ins = f.readlines()
filter_ins = [d.strip() for d in filter_ins]

def normalize_mesh(mesh: trimesh.Trimesh, rescale=1):
    bbox = mesh.bounding_box.bounds
    loc = (bbox[0] + bbox[1]) / 2
    scale = (bbox[1] - bbox[0]).max()
    mesh.apply_translation(-loc)
    mesh.apply_scale(rescale / scale)
    return mesh, loc, scale, bbox

def normalize_mesh_to_05(mesh: trimesh.Trimesh):
    bbox = mesh.bounding_box.bounds
    loc = (bbox[0] + bbox[1]) / 2
    scale = (bbox[1] - bbox[0]).max()
    mesh.apply_scale(1 / 2.)
    return mesh, loc, scale, bbox        

def post_process_info1(anno_infos_root):
    uids = os.listdir(anno_infos_root)

    valid_uids = []
    for uid in tqdm(uids, total=len(uids)):
        ins_dir = os.path.join(anno_infos_root, uid)
        glb_path = os.path.join(ins_dir, uid + '_segmented.glb')
        f2l_path = os.path.join(ins_dir, uid + '_face2label.json')
        info_path = os.path.join(ins_dir, uid + '_info.json')
        if not os.path.exists(glb_path) or not os.path.exists(f2l_path) or os.path.exists(info_path):
            continue
            
        try:
            with open(f2l_path, 'r') as f:
                f2l_dict = json.load(f)
        except:
            print('Error json: ', f2l_path)
            continue
        l2f_dict = {}
        for k, v in f2l_dict.items():
            k, v = int(k), int(v)
            if v not in l2f_dict:
                l2f_dict[v] = [k]
            else:
                l2f_dict[v].append(k)
        
        mesh = trimesh.load(glb_path, force='mesh', process=False)

        bboxes = []
        part_faceid = []
        part_face_label = []
        for l, f in l2f_dict.items():
            part = mesh.submesh([f], append=True)
            bbox = part.bounding_box.bounds
            bboxes.append(bbox)
            part_faceid.append(f)
            part_face_label.append(l)
        
        if len(bboxes) == 0:
            print('Error box: ', glb_path)
            continue
        bboxes = np.stack(bboxes) # [np, 2, 3]
        dxyz = bboxes[:, 1] - bboxes[:, 0]
        weights = dxyz[:, 0] * dxyz[:, 1] + dxyz[:, 1] * dxyz[:, 2] + dxyz[:, 2] * dxyz[:, 0]
        sort_idx = np.argsort(weights)[::-1]
        weights = weights[sort_idx]
        bboxes = bboxes[sort_idx]
        part_faceid = [part_faceid[i] for i in sort_idx]
        part_face_label = [part_face_label[i] for i in sort_idx]
        info = {'bboxes': bboxes.tolist(), 'ordered_faceid': part_faceid, 'weights': weights.tolist(), 'ordered_face_label': part_face_label}
        valid_uids.append(uid)
        with open(info_path, 'w') as f:
            json.dump(info, f)

    return valid_uids

def post_process_info2(valid_uids, save_path='data/partverse_global_infos.json'):
    weights_statistic = []
    for uid in tqdm(valid_uids, total=len(valid_uids)):
        ins_dir = os.path.join(anno_infos_root, uid)
        info_path = os.path.join(ins_dir, uid + '_info.json')
        with open(info_path, 'r') as f:
            info = json.load(f)
        weights = np.array(info['weights'])
        weights_statistic.append(weights)
    weights_statistic = np.concatenate(weights_statistic)
    weights_statistic = np.sort(weights_statistic)[::-1]
    bin_resolution = 10 # percentage
    percent_ = 0
    weight_bound = []
    for i in range(1, 100 // bin_resolution):
        percent_ += bin_resolution
        weight_bound.append(weights_statistic[weights_statistic.shape[0] // (100 // bin_resolution) * i])
    print('priority weight bound: ', weight_bound)
    for uid in tqdm(valid_uids, total=len(valid_uids)):
        ins_dir = os.path.join(anno_infos_root, uid)
        info_path = os.path.join(ins_dir, uid + '_info.json')
        with open(info_path, 'r') as f:
            info = json.load(f)
        bboxes = np.array(info['bboxes']) # np, 2, 3
        centers = (bboxes[:, 0] + bboxes[:, 1]) / 2
        weights = np.array(info['weights']) # np
        ordered_part_level = np.digitize(weights, weight_bound) # np
        same_level_order = np.arange(len(weights)) # np
        for level in np.unique(ordered_part_level):
            parts_same_level_mask = ordered_part_level == level
            sort_idx = np.argsort(centers[parts_same_level_mask, 1]) # index=1 is z axis for objaverse
            same_level_order[parts_same_level_mask] = same_level_order[parts_same_level_mask][sort_idx]
        info['ordered_part_level'] = ordered_part_level[same_level_order].tolist() 
        info['weights'] = weights[same_level_order].tolist()
        info['bboxes'] = [info['bboxes'][i] for i in same_level_order]
        info['ordered_faceid'] = [info['ordered_faceid'][i] for i in same_level_order]
        info['ordered_face_label'] = [info['ordered_face_label'][i] for i in same_level_order]
        with open(info_path, 'w') as f:
            json.dump(info, f)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump({'weight_bound': weight_bound, 'uids': valid_uids}, f, indent=4)
    with open(os.path.join(os.path.dirname(save_path), 'valid_ins.txt'), 'w') as f:
        f.write('\n'.join(valid_uids))

@torch.no_grad()
def post_process_render_mask(anno_infos_root, textured_part_glbs_root, valid_uids, max_visible_info_save_path):
    cameras = flex_render.get_rotate_camera_batch(
        8, iter_res=[512, 512], cam_radius=1.59, device='cuda', rot_yx_90=True, azimuth=[-np.pi/2, np.pi/2*3], # match for blender
    )
    
    infos = {}
    unvisible_ins = []
    unvalid_ins = []
    for ins in tqdm(valid_uids, total=len(valid_uids)):
        part_ins_dir = os.path.join(textured_part_glbs_root, ins)
        glb_path = os.path.join(anno_infos_root, ins, ins + '_segmented.glb')
        info_path = os.path.join(anno_infos_root, ins, ins + '_info.json')

        with open(info_path, 'r') as f:
            info = json.load(f)
        mesh = trimesh.load(glb_path, process=False, force='mesh')
        mesh, loc, scale, bbox = normalize_mesh_to_05(mesh)
        mesh_v, mesh_f = torch.as_tensor(np.array(mesh.vertices, dtype=np.float32)).cuda(), torch.as_tensor(np.array(mesh.faces)).cuda()

        part_ids = os.listdir(part_ins_dir)
        part_ids = [int(i[:-4]) for i in part_ids]
        if len(part_ids) == 0:
            continue
        num_parts = len(part_ids)
        ordered_part_level = np.array(info['ordered_part_level'])
        ins_num_parts = len(ordered_part_level)
        # assert ins_num_parts == num_parts, f'{ins} {ins_num_parts} {num_parts}'
        if ins_num_parts != num_parts:
            print(f'Error part number for {ins}: {ins_num_parts} != {num_parts}')
            unvalid_ins.append(ins)
            continue
        part_id_mapping = info.get('ordered_face_label', None) # List[2, 5, 1, 0, ...]

        mesh_kal = kal.rep.SurfaceMesh(mesh_v, mesh_f)
        render_targets = flex_render.render_mesh(mesh_kal, cameras, [512, 512], return_types=['face_id', 'depth']) # dict (v,h,w,c)
        # debug vis
        # depth = render_targets['depth'] # (v,h,w,4)
        # torchvision.utils.save_image(depth.float().permute(0, 3, 1, 2)[:1], f'debug/{ins}_depth.png')
        # exit()
        face_id = render_targets['face_id'].long().squeeze(-1) # (v,h,w)
        mesh_label = torch.zeros((mesh_f.shape[0],), dtype=int).cuda()
        for cur_part_rank in range(num_parts):
            mesh_label[info['ordered_faceid'][cur_part_rank]] = cur_part_rank + 1
        mesh_label_ = mesh_label + 1
        mesh_label_ = torch.cat((torch.zeros_like(mesh_label[:1]), mesh_label_), dim=0) # pad 0 on index 0 for background(face_id=0)
        render_labels = mesh_label_[face_id] # (v,h,w)

        ins_info = {}
        for cur_part_rank in range(num_parts):
            cur_part_mesh_unnorm = mesh.submesh([info['ordered_faceid'][cur_part_rank]], append=True)
            part_mesh_v = torch.as_tensor(np.array(cur_part_mesh_unnorm.vertices, dtype=np.float32)).cuda()
            part_mesh_f = torch.as_tensor(np.array(cur_part_mesh_unnorm.faces)).cuda()
            org_part_id = part_id_mapping[cur_part_rank]
            
            mesh_kal = kal.rep.SurfaceMesh(part_mesh_v, part_mesh_f)
            render_targets = flex_render.render_mesh(mesh_kal, cameras, [512, 512], return_types=['mask']) # dict (v,h,w,1)
            mask = render_labels == (cur_part_rank + 2)
            # torchvision.utils.save_image(mask.float().unsqueeze(1), f'debug/{ins}_{org_part_id}.png')
            mask_sum = mask.sum(dim=[1,2]) # (v,)
            max_visible_view_id = mask_sum.argmax().item()
            if mask_sum.max() == 0:
                unvisible_ins.append(ins + f'_{org_part_id}')
            indices = torch.nonzero(mask[max_visible_view_id], as_tuple=False)
            if indices.numel() == 0:
                bbox2d_max_visible = (-1, -1, -1, -1)
            else:
                y_coords = indices[:, 0]
                x_coords = indices[:, 1]
                x_min = x_coords.min().item()
                x_max = x_coords.max().item()
                y_min = y_coords.min().item()
                y_max = y_coords.max().item()
                bbox2d_max_visible = (x_min, y_min, x_max, y_max)
            ins_info[org_part_id] = {'max_visible_view_id': max_visible_view_id, 'bbox2d_max_visible': bbox2d_max_visible}
        infos[ins] = ins_info
    
    os.makedirs(os.path.dirname(max_visible_info_save_path), exist_ok=True)
    with open(max_visible_info_save_path, 'w') as f:
        json.dump(infos, f, indent=4)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data_root', type=str, required=True, help='Root directory of the dataset')
    arg_parser.add_argument('--global_info_save_path', required=True, type=str, help='Path to save the global info')
    arg_parser.add_argument('--max_visible_info_save_path', required=True, type=str, help='Path to save the max visible info')
    args = arg_parser.parse_args()
    anno_infos_root = os.path.join(args.data_root, 'anno_infos')
    textured_part_glbs_root = os.path.join(args.data_root, 'textured_part_glbs')

    valid_uids = post_process_info1(anno_infos_root=anno_infos_root)
    post_process_info2(valid_uids, save_path=args.global_info_save_path)
    post_process_render_mask(
        anno_infos_root, textured_part_glbs_root, valid_uids=valid_uids, max_visible_info_save_path=args.max_visible_info_save_path)