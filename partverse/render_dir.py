import os
import json
import argparse
import numpy as np
from utils import sphere_hammersley_sequence, rot_360_sequence
from glob import glob
from tqdm import tqdm


BLENDER_LINK = 'https://download.blender.org/release/Blender4.4/blender-4.4.0-linux-x64.tar.xz'
BLENDER_INSTALLATION_PATH = '~'
BLENDER_PATH = f'{BLENDER_INSTALLATION_PATH}/blender-4.4.0-linux-x64/blender'

def _install_blender():
    if not os.path.exists(BLENDER_PATH):
        os.system('sudo apt-get update')
        os.system('sudo apt-get install -y libxrender1 libxi6 libxkbcommon-x11-0 libsm6')
        os.system(f'wget {BLENDER_LINK} -P {BLENDER_INSTALLATION_PATH}')
        os.system(f'tar -xvf {BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64.tar.xz -C {BLENDER_INSTALLATION_PATH}')


def _render(file_path, output_folder, num_views, use_random_views, elevation=30):
    if os.path.exists(os.path.join(output_folder, 'transforms.json')):
        return True
    
    # Build camera {yaw, pitch, radius, fov}
    yaws = []
    pitchs = []
    offset = (np.random.rand(), np.random.rand())
    for i in range(num_views):
        if not use_random_views:
            y, p = rot_360_sequence(i, num_views, theta=elevation)
        else:
            y, p = sphere_hammersley_sequence(i, num_views, offset)
        yaws.append(y)
        pitchs.append(p)
    radius = [2] * num_views
    fov = [40 / 180 * np.pi] * num_views
    views = [{'yaw': y, 'pitch': p, 'radius': r, 'fov': f} for y, p, r, f in zip(yaws, pitchs, radius, fov)]
    args = [
        BLENDER_PATH, '-b', '-P', os.path.join(os.path.dirname(__file__), 'blender_script', 'render.py'),
        '--',
        '--object', os.path.expanduser(file_path),
        '--resolution', '512',
        '--output_folder', output_folder,
        '--engine', 'CYCLES', # 'BLENDER_EEVEE_NEXT' for blender-4
        # '--save_depth',
        # '--save_mesh',
    ]
    if file_path.endswith('.blend'):
        args.insert(1, file_path)
    
    cmd = ' '.join(args)
    cmd += f" --views '{json.dumps(views)}'"
    # cmd += ' > ~/tmp.log'
    os.system(cmd)
    
    if os.path.exists(os.path.join(output_folder, 'transforms.json')):
        return True
    else:
        return False

def render_dir(root_dir, ins_list, out_dir, num_views, use_random_views=False, elevation=30):
    _install_blender()
    os.makedirs(out_dir, exist_ok=True)
    
    fail_list = []
    empty_list = []
    for ins in tqdm(ins_list, total=len(ins_list), desc='Rendering dir'):
        file_path = os.path.join(root_dir, ins + '.glb')
        if not os.path.exists(file_path):
            empty_list.append(ins)
            continue
        save_ins_dir = os.path.join(out_dir, ins)
        os.makedirs(save_ins_dir, exist_ok=True)
        success = True
        success = success and _render(
            file_path, save_ins_dir, num_views, use_random_views=use_random_views, elevation=elevation)
        if not success:
            fail_list.append(ins)
    
    print(f"Num Fail: {len(fail_list)}")
    os.makedirs('debug', exist_ok=True)
    with open('debug/fail_ins_list.txt', 'a') as f:
        for item in fail_list:
            f.write(f"{item}\n")

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--id_range', type=str, default='0-999999', help='Start and end index of the range of instances to render')
    arg_parser.add_argument('--data_root', type=str, required=True, help='Root directory where storage *.glb files')
    arg_parser.add_argument('--out_dir', type=str, required=True, help='Output directory for rendered images')
    arg_parser.add_argument('--ins_file', type=str, default=None, help='TXT file containing the list of instance names to render')
    arg_parser.add_argument('--num_views', type=int, default=8, help='Number of views to render for each part')
    arg_parser.add_argument('--use_random_views', action='store_true', help='Use spherical hammersley sequence rendering instead of 360-degree')
    arg_parser.add_argument('--elevation', type=int, default=30, help='Render elevation angle in degrees, used only when not use_random_views')
    args = arg_parser.parse_args()
    
    
    id_range = args.id_range.split('-')
    if len(id_range) != 2:
        raise ValueError("id_range should be in the format 'start-end', e.g., '0-1000'")
    id_range_l, id_range_h = int(id_range[0]), int(id_range[1])
    root_dir = args.data_root
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    num_views = args.num_views
    use_random_views = args.use_random_views
    ins_file = args.ins_file
    if ins_file is None:
        ins_list = sorted(os.listdir(root_dir))
    else:
        with open(ins_file, 'r') as f:
            ins_list = f.readlines()
        ins_list = sorted([ins.strip() for ins in ins_list])
    print(f"Total Num Ins: {len(ins_list)}")
    ins_list = ins_list[id_range_l:id_range_h]
    render_dir(root_dir, ins_list, out_dir, num_views=num_views, use_random_views=use_random_views, elevation=args.elevation)
