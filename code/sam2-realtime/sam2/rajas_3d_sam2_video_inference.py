import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import imageio
import random
from pathlib import Path
from collections import Counter

from sam2.build_sam import build_sam2_camera_predictor
import time
import colorsys
from datetime import datetime
import pickle
import tqdm

import icp_3d

import argparse

parser = argparse.ArgumentParser()

# =========================================
#   The code is adapted from the benchmark.py script. Refer that to view the original version.
# =========================================



# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True



parser.add_argument("-v", "--video_dir_path", required=True, type=str)
# parser.add_argument("-dv", "--depth_array_path", required=True, type=str)
# parser.add_argument("-extparams", "--extrinsicparams", required=True, type=str, default="../../videos/new_recording_0324/extrinsic_params.txt")
parser.add_argument("--out_dir", type=str, default="../../videos/")
parser.add_argument("--model","--model_checkpoint_path", type=str, default="../checkpoints/sam2.1_hiera_tiny.pt")
parser.add_argument("--cfg","--model_config_path", type=str, default="configs/sam2.1/sam2.1_hiera_t_512")
parser.add_argument("--hardcode_prompt", action='store_true', default=False)


args = parser.parse_args()

def generate_fluorescent_color(num = 10):
    """
    Generates a random bright fluorescent color as an RGB tuple.
    """
    colors = []
    
    for _ in range(num):
        # Generate bright colors by using high values for at least one channel
        h,s,l = random.random(), 0.5 + random.random()/2.0, 0.4 + random.random()/5.0
        r,g,b = [int(256*i) for i in colorsys.hls_to_rgb(h,l,s)]
        colors.append((b,g,r))

    return colors

def get_3Dmask_coords(pcd_dict):
    """
    Returns the 3D coordinates of the mask from the PCD dictionary.
    """
    mask3D_points = []
    for label, coord in zip(pcd_dict["group"], pcd_dict["coord"]):
        if label == 1:
            mask3D_points.append(coord)
    return np.array(mask3D_points)


def get_pcd(rgb_map, depth_map,  object_mask, extrinsic_params):
    # intrinsic_path = join(rgb_path, scene_name, 'intrinsics', 'intrinsic_depth.txt')
    # depth_intrinsic = np.loadtxt(intrinsic_path)

    depth_intrinsic = np.array([[892.2883911132812, 0, 643.8308715820312, 0],
                            [0, 892.2883911132812, 376.31494140625, 0],
                            [0, 0, 1, 0]])

    # pose = join(rgb_path, scene_name, 'pose', color_name[0:-4] + '.txt')
    # depth = join(rgb_path, scene_name, 'depth', color_name[0:-4] + '.png')
    # color = join(rgb_path, scene_name, 'color', color_name)

    depth_img = depth_map.copy() #cv2.imread(depth, -1) # read 16bit grayscale image
    valid_depth_map = (depth_img != 0)
    color_image = rgb_map.copy() # cv2.imread(color)
    # color_image = cv2.resize(color_image, (640, 480))

    # save_2dmask_path = join(save_2dmask_path, scene_name)
    # if mask_generator is not None:
    #     group_ids = get_sam(color_image, mask_generator)
    #     if not os.path.exists(save_2dmask_path):
    #         os.makedirs(save_2dmask_path)
    #     img = Image.fromarray(num_to_natural(group_ids).astype(np.int16), mode='I;16')
    #     img.save(join(save_2dmask_path, color_name[0:-4] + '.png'))
    # else:
    #     group_path = join(save_2dmask_path, color_name[0:-4] + '.png')
    #     img = Image.open(group_path)
    #     group_ids = np.array(img, dtype=np.int16)

    color_image = np.reshape(color_image[valid_depth_map], [-1,3])
    group_ids = object_mask[valid_depth_map]
    colors = np.zeros_like(color_image)
    colors[:,0] = color_image[:,2]
    colors[:,1] = color_image[:,1]
    colors[:,2] = color_image[:,0]

    # pose = np.loadtxt(pose)
    
    depth_shift = 1000.0
    x,y = np.meshgrid(np.linspace(0,depth_img.shape[1]-1,depth_img.shape[1]), np.linspace(0,depth_img.shape[0]-1,depth_img.shape[0]))
    uv_depth = np.zeros((depth_img.shape[0], depth_img.shape[1], 3))
    uv_depth[:,:,0] = x
    uv_depth[:,:,1] = y
    uv_depth[:,:,2] = depth_img/depth_shift
    uv_depth = np.reshape(uv_depth, [-1,3])
    uv_depth = uv_depth[np.where(uv_depth[:,2]!=0),:].squeeze()
    
    # intrinsic_inv = np.linalg.inv(depth_intrinsic)
    fx = depth_intrinsic[0,0]
    fy = depth_intrinsic[1,1]
    cx = depth_intrinsic[0,2]
    cy = depth_intrinsic[1,2]
    bx = depth_intrinsic[0,3]
    by = depth_intrinsic[1,3]
    n = uv_depth.shape[0]
    points = np.ones((n,4))
    X = (uv_depth[:,0]-cx)*uv_depth[:,2]/fx + bx
    Y = (uv_depth[:,1]-cy)*uv_depth[:,2]/fy + by
    points[:,0] = X
    points[:,1] = Y
    points[:,2] = uv_depth[:,2]
    points_world = np.dot(points, np.transpose(extrinsic_params))
    # print ("points world:", points_world.shape)

    # print(points_world[1000:1010])

    # binary_3d_mask = np.ones(len(points_world), dtype=np.uint8)  # 1 for object points

    # group_ids = num_to_natural(group_ids)
    save_dict = dict(coord=points_world[:,:3], color=colors, group=group_ids)

    # save_dict = voxelize(save_dict)
    return save_dict

sam2_checkpoint = args.model #"../checkpoints/sam2.1_hiera_tiny.pt"
model_cfg = args.cfg #"configs/sam2.1/sam2.1_hiera_t_512"

predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)

video_path = os.path.join(args.video_dir_path, "color_video.avi")  #"rtsp://127.0.0.1:8554/stream" #"../../videos/randomized_tilt.mp4"
depth_video_path = os.path.join(args.video_dir_path, "raw_depth_frames.npy")
output_path = f"{args.out_dir}/output_{Path(video_path).name}"

if output_path[-3:] != 'mp4':
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d%H%M%S")
    output_path = output_path + f"_{timestamp}_.mp4"


if not os.path.isdir(args.out_dir):
    os.makedirs(args.out_dir)

cap = cv2.VideoCapture(video_path)
# depth_cap = cv2.VideoCapture(depth_video_path)
depth_maps = np.load(depth_video_path)


frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define codec and create VideoWriter object
# out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (frame_width, frame_height))

colors = generate_fluorescent_color(10) # Generate 10 random bright colors

extrinsic_params = np.loadtxt(os.path.join(Path(args.video_dir_path).parent, "extrinsic_params.txt"))

if_init = False
fcount = 0
T_matrices = {}

pbar = tqdm.tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

while True:
    ret, frame = cap.read()
    # d_ret, depth_frame = depth_cap.read()
    if not ret:
        break

    depth_frame = depth_maps[fcount]
    
    fcount += 1
    pbar.update(1)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # depth_frame = cv2.cvtColor(depth_frame, cv2.COLOR_BGR2GRAY)

    width, height = frame.shape[:2][::-1]
    if not if_init:

        predictor.load_first_frame(frame)
        if_init = True

        ann_frame_idx = 0  # the frame index we interact with
        # Let's add a positive click at (x, y) = (210, 350) to get started


        ##! add points, `1` means positive click and `0` means negative click
        # points = np.array([[660, 267]], dtype=np.float32)
        # labels = np.array([1], dtype=np.int32)

        ann_obj_id = [1]  # give a unique id to each object we interact with (it can be any integers)
        points, labels = {1:[]}, {1:[]}
        print("Object ID:", ann_obj_id)

        # Mouse callback function to capture points
        def select_points(event, x, y, flags, params):
            obj_id = ann_obj_id[-1]
            if event == cv2.EVENT_LBUTTONDOWN: # mark positve point
                points[obj_id].append((x, y))
                labels[obj_id].append(1)
                cv2.circle(frame, (x, y), 5, colors[obj_id], -1)
                cv2.imshow("Select Key Points", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                print("Object ID:", obj_id, "\tPositive point:", x,y )
            elif event == cv2.EVENT_MBUTTONDOWN: # mark negative point
                points[obj_id].append((x, y))
                labels[obj_id].append(0)
                cv2.circle(frame, (x, y), 5, colors[obj_id], 2)
                cv2.imshow("Select Key Points", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                print("Object ID:", obj_id, "\tNegative point:", x,y )
            elif event == cv2.EVENT_MOUSEWHEEL and len(points[ann_obj_id[-1]]) > 0: # Create new object id
                ann_obj_id.append(obj_id+1)
                points[ann_obj_id[-1]], labels[ann_obj_id[-1]] = [], []
                print("Object ID changed to:", ann_obj_id[-1])#, "---- Color :", colors[ann_obj_id[-1]])
        
        # Display frame for point selection
        if args.hardcode_prompt == False:
            cv2.imshow("Select Key Points", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cv2.setMouseCallback("Select Key Points", select_points, ann_obj_id)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("No GUI mode : Points are hardcoded")
            points = {1:[(369,151), (399,138), (410,170), (379,186), (391,162)]}
            labels = {1:[1, 1, 1, 1, 1]}

        # print(points, labels)
        
        assert len(ann_obj_id) <= 3, "Object limit is set to 3 as the visualization code supports only 3 colors. Change it to track more objects :)"

        for i in ann_obj_id:
            if len(points[i]) > 0:
                _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
                    frame_idx=ann_frame_idx, obj_id=i, points=points[i], labels=labels[i]
                )
        
        T_matrices = {i:{} for i in ann_obj_id}
        prev_frame_mask = [np.zeros((1,3), dtype=np.uint8)]*len(ann_obj_id)




        ## ! add bbox
        # bbox = np.array([[600, 214], [765, 286]], dtype=np.float32)
        # _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
        #     frame_idx=ann_frame_idx, obj_id=ann_obj_id, bbox=bbox
        # )

        ##! add mask
        # mask_img_path="../notebooks/masks/aquarium/aquarium_mask.png"
        # mask = cv2.imread(mask_img_path, cv2.IMREAD_GRAYSCALE)
        # mask = mask / 255

        # _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
        #     frame_idx=ann_frame_idx, obj_id=ann_obj_id, mask=mask
        # )

    else:
        out_obj_ids, out_mask_logits = predictor.track(frame)

        # all_mask = np.zeros((height, width, 1), dtype=np.uint8)
        # print(all_mask.shape)
        for i in range(0, len(out_obj_ids)):
            out_mask = (out_mask_logits[i] > 0.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)

            pcd = get_pcd(rgb_map=frame, depth_map=depth_frame, object_mask=out_mask, extrinsic_params=extrinsic_params)
            mask_3d = get_3Dmask_coords(pcd)
            # print(mask_3d.shape)
            
            if fcount != 1:
                M = icp_3d.icp_3d(source_mask=prev_frame_mask[i], target_mask=mask_3d, mask_coords=True)
                T_matrices[out_obj_ids[i]][fcount] = M
            
            prev_frame_mask[i] = mask_3d

            # out_mask = cv2.cvtColor(out_mask, cv2.COLOR_GRAY2RGB)
            # out_mask[:, :, i] = np.clip(out_mask[:, :, i] * 255, 0, 255).astype(np.uint8)
            # frame = cv2.addWeighted(frame, 1, out_mask, 0.5, 0)

            out_mask = np.clip(out_mask*255, 0, 255)
            cv2.imwrite(f"../../videos/frames/{out_obj_ids[i]}_{str(fcount).zfill(5)}.jpg", out_mask)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.imshow("frame", frame)
    # out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

pbar.close()

with open(f"{args.out_dir}/{Path(args.video_path).stem}.pkl", "wb") as f:
    pickle.dump(T_matrices, f)
print(f"Video saved at {output_path}")
cap.release()
# out.release()
# gif = imageio.mimsave("./result.gif", frame_list, "GIF", duration=0.00085)