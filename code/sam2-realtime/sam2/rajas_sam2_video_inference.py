import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import imageio
import random
from pathlib import Path

from sam2.build_sam import build_sam2_camera_predictor
import time
import colorsys
from datetime import datetime
import pickle

import icp_2d

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



parser.add_argument("-v", "--video_path", required=True, type=str)
parser.add_argument("--out_dir", type=str, default="../../videos/")
parser.add_argument("--model","--model_checkpoint_path", type=str, default="../checkpoints/sam2.1_hiera_tiny.pt")
parser.add_argument("--cfg","--model_config_path", type=str, default="configs/sam2.1/sam2.1_hiera_t_512")

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

sam2_checkpoint = args.model #"../checkpoints/sam2.1_hiera_tiny.pt"
model_cfg = args.cfg #"configs/sam2.1/sam2.1_hiera_t_512"

predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)

video_path = args.video_path#"rtsp://127.0.0.1:8554/stream" #"../../videos/randomized_tilt.mp4"
output_path = f"{args.out_dir}/output_{Path(video_path).name}"

if output_path[-3:] != 'mp4':
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d%H%M%S")
    output_path = output_path + f"_{timestamp}_.mp4"


if not os.path.isdir(args.out_dir):
    os.makedirs(args.out_dir)

cap = cv2.VideoCapture(video_path)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define codec and create VideoWriter object
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (frame_width, frame_height))

colors = generate_fluorescent_color(10) # Generate 10 random bright colors

if_init = False
fcount = 0
T_matrices = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    fcount += 1

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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
        cv2.imshow("Select Key Points", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cv2.setMouseCallback("Select Key Points", select_points, ann_obj_id)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # print(points, labels)
        
        assert len(ann_obj_id) <= 3, "Object limit is set to 3 as the visualization code supports only 3 colors. Change it to track more objects :)"

        for i in ann_obj_id:
            if len(points[i]) > 0:
                _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
                    frame_idx=ann_frame_idx, obj_id=i, points=points[i], labels=labels[i]
                )
        
        T_matrices = {i:{} for i in ann_obj_id}
        prev_frame_mask = [np.zeros((frame_height, frame_width, 1), dtype=np.uint8)]*len(ann_obj_id)




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

        # print (out_obj_ids)

        all_mask = np.zeros((height, width, 1), dtype=np.uint8)
        # print(all_mask.shape)
        for i in range(0, len(out_obj_ids)):
            out_mask = (out_mask_logits[i] > 0.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)

            if fcount != 1:
                M = icp_2d.icp(source_mask=prev_frame_mask[i], target_mask=out_mask)
                T_matrices[out_obj_ids[i]][fcount] = M
            
            prev_frame_mask[i] = out_mask

            out_mask = cv2.cvtColor(out_mask, cv2.COLOR_GRAY2RGB)
            out_mask[:, :, i] = np.clip(out_mask[:, :, i] * 255, 0, 255).astype(np.uint8)
            frame = cv2.addWeighted(frame, 1, out_mask, 0.5, 0)

            # cv2.imwrite(f"../../videos/frames/{out_obj_ids[i]}_{str(fcount).zfill(5)}.jpg", out_mask)

    # print("Frame: ", fcount, end='\r')
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.imshow("frame", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

with open(f"{args.out_dir}/{Path(args.video_path).stem}.pkl", "wb") as f:
    pickle.dump(T_matrices, f)
print(f"Video saved at {output_path}")
cap.release()
out.release()
# gif = imageio.mimsave("./result.gif", frame_list, "GIF", duration=0.00085)