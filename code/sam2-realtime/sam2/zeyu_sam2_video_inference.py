import os
import torch
import numpy as np
import cv2
import random
from pathlib import Path

from sam2.build_sam import build_sam2_camera_predictor
import colorsys
from datetime import datetime

# Now import using the path from the project root
from simulation.demo.common_functions import (process_consecutive_frames, extract_points_from_mask, extract_transform_features)

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
parser.add_argument("--out_dir", type=str, default="../videos/")
parser.add_argument("--model","--model_checkpoint_path", type=str, default="../checkpoints/sam2.1_hiera_tiny.pt")
parser.add_argument("--cfg","--model_config_path", type=str, default="configs/sam2.1/sam2.1_hiera_t_512")

args = parser.parse_args()

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Current device: {torch.cuda.current_device()}")
print(f"Device name: {torch.cuda.get_device_name()}")


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
# T_matrices = {}

# zeyu: need to clean this when run long
top_camera_object_contours = []
top_camera_panel_contours = []
window = 30

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

        # After creating the predictor, add a print statement like:
        print(f"Predictor device: {predictor.device}")

        ##! add points, `1` means positive click and `0` means negative click
        # points = np.array([[660, 267]], dtype=np.float32)
        # labels = np.array([1], dtype=np.int32)

        ann_obj_id = [1]  # give a unique id to each object we interact with (it can be any integers)
        points, labels = {1:[]}, {1:[]}
        print("Object ID:", ann_obj_id)

        # Mouse callback function to capture points
        def select_points(event, x, y, flags, params):
            obj_id = ann_obj_id[-1]
            if event == cv2.EVENT_LBUTTONDOWN:  # mark positive point
                points[obj_id].append((x, y))
                labels[obj_id].append(1)
                cv2.circle(frame, (x, y), 5, colors[obj_id], -1)
                cv2.imshow("Select Key Points", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                print("Object ID:", obj_id, "\tPositive point:", x, y)
            elif event == cv2.EVENT_MBUTTONDOWN:  # mark negative point
                points[obj_id].append((x, y))
                labels[obj_id].append(0)
                cv2.circle(frame, (x, y), 5, colors[obj_id], 2)
                cv2.imshow("Select Key Points", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                print("Object ID:", obj_id, "\tNegative point:", x, y)
            # Remove the RBUTTONDOWN event here since we'll handle it with keyboard

        # Then modify your point selection part to handle keyboard events
        cv2.imshow("Select Key Points", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cv2.setMouseCallback("Select Key Points", select_points, ann_obj_id)

        # Wait for user input in a loop
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            # Check for Space key (ASCII code 13)
            if key == 32 and len(points[ann_obj_id[-1]]) > 0:
                obj_id = ann_obj_id[-1]
                ann_obj_id.append(obj_id+1)
                points[ann_obj_id[-1]], labels[ann_obj_id[-1]] = [], []
                print("Object ID changed to:", ann_obj_id[-1])
            
            # Check for Enter key to exit the selection mode
            elif key == 13:  # Enter key
                break

        cv2.destroyAllWindows()

        # print(points, labels)
        
        assert len(ann_obj_id) <= 3, "Object limit is set to 3 as the visualization code supports only 3 colors. Change it to track more objects :)"

        for i in ann_obj_id:
            if len(points[i]) > 0:
                _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
                    frame_idx=ann_frame_idx, obj_id=i, points=points[i], labels=labels[i]
                )
        
        # T_matrices = {i:{} for i in ann_obj_id}
        # prev_frame_mask = [np.zeros((frame_height, frame_width, 1), dtype=np.uint8)]*len(ann_obj_id)

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
        
        # After creating the predictor, add a print statement like:
        print(f"Predictor device: {predictor.device}")

        print ("out_obj_ids", out_obj_ids)
        # import sys
        # sys.exit()

        all_mask = np.zeros((height, width, 1), dtype=np.uint8)

            # cv2.imwrite(f"../../videos/frames/{out_obj_ids[i]}_{str(fcount).zfill(5)}.jpg", out_mask)

        top_camera_object_out_mask = (out_mask_logits[0] > 0.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        top_camera_panel_out_mask = (out_mask_logits[1] > 0.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)

        top_camera_object_contours.append(top_camera_object_out_mask)
        top_camera_panel_contours.append(top_camera_panel_out_mask)
        # print("len(top_camera_object_contours)", len(top_camera_object_contours))
        # print("len(top_camera_panel_contours)", len(top_camera_panel_contours))

        if fcount > window:
            top_camera_object_pre_points = extract_points_from_mask(top_camera_object_contours[-window])
            top_camera_object_current_points = extract_points_from_mask(top_camera_object_out_mask)
            top_camera_panel_pre_points = extract_points_from_mask(top_camera_panel_contours[-window])
            top_camera_panel_current_points = extract_points_from_mask(top_camera_panel_out_mask)
            if len(top_camera_object_pre_points) > 0 and len(top_camera_object_current_points) > 0 and len(top_camera_panel_pre_points) > 0 and len(top_camera_panel_current_points) > 0:
                top_camera_object_pre_contours = [top_camera_object_pre_points.reshape(-1, 1, 2).astype(np.int32)]
                top_camera_object_current_contours = [top_camera_object_current_points.reshape(-1, 1, 2).astype(np.int32)]
                top_camera_panel_pre_contours = [top_camera_panel_pre_points.reshape(-1, 1, 2).astype(np.int32)]
                top_camera_panel_current_contours = [top_camera_panel_current_points.reshape(-1, 1, 2).astype(np.int32)]
                # zeyu: to do no window calculated in
                top_camera_object_transforms = process_consecutive_frames(top_camera_object_pre_contours, top_camera_object_current_contours)
                top_camera_object_features = extract_transform_features(top_camera_object_transforms)
                top_camera_panel_transforms = process_consecutive_frames(top_camera_panel_pre_contours, top_camera_panel_current_contours)
                top_camera_panel_features = extract_transform_features(top_camera_panel_transforms)
                print("object_features", top_camera_object_features)
                print("panel_features", top_camera_panel_features)
            else:
                print("Not enough points to calculate transforms")
        
        top_camera_object_out_mask = cv2.cvtColor(top_camera_object_out_mask, cv2.COLOR_GRAY2RGB)
        top_camera_object_out_mask[:, :, 0] = np.clip(top_camera_object_out_mask[:, :, 0] * 255, 0, 255).astype(np.uint8)
        frame = cv2.addWeighted(frame, 1, top_camera_object_out_mask, 0.5, 0)

        top_camera_panel_out_mask = cv2.cvtColor(top_camera_panel_out_mask, cv2.COLOR_GRAY2RGB)
        top_camera_panel_out_mask[:, :, 1] = np.clip(top_camera_panel_out_mask[:, :, 1] * 255, 0, 255).astype(np.uint8)
        frame = cv2.addWeighted(frame, 1, top_camera_panel_out_mask, 0.5, 0)

    # print("Frame: ", fcount, end='\r')
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.imshow("frame", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# with open(f"{args.out_dir}/{Path(args.video_path).stem}.pkl", "wb") as f:
    # pickle.dump(T_matrices, f)
print(f"Video saved at {output_path}")
cap.release()
out.release()
# gif = imageio.mimsave("./result.gif", frame_list, "GIF", duration=0.00085)