# segment-anything-2 real-time
Run Segment Anything Model 2 on a **live video stream**

## News
- 13/12/2024 : Update to sam2.1
- 20/08/2024 : Fix management of ```non_cond_frame_outputs``` for better performance and add bbox prompt

## Demos
<div align=center>
<p align="center">
<img src="./assets/blackswan.gif" width="880">
</p>

</div>



## Getting Started

### Installation

```bash
pip install -e .
```
### Download Checkpoint

Then, we need to download a model checkpoint.

```bash
cd checkpoints
./download_ckpts.sh
```

### Inference
```
export DISPLAY=:1
python3.10 zeyu_sam2_video_inference.py -v /home/zeyu/Downloads/record_gray_cylinder_0324/record_gray_cylinder/color_video.avi
python3.10 zeyu_sam2_video_inference.py --input_type video --video_path /home/zeyu/Downloads/record_gray_cylinder_0324/record_gray_cylinder/color_video.avi

python3.10 zeyu_sam2_video_inference.py --list_cameras
python3.10 zeyu_sam2_video_inference.py --input_type realsense --camera_index 0
python3.10 zeyu_sam2_video_inference.py --input_type realsense --camera_serial 217222067304
python3.10 zeyu_sam2_video_risk_inference.py --input_type realsense --camera_serial 217222067304 --risk_model_path /home/zeyu/PHD_LAB/Material_handling_2024/zeyu-failure-prediction/code/simple_model/best_model_ResNet18.pth
```

### Data Collection
```
export DISPLAY=:1

python3.10 zeyu_sam2_data_collection.py --input_type realsense --camera_serial 217222067304
```

Then SAM-2-online can be used in a few lines as follows for image and video and **camera** prediction.

### Camera prediction

```python
import torch
from sam2.build_sam import build_sam2_camera_predictor

sam2_checkpoint = "../checkpoints/sam2.1_hiera_small.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
predictor = build_sam2_camera_predictor(model_cfg, checkpoint)

cap = cv2.VideoCapture(<your video or camera >)

if_init = False

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        width, height = frame.shape[:2][::-1]

        if not if_init:
            predictor.load_first_frame(frame)
            if_init = True
            _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(<your promot >)

        else:
            out_obj_ids, out_mask_logits = predictor.track(frame)
            ...
```

### With model compilation

You can use the `vos_inference` argument in the `build_sam2_camera_predictor` function to enable model compilation. The inference may be slow for the first few execution as the model gets warmed up, but should result in significant inference speed improvement. 

We provide the modified config file `sam2/configs/sam2.1/sam2.1_hiera_t_512.yaml`, with the modifications necessary to run SAM2 at a 512x512 resolution. Notably the parameters that need to be changed are highlighted in the config file at lines 24, 43, 54 and 89.

We provide the file `sam2/benchmark.py` to test the speed gain from using the model compilation.

## References:

- SAM2 Repository: https://github.com/facebookresearch/sam2
