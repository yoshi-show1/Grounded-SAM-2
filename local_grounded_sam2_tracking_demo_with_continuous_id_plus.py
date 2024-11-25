import os
import cv2
import torch
import numpy as np
import supervision as sv
from torchvision.ops import box_convert
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
# from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images
from utils.common_utils import CommonUtils
from utils.mask_dictionary_model import MaskDictionaryModel, ObjectInfo
import json
import copy

"""
Hyperparam for Ground and Tracking
"""
GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
# GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth"
GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/original_groundingdino_swint_ogc.pth"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
VIDEO_PATH = "./assets/vlog.mp4"
# VIDEO_PATH = "./assets/color_20240826_132239_0217.jpg"
# TEXT_PROMPT = "dish. food. tray."
TEXT_PROMPT = "dish. tray."
OUTPUT_PATH = "outputs_tray_after"
OUTPUT_VIDEO_PATH = "dish_tracking.mp4"
# OUTPUT_VIDEO_PATH = "./test.jpg"
SOURCE_VIDEO_FRAME_DIR = "./custom_video_frames"
SAVE_TRACKING_RESULTS_DIR = "./tracking_results"
PROMPT_TYPE_FOR_VIDEO = "box" # choose from ["point", "box", "mask"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

"""
Step 1: Environment settings and model initialization for Grounding DINO and SAM 2
"""
# # use bfloat16 for the entire notebook
# torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
# use float32 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.float32).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# build grounding dino model from local path
grounding_model = load_model(
    model_config_path=GROUNDING_DINO_CONFIG, 
    model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
    device=DEVICE
)

# init sam image predictor and video predictor model
sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device", device)

video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
sam2_image_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
image_predictor = SAM2ImagePredictor(sam2_image_model)

"""
Custom video input directly using video files
"""
video_info = sv.VideoInfo.from_video_path(VIDEO_PATH)  # get video info
print(video_info)
frame_generator = sv.get_video_frames_generator(VIDEO_PATH, stride=1, start=0, end=None)

# saving video to frames
source_frames = Path(SOURCE_VIDEO_FRAME_DIR)
source_frames.mkdir(parents=True, exist_ok=True)

with sv.ImageSink(
    target_dir_path=source_frames, 
    overwrite=True, 
    image_name_pattern="{:05d}.jpg"
) as sink:
    for frame in tqdm(frame_generator, desc="Saving Video Frames"):
        sink.save_image(frame)

# 'output_dir' is the directory to save the annotated frames
output_dir = OUTPUT_PATH
# 'output_video_path' is the path to save the final video
# output_video_path = "./outputs/output.mp4"
output_video_path = os.path.join(output_dir,OUTPUT_VIDEO_PATH)
# create the output directory
mask_data_dir = os.path.join(output_dir, "mask_data")
json_data_dir = os.path.join(output_dir, "json_data")
result_dir = os.path.join(output_dir, "result")
CommonUtils.creat_dirs(mask_data_dir)
CommonUtils.creat_dirs(json_data_dir)

# scan all the JPEG frame names in this directory
frame_names = [
    p for p in os.listdir(SOURCE_VIDEO_FRAME_DIR)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

# init video predictor state
inference_state = video_predictor.init_state(video_path=SOURCE_VIDEO_FRAME_DIR)
step = 10 # the step to sample frames for Grounding DINO predictor

sam2_masks = MaskDictionaryModel()
PROMPT_TYPE_FOR_VIDEO = "mask" # box, mask or point
objects_count = 0
frame_object_count = {}
"""
Step 2: Prompt Grounding DINO and SAM image predictor to get the box and mask for all frames
"""
print("Total frames:", len(frame_names))
for start_frame_idx in range(0, len(frame_names), step):
# prompt grounding dino to get the box coordinates on specific frame
    print("start_frame_idx", start_frame_idx)
    # continue
    img_path = os.path.join(SOURCE_VIDEO_FRAME_DIR, frame_names[start_frame_idx])
    # image = Image.open(img_path).convert("RGB")
    image_source, image = load_image(img_path)

    image_base_name = frame_names[start_frame_idx].split(".")[0]
    mask_dict = MaskDictionaryModel(promote_type = PROMPT_TYPE_FOR_VIDEO, mask_name = f"mask_{image_base_name}.npy")

    boxes, confidences, labels = predict(
        model=grounding_model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
    )

    # process the box prompt for SAM 2
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    if boxes.shape[0] == 0:
        input_boxes = np.zeros((1, 4))
        labels = ['']
    else:
        input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    confidences = confidences.numpy().tolist()
    class_names = labels
    print("input_boxes")
    print(input_boxes)

    # prompt SAM image predictor to get the mask for the object
    image_predictor.set_image(image_source)

    # prompt SAM image predictor to get the mask for the object
    OBJECTS = class_names

    print(OBJECTS)

    # # FIXME: figure how does this influence the G-DINO model
    # torch.autocast(device_type=DEVICE, dtype=torch.bfloat16).__enter__()

    # if torch.cuda.get_device_properties(0).major >= 8:
    #     # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    #     torch.backends.cuda.matmul.allow_tf32 = True
    #     torch.backends.cudnn.allow_tf32 = True

    # prompt SAM 2 image predictor to get the mask for the object
    masks, scores, logits = image_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )
    # convert the mask shape to (n, H, W)
    # if masks.ndim == 2:
    #     masks = masks[None]
    #     scores = scores[None]
    #     logits = logits[None]
    if masks.ndim == 4:
        masks = masks.squeeze(1)
    """
    Step 3: Register each object's positive points to video predictor
    """

    # If you are using point prompts, we uniformly sample positive points based on the mask
    if mask_dict.promote_type == "mask":
        mask_dict.add_new_frame_annotation(mask_list=torch.tensor(masks).to(device), box_list=torch.tensor(input_boxes), label_list=OBJECTS)
    else:
        raise NotImplementedError("SAM 2 video predictor only support mask prompts")


    """
    Step 4: Propagate the video predictor to get the segmentation results for each frame
    """
    objects_count = mask_dict.update_masks(tracking_annotation_dict=sam2_masks, iou_threshold=0.8, objects_count=objects_count)
    frame_object_count[start_frame_idx] = objects_count
    print("objects_count", objects_count)
    video_predictor.reset_state(inference_state)
    if len(mask_dict.labels) == 0:
        print("No object detected in the frame, skip the frame {}".format(start_frame_idx))
        continue
    video_predictor.reset_state(inference_state)

    for object_id, object_info in mask_dict.labels.items():
        frame_idx, out_obj_ids, out_mask_logits = video_predictor.add_new_mask(
                inference_state,
                start_frame_idx,
                object_id,
                object_info.mask,
            )
    
    video_segments = {}  # output the following {step} frames tracking masks
    for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state, max_frame_num_to_track=step, start_frame_idx=start_frame_idx):
        frame_masks = MaskDictionaryModel()
        
        for i, out_obj_id in enumerate(out_obj_ids):
            out_mask = (out_mask_logits[i] > 0.0) # .cpu().numpy()
            object_info = ObjectInfo(instance_id = out_obj_id, mask = out_mask[0], class_name = mask_dict.get_target_class_name(out_obj_id), logit=mask_dict.get_target_logit(out_obj_id))
            object_info.update_box()
            frame_masks.labels[out_obj_id] = object_info
            image_base_name = frame_names[out_frame_idx].split(".")[0]
            frame_masks.mask_name = f"mask_{image_base_name}.npy"
            frame_masks.mask_height = out_mask.shape[-2]
            frame_masks.mask_width = out_mask.shape[-1]

        video_segments[out_frame_idx] = frame_masks
        sam2_masks = copy.deepcopy(frame_masks)

    print("video_segments:", len(video_segments))
    """
    Step 5: save the tracking masks and json files
    """
    for frame_idx, frame_masks_info in video_segments.items():
        mask = frame_masks_info.labels
        mask_img = torch.zeros(frame_masks_info.mask_height, frame_masks_info.mask_width)
        for obj_id, obj_info in mask.items():
            mask_img[obj_info.mask == True] = obj_id

        mask_img = mask_img.numpy().astype(np.uint16)
        np.save(os.path.join(mask_data_dir, frame_masks_info.mask_name), mask_img)

        json_data_path = os.path.join(json_data_dir, frame_masks_info.mask_name.replace(".npy", ".json"))
        frame_masks_info.to_json(json_data_path)
       

CommonUtils.draw_masks_and_box_with_supervision(SOURCE_VIDEO_FRAME_DIR, mask_data_dir, json_data_dir, result_dir)

print("try reverse tracking")
start_object_id = 0
object_info_dict = {}
for frame_idx, current_object_count in frame_object_count.items():
    print("reverse tracking frame", frame_idx, frame_names[frame_idx])
    if frame_idx != 0:
        video_predictor.reset_state(inference_state)
        image_base_name = frame_names[frame_idx].split(".")[0]
        json_data_path = os.path.join(json_data_dir, f"mask_{image_base_name}.json")
        json_data = MaskDictionaryModel().from_json(json_data_path)
        mask_data_path = os.path.join(mask_data_dir, f"mask_{image_base_name}.npy")
        mask_array = np.load(mask_data_path)
        for object_id in range(start_object_id+1, current_object_count+1):
            print("reverse tracking object", object_id)
            object_info_dict[object_id] = json_data.labels[object_id]
            video_predictor.add_new_mask(inference_state, frame_idx, object_id, mask_array == object_id)
    start_object_id = current_object_count
        
    
    for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state, max_frame_num_to_track=step*2,  start_frame_idx=frame_idx, reverse=True):
        image_base_name = frame_names[out_frame_idx].split(".")[0]
        json_data_path = os.path.join(json_data_dir, f"mask_{image_base_name}.json")
        json_data = MaskDictionaryModel().from_json(json_data_path)
        mask_data_path = os.path.join(mask_data_dir, f"mask_{image_base_name}.npy")
        mask_array = np.load(mask_data_path)
        # merge the reverse tracking masks with the original masks
        for i, out_obj_id in enumerate(out_obj_ids):
            out_mask = (out_mask_logits[i] > 0.0).cpu()
            if out_mask.sum() == 0:
                print("no mask for object", out_obj_id, "at frame", out_frame_idx)
                continue
            object_info = object_info_dict[out_obj_id]
            object_info.mask = out_mask[0]
            object_info.update_box()
            json_data.labels[out_obj_id] = object_info
            mask_array = np.where(mask_array != out_obj_id, mask_array, 0)
            mask_array[object_info.mask] = out_obj_id
            # print("mask_array.shape",mask_array.shape)
            # print("object_info.mask.shape",object_info.mask.shape)
        
        np.save(mask_data_path, mask_array)
        json_data.to_json(json_data_path)

        



"""
Step 6: Draw the results and save the video
"""
CommonUtils.draw_masks_and_box_with_supervision(SOURCE_VIDEO_FRAME_DIR, mask_data_dir, json_data_dir, result_dir+"_reverse")

create_video_from_images(result_dir, output_video_path, frame_rate=15)