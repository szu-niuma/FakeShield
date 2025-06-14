import argparse
import json
import os
import random
import sys
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPImageProcessor

from MFLM.model.GLaMM import GLaMMForCausalLM
from MFLM.model.llava import conversation as conversation_lib
from MFLM.model.llava.mm_utils import tokenizer_image_token
from MFLM.model.SAM.utils.transforms import ResizeLongestSide
from MFLM.tools.generate_utils import center_crop, create_feathered_mask
from MFLM.tools.markdown_utils import (
    ImageSketcher,
    article,
    colors,
    description,
    draw_bbox,
    examples,
    markdown_default,
    process_markdown,
    title,
)
from MFLM.tools.utils import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)


def parse_args(args):
    parser = argparse.ArgumentParser(description="FakeShield Model Demo")
    parser.add_argument("--version", default="./weight/fakeshield-v1-22b/MFLM")
    parser.add_argument("--DTE-FDM-output", type=str)
    parser.add_argument("--MFLM-output", type=str)
    parser.add_argument("--precision", default="bf16", type=str)
    parser.add_argument("--image_size", default=1024, type=int, help="Image size for grounding image encoder")
    parser.add_argument("--model_max_length", default=1536, type=int)
    parser.add_argument("--vision-tower", default="openai/clip-vit-large-patch14-336", type=str)
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--conv_type", default="llava_v1", type=str, choices=["llava_v1", "llava_llama_2"])
    parser.add_argument("--target_type", type=str, default="real_img")
    return parser.parse_args(args)


def read_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.loads(f.readline().strip())


def read_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def setup_tokenizer_and_special_tokens(args):
    """Load tokenizer and add special tokens."""
    tokenizer = AutoTokenizer.from_pretrained(
        args.version, model_max_length=args.model_max_length, padding_side="right", use_fast=False
    )
    print("\033[92m" + "---- Initialized tokenizer from: {} ----".format(args.version) + "\033[0m")
    tokenizer.pad_token = tokenizer.unk_token
    args.bbox_token_idx = tokenizer("<bbox>", add_special_tokens=False).input_ids[0]
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    args.bop_token_idx = tokenizer("<p>", add_special_tokens=False).input_ids[0]
    args.eop_token_idx = tokenizer("</p>", add_special_tokens=False).input_ids[0]

    return tokenizer


def initialize_model(args, tokenizer):
    """Initialize the GLaMM model."""
    model_args = {k: getattr(args, k) for k in ["seg_token_idx", "bbox_token_idx", "eop_token_idx", "bop_token_idx"]}

    model = GLaMMForCausalLM.from_pretrained(
        args.version, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, **model_args
    )
    print("\033[92m" + "---- Initialized model from: {} ----".format(args.version) + "\033[0m")

    # Configure model tokens
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    return model


def prepare_model_for_inference(model, args):
    # Initialize vision tower
    print(
        "\033[92m"
        + "---- Initialized Global Image Encoder (vision tower) from: {} ----".format(args.vision_tower)
        + "\033[0m"
    )
    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch.bfloat16, device=args.local_rank)
    model = model.bfloat16().cuda()
    return model


def grounding_enc_processor(x: torch.Tensor) -> torch.Tensor:
    IMG_MEAN = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    IMG_STD = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    IMG_SIZE = 1024
    x = (x - IMG_MEAN) / IMG_STD
    h, w = x.shape[-2:]
    x = F.pad(x, (0, IMG_SIZE - w, 0, IMG_SIZE - h))
    return x


def region_enc_processor(orig_size, post_size, bbox_img):
    orig_h, orig_w = orig_size
    post_h, post_w = post_size
    y_scale = post_h / orig_h
    x_scale = post_w / orig_w

    bboxes_scaled = [[bbox[0] * x_scale, bbox[1] * y_scale, bbox[2] * x_scale, bbox[3] * y_scale] for bbox in bbox_img]

    tensor_list = []
    for box_element in bboxes_scaled:
        ori_bboxes = np.array([box_element], dtype=np.float64)
        # Normalizing the bounding boxes
        norm_bboxes = ori_bboxes / np.array([post_w, post_h, post_w, post_h])
        # Converting to tensor, handling device and data type as in the original code
        tensor_list.append(torch.tensor(norm_bboxes, device="cuda").half().to(torch.bfloat16))

    if len(tensor_list) > 1:
        bboxes = torch.stack(tensor_list, dim=1)
        bboxes = [bboxes.squeeze()]
    else:
        bboxes = tensor_list
    return bboxes


def prepare_mask(input_image, image_np, pred_masks, text_output, color_history):
    """
    用于生成图像篡改区域的精确定位掩码，帮助用户直观地看到图像中哪些部分被篡改过。
    """
    save_img = None
    curr_mask = None  # 初始化curr_mask以避免后续引用前未定义的错误

    for i, pred_mask in enumerate(pred_masks):
        if pred_mask.shape[0] == 0:
            continue
        pred_mask = pred_mask.detach().cpu().numpy()
        mask_list = [pred_mask[i] for i in range(pred_mask.shape[0])]
        if len(mask_list) > 0:
            save_img = image_np.copy()
            colors_temp = colors.copy()  # 使用colors的副本以避免修改原始列表
            seg_count = text_output.count("[SEG]")
            mask_list = mask_list[-seg_count:]
            for curr_mask in mask_list:
                # 确保colors_temp不为空
                if not colors_temp:
                    colors_temp = colors.copy()  # 如果为空，重新填充

                color = random.choice(colors_temp)
                colors_temp.remove(color)
                color_history.append(color)
                curr_mask = curr_mask > 0
                save_img[curr_mask] = (image_np * 0.5 + curr_mask[:, :, None].astype(np.uint8) * np.array(color) * 0.5)[
                    curr_mask
                ]
    seg_mask = np.zeros((curr_mask.shape[0], curr_mask.shape[1], 3), dtype=np.uint8)
    seg_mask[curr_mask] = [255, 255, 255]  # white for True values
    seg_mask[~curr_mask] = [0, 0, 0]  # black for False values
    seg_mask = Image.fromarray(seg_mask)
    mask_path = input_image.replace("image", "mask")
    # seg_mask.save('mask.jpg')

    return save_img, seg_mask


def inference(input_str, all_inputs, follow_up, generate):
    bbox_img = all_inputs["boxes"]
    input_image = all_inputs["image"]

    if not follow_up:
        conv = conversation_lib.conv_templates[args.conv_type].copy()
        conv.messages = []
        conv_history = {"user": [], "model": []}
        conv_history["user"].append(input_str)

    input_str = input_str.replace("&lt;", "<").replace("&gt;", ">")
    prompt = input_str
    prompt = f"The {DEFAULT_IMAGE_TOKEN} provides an overview of the picture." + "\n" + prompt
    if args.use_mm_start_end:
        replace_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

    if not follow_up:
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], "")
    else:
        conv.append_message(conv.roles[0], input_str)
        conv.append_message(conv.roles[1], "")
    prompt = conv.get_prompt()

    image_np = cv2.imread(input_image)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = image_np.shape[:2]
    original_size_list = [image_np.shape[:2]]

    # Prepare input for Global Image Encoder
    # 功能：提取图像的整体语义特征，理解图像的全局内容
    # 基础模型：使用 CLIP 模型 (openai/clip-vit-large-patch14-336)
    # 预处理：使用 CLIP 的标准预处理流程，包括缩放和标准化
    # 输出：图像的全局语义表征，用于图像和文本的跨模态理解
    # 大模型的vision_tower
    global_enc_image = (
        global_enc_processor.preprocess(image_np, return_tensors="pt")["pixel_values"][0].unsqueeze(0).cuda()
    )
    global_enc_image = global_enc_image.bfloat16()

    # Prepare input for Grounding Image Encoder
    # 归一化图像，使其符合模型的输入要求
    image = transform.apply_image(image_np)
    resize_list = [image.shape[:2]]
    grounding_enc_image = (
        grounding_enc_processor(torch.from_numpy(image).permute(2, 0, 1).contiguous()).unsqueeze(0).cuda()
    )
    grounding_enc_image = grounding_enc_image.bfloat16()

    # Prepare input for Region Image Encoder
    post_h, post_w = global_enc_image.shape[1:3]
    bboxes = None
    if len(bbox_img) > 0:
        bboxes = region_enc_processor((orig_h, orig_w), (post_h, post_w), bbox_img)

    input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
    input_ids = input_ids.unsqueeze(0).cuda()

    # Pass prepared inputs to model
    output_ids, pred_masks = model.evaluate(
        global_enc_image,  # vision_tower编码
        grounding_enc_image,  # 归一化图像
        input_ids,  # prompt: 图像 + 文字理解
        resize_list,
        original_size_list,
        max_tokens_new=512,
        bboxes=bboxes,  # 测试代码bboxes一定为空
    )

    # 从输出的标记(token)序列中移除图像标记(image token)的索引。
    output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]

    text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
    text_output = text_output.replace("\n", "").replace("  ", " ")
    text_output = text_output.split("ASSISTANT: ")[-1]

    # For multi-turn conversation
    conv.messages.pop()
    conv.append_message(conv.roles[1], text_output)
    conv_history["model"].append(text_output)
    color_history = []
    save_img = None
    if "[SEG]" in text_output:
        save_img, seg_mask = prepare_mask(input_image, image_np, pred_masks, text_output, color_history)
    output_str = text_output  # input_str
    # if save_img is not None:
    #     output_image = save_img  # input_image
    # else:
    #     if len(bbox_img) > 0:
    #         output_image = draw_bbox(image_np.copy(), bbox_img)
    #     else:
    #         output_image = input_image

    # markdown_out = process_markdown(output_str, color_history)

    # return output_image, markdown_out
    return seg_mask, output_str


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    print("======== MFLM Model Loading ========")
    tokenizer = setup_tokenizer_and_special_tokens(args)
    model = initialize_model(args, tokenizer)
    model = prepare_model_for_inference(model, args)
    global_enc_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
    transform = ResizeLongestSide(args.image_size)
    model.eval()

    # DTE_FDM_output = read_jsonl(args.DTE_FDM_output)
    dte_fdm_output = Path(args.DTE_FDM_output)
    # 读取文件夹下所有的json文件
    json_files = [f for f in dte_fdm_output.glob("*.json") if f.is_file()]
    ret = []
    for json_file in json_files:
        json_data = read_json(json_file)
        total_len = len(json_data)
        output_path: Path = Path(args.MFLM_output) / f"{json_file.stem}"
        output_path.mkdir(parents=True, exist_ok=True)
        for item in tqdm(json_data, desc=f"Processing {json_file.name}", leave=False):
            input_image, input_text = item[0], item[3]
            filename = os.path.basename(input_image)
            # if "has not been tampered with" in input_text:
            #     output_image = Image.open(input_image)
            #     output_image = Image.fromarray(
            #         np.zeros((output_image.size[1], output_image.size[0], 3), dtype=np.uint8)
            #     )
            #     markdown_out = None
            # else:
            #     output_image, markdown_out = inference(input_text, {"image": input_image, "boxes": []}, False, False)
            output_image, markdown_out = inference(input_text, {"image": input_image, "boxes": []}, False, False)
            save_path: Path = (output_path / filename).absolute().as_posix()
            output_image.save(save_path)
            ret.append(
                {
                    "image": item[0],
                    "gt_mask": item[1],
                    "pred_mask": save_path,
                    "gt_label": item[2],
                    "pred_label": 1 if "has not been tampered with" in input_text else 0,
                    "DTE_output": item[3],
                }
            )
        # 保存ret的结果
        output_json_path = Path(args.MFLM_output) / "{json_file.stem}.json"
        with open(output_json_path, "w") as f:
            json.dump(ret, f, indent=4)
