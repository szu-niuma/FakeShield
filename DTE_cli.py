import argparse
import json
from io import BytesIO
from pathlib import Path

import requests
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from transformers import TextStreamer

from DTE_FDM.llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from DTE_FDM.llava.conversation import SeparatorStyle, conv_templates
from DTE_FDM.llava.mm_utils import (
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from DTE_FDM.llava.model.builder import load_pretrained_model
from DTE_FDM.llava.utils import disable_torch_init


class DomainTagGenerator:
    def __init__(self, model_path, num_classes=3, device=None):
        """
        Initialize the DomainTagGenerator class.

        parameter:
        - model_path (str): The path to the model weight file.
        - num_classes (int): The number of categories in the category.
        - device (torch.device, optional): Device type, such as 'cpu' or 'cuda'."""
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes

        self.model = models.resnet50(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def predict(self, image_path):
        """
        Classification prediction of single images.

        parameter:
        - image_path (str): The path to the image file.

        return:
        - int: predicted category tags.
        """
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(image)
            _, predicted = torch.max(output, 1)
            label = predicted.item()

        return label


def load_image(image_file):
    if image_file.startswith("http://") or image_file.startswith("https://"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def DTE_FDM_init(args):
    # Model
    disable_torch_init()
    model_name = "llava-v1.5-13b"
    DTG = DomainTagGenerator(model_path=args.DTG_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device
    )

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    return tokenizer, model, image_processor, context_len, DTG, model_name


def DTE_FDM_cli(args):
    print("======== DTE_FDM Model Loading ========")
    tokenizer, model, image_processor, context_len, DTG, model_name = DTE_FDM_init(args)
    test_data = Path("/home/yuyangxin/data/imdl-demo/datasets/magic/test.json")
    # 读取json文件
    # TEST_DATA : [DATASET_NAME: DATASET_PATH]}
    with open(test_data, "r") as f:
        test_data = json.load(f)
    print("======== Image Processing ========")

    for dataset_name, dataset_path in test_data.items():
        # 读取json文件
        dataset_path = "/data0/yuyangxin/finetune-qwen/resource/datasets/without_instruct/coverage.json"
        with open(dataset_path, "r") as f:
            data = json.load(f)
        outputs = []
        print(f"======== {dataset_name} ========")
        for info in tqdm(data, desc=f"{dataset_name} 进度", unit="img"):
            img_path, mask_path = info[0], info[1]
            if mask_path == "positive":
                gt_label = 0
            else:
                gt_label = 1
            text = image_process(
                tokenizer,
                model,
                image_processor,
                context_len,
                DTG,
                model_name,
                img_path,
                args,
            )
            outputs.append([img_path, mask_path, gt_label, text])
        save_path = Path(args.output_path + f"{dataset_name}.json")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(outputs, f, indent=4, ensure_ascii=False)


def image_process(tokenizer, model, image_processor, context_len, DTG, model_name, image_path, args):
    conv = conv_templates[args.conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ("user", "assistant")
    else:
        roles = conv.roles
    image = load_image(image_path)
    label = DTG.predict(image_path)
    print("======== DTE_FDM Model Loaded ========")

    image_size = image.size
    image_tensor = process_images([image], image_processor, model.config)
    if type(image_tensor) is list:
        image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    # inp = input(f"{roles[0]}: ")
    inp = "Was this photo taken directly from the camera without any processing? Has it been tampered with by any artificial photo modification techniques such as ps? Please zoom in on any details in the image, paying special attention to the edges of the objects, capturing some unnatural edges and perspective relationships, some incorrect semantics, unnatural lighting and darkness etc."
    if label == 0:
        inp = "This is a picture that is suspected to have been tampered with by AIGC inpainting. " + inp
    elif label == 1:
        inp = "This is a picture that is suspected to have been tampered with by DeepFake. " + inp
    elif label == 2:
        inp = "This is a picture that is suspected to have been tampered with by Photoshop. " + inp

    print(f"{roles[1]}: ", end="")

    if image is not None:
        # first message
        if model.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + inp
        else:
            inp = DEFAULT_IMAGE_TOKEN + "\n" + inp
        conv.append_message(conv.roles[0], inp)
        image = None
    else:
        # later messages
        conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
    )
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    print("======== DTE_FDM Detect Begin ========")

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image_size],
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            streamer=streamer,
            use_cache=True,
        )

    outputs = tokenizer.decode(output_ids[0]).strip()
    conv.messages[-1][-1] = outputs

    if args.debug:
        print("\n", {"prompt": prompt, "outputs": outputs}, "\n")

    outputs = outputs.replace("<s>", "").replace("</s>", "")
    return outputs


if __name__ == "__main__":
    WEIGHT_PATH = "/home/yuyangxin/.cache/huggingface/hub/models--zhipeixu--fakeshield-v1-22b/snapshots/d0487aa9a8d7313c85e4cbef9de1e3a00fdc23c2"
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=f"{WEIGHT_PATH}/DTE-FDM")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--DTG-path", type=str, default=f"{WEIGHT_PATH}/DTG.pth")
    parser.add_argument("--output-path", type=str, default="output/")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--target_type", type=str, default="real_img")
    args = parser.parse_args()
    DTE_FDM_cli(args)
