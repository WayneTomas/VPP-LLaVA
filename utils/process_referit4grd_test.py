# import debugpy; debugpy.connect(('22.9.42.22', 5673))
import json
from typing import List
import cv2
from PIL import Image
import numpy as np
import re
import os
import random
import torch

max_select_num = 20
image_mean = [
    0.48145466,
    0.4578275,
    0.40821073
  ]
coordinate_pattern = re.compile(r'\[(.*?)\]')
data_root = "playground/data"
color_1 = (0, 255, 0)  # 绿色
color_2 = (0, 0, 255)  # 绿色
line_thickness = 1
tmp_saved_root = "tmp_saved"

def read_json(path) -> List[dict]:
    with open(path, 'r') as f:
        json_str = f.read()

    data_list = json.loads(json_str)

    return data_list


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

def transfer2ori(w, h, bbox_norm):
    # 计算长边长度
    L = max(w, h)

    x_min = int(bbox_norm[0] * w)
    y_min = int(bbox_norm[1] * h)
    x_max = int(bbox_norm[2] * w)
    y_max = int(bbox_norm[3] * h)
    bbox_ori = [x_min, y_min, x_max, y_max]

    return bbox_ori

def trasfer2pad(w, h, bbox_ori):
    x_min = bbox_ori[0]
    y_min = bbox_ori[1]
    x_max = bbox_ori[2]
    y_max = bbox_ori[3]
    max_dim = max(w, h)
    padded_width = max_dim
    padded_height = max_dim
    x_min_padded = (x_min + (padded_width - w) // 2) / padded_width
    y_min_padded = (y_min + (padded_height - h) // 2) / padded_height
    x_max_padded = (x_max + (padded_width - w) // 2) / padded_width
    y_max_padded = (y_max + (padded_height - h) // 2) / padded_height
    # 新的归一化坐标
    new_bbox_norm_padded = [x_min_padded, y_min_padded, x_max_padded, y_max_padded]
    new_bbox_norm_padded = [abs(round(coord, 2)) for coord in new_bbox_norm_padded]
    return new_bbox_norm_padded

def val_bbox(img, original_coordinates, bbox_padded, i, img_path):
    w = img.width
    h = img.height
    L = max(w,h)
    pil_img_array = np.array(img)
    cv2_img = cv2.cvtColor(pil_img_array, cv2.COLOR_RGB2BGR)
    base_name = os.path.basename(img_path)

    # 1. 验证未pad之前的    
    x_min = int(original_coordinates[0])
    y_min = int(original_coordinates[1])
    x_max = int(original_coordinates[2])
    y_max = int(original_coordinates[3])
    cv2.rectangle(cv2_img, (x_min, y_min), (x_max, y_max), color_1, line_thickness)

    output_path = os.path.join(tmp_saved_root, "not_pad", base_name.split('.')[0]+f"_{str(i)}."+base_name.split('.')[1])
    cv2.imwrite(output_path, cv2_img)

    # 2. 验证pad之后的
    padded_img = expand2square(img, tuple(int(x*255) for x in image_mean))
    pil_img_array = np.array(padded_img)
    cv2_img = cv2.cvtColor(pil_img_array, cv2.COLOR_RGB2BGR)
    x_min = int(bbox_padded[0]*L)
    y_min = int(bbox_padded[1]*L)
    x_max = int(bbox_padded[2]*L)
    y_max = int(bbox_padded[3]*L)
    cv2.rectangle(cv2_img, (x_min, y_min), (x_max, y_max), color_2, line_thickness)

    output_path = os.path.join(tmp_saved_root, "pad", base_name.split('.')[0]+f"_{str(i)}."+base_name.split('.')[1])
    cv2.imwrite(output_path, cv2_img)

    

def data_refine(data_list):
    count = 0

    #############################################################################################################################
    new_data_list = []
    for line in data_list:
        img_name, bbox_ori, expression = line[0], list(line[2]), line[3]
        new_conversations = []

        img_path = os.path.join(data_root, 'referit/images', img_name)

        try:
            img = Image.open(img_path).convert('RGB')
        except:
            print(f"Error loading image: {img_path}")
            print()
            continue
        w = img.width
        h = img.height

        new_data_list.append({"image": f"referit/images/{img_name}", "sent": expression, "bbox": [float(item) for item in bbox_ori], "height": h, "width": w})
        # bbox_padded = trasfer2pad(w=w, h=h, bbox_ori=bbox_ori)

        # mod_question = f"<image>\nPlease provide the bounding box coordinate of the region this sentence describes: {expression}."
        # mod_ans = f"{str(bbox_padded)}"
        
        # conv_human = {
        #         'from': 'human',
        #         'value': mod_question
        #     }
        # conv_gpt = {
        #     'from': 'gpt',
        #     'value': mod_ans
        # }
        # new_conversations = [conv_human, conv_gpt]
        # new_data_list.append({"id": f"{os.path.splitext(img_name)[0]}", "image": f"flickr30k/flickr30k-images/{img_name}", "conversations":new_conversations})

        count += 1

    print("referit_val写入开始...")
    with open("referit_val.jsonl", "w") as file:
        for item in new_data_list:
            json.dump(item, file, ensure_ascii=False)
            file.write("\n")  # 每个 JSON 对象后添加换行符
    print("referit_val写入开始...")

    

if __name__ == "__main__":
    data_path = "playground/data/transvg_split/data/referit/referit_val.pth"

    data_list = torch.load(data_path, map_location="cpu")
    refined_data = data_refine(data_list=data_list)
    print("all done!")

    