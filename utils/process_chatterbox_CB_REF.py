# import debugpy; debugpy.connect(('10.140.12.33', 5672))
import json
from typing import List
import cv2
from PIL import Image
import numpy as np
import re
import os
import random

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
    new_bbox_norm_padded = [round(coord, 2) for coord in new_bbox_norm_padded]
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
    new_data_list = []
    
    for line in data_list:
        new_conversations = []
        flag = False
        conversations = line["conversation"]
        img_path = os.path.join(data_root, 'vg', line["image"])
        if "coco" in img_path:
            print(f"Igoring image {img_path}")
            print()
            continue
        try:
            img = Image.open(img_path).convert('RGB')
        except:
            print(f"Error loading image: {img_path}")
            continue
        w = img.width
        h = img.height
        for i, conv in enumerate(conversations):
            if conv["from"] == "human":
                v = conv["value"]

                # mod_question = f"<image>\nPlease provide a short description for this region: {expression}"
                match = coordinate_pattern.search(v)
                if match:
                    original_coordinates_str = match.group()
                    original_coordinates = json.loads(original_coordinates_str)
                    bbox_ori = original_coordinates

                        # bbox_ori = transfer2ori(w=w, h=h, bbox_norm=original_coordinates)
                    bbox_padded = trasfer2pad(w=w, h=h, bbox_ori=bbox_ori)
                    # val_bbox(img=img, original_coordinates=original_coordinates, bbox_padded=bbox_padded, i=i, img_path=img_path)
                        # updated_line = re.sub(r'\[\d+\.\d+, \d+\.\d+, \d+\.\d+, \d+\.\d+\]', str(bbox_padded), v)
                    if i == 0:
                        conversations[i]['value'] = f"<image>\nPlease provide a short description for this region: {bbox_padded}."
                    else:
                        conversations[i]['value'] = f"Please provide a short description for this region: {bbox_padded}."

        new_data_list.append({"id": f"{line['id']}", "image": f"vg/{line['image']}", "conversations":conversations})
        count += 1


    print("写入开始...")
    with open("playground/data/Chatterbox_CB_REF_refined.json", 'w', encoding='utf-8') as f:
        json.dump(new_data_list, f, ensure_ascii=False)
    

if __name__ == "__main__":
    data_path = "CB_REF.json"

    data_list = read_json(data_path)["data"]
    refined_data = data_refine(data_list=data_list)
    print("all done!")




    # test_path = "playground/data/vg/VG_100K/2368107.jpg"
    # test_img = Image.open(test_path)
    # width= test_img.width
    # height = test_img.height
    # padded_img = expand2square(test_img, tuple(int(x*255) for x in image_mean))
    # bbox_norm = [0.222, 0.630, 0.335, 0.764]
    # bbox_ori = transfer2ori(w=width,h=height,bbox_norm=bbox_norm)
    # new_bbox_norm_padded = trasfer2pad(w=width,h=height,bbox_ori=bbox_ori)
    
    # # 将Pillow图像转换为numpy数组
    # pil_img_array = np.array(padded_img)

    # # 将RGB格式转换为BGR格式
    # cv2_img = cv2.cvtColor(pil_img_array, cv2.COLOR_RGB2BGR)
    # # 设置边界框的颜色和线宽
    # color = (0, 255, 0)  # 绿色
    # line_thickness = 2

    # x_min = int(bbox_padded[0]*max(w,h))
    # y_min = int(bbox_padded[1]*max(w,h))
    # x_max = int(bbox_padded[2]*max(w,h))
    # y_max = int(bbox_padded[3]*max(w,h))

    # # 在图像上绘制边界框
    # cv2.rectangle(cv2_img, (x_min, y_min), (x_max, y_max), color, line_thickness)
    # output_path = 'path_to_save_image.jpg'  # 替换为你想要保存图像的路径
    # cv2.imwrite(output_path, cv2_img)
    # print()