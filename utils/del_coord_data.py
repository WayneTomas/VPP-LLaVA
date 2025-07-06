# import debugpy; debugpy.connect(('10.140.12.31', 5678))
from PIL import Image, ImageDraw, ImageFont, ImageStat
import numpy as np
import json
from typing import List
import re
import random
# from llava.prompt_utils import add_axis_pil, add_axis_pil_double

pattern = r'\[(\d+\.\d+(?:\s*,\s*\d+\.\d+)*)\]'
# img_path = "test_img/COCO_train2014_000000537553.jpg"
# img = Image.open(img_path)
img = Image.new('RGB', (336, 336), 'white')

image_mean = [
    0.48145466,
    0.4578275,
    0.40821073
  ]
font_path = "TIMES.TTF"  # 例如："arial.ttf"
font_size = 10
font = ImageFont.truetype(font_path, font_size)
point_size = 1


def add_axis_pil(img: Image, save_path: str=''):
    width, height = img.size
    # 由于没有提供font_path，这里使用默认字体
    # font = ImageFont.load_default()

    # 创建一个用于绘图的图层
    draw = ImageDraw.Draw(img)

    # 定义坐标轴和刻度的样式
    base_color = 'white'  # 坐标轴颜色
    tick_length = -10      # 刻度线长度
    tick_width = 1        # 刻度线宽度

    # 绘制X轴（上侧）

    draw.line((0, 0, width, 0), fill='white', width=tick_width)
    # 绘制Y轴（左侧）
    draw.line((0, height - 1, 0, 0), fill='white', width=tick_width)
    # draw.text((-tick_length - (-4), -tick_length + 4), str(f"{0.0/10:.0f}"), font=font, fill=base_color)

    # 绘制X轴刻度（上侧）
    for x in range(11):  # 0到10，共11个刻度
        tick_position = int(x * (width / 10))
        average_color = get_average_color(tick_position, 0, img, 15)
        axis_color = 'black' if average_color > 200 else 'white'  # 假设亮度基于R通道
        # axis_color = 'white'
        draw.line((tick_position-1, 0, tick_position-1, -tick_length), fill=axis_color, width=tick_width)
        # 文本位置在刻度的正上方，根据文本宽度调整偏移
        text = str(f"{x/10:.1f}")
        text_offset = tick_position - 10  # 偏移量，可根据需要调整
        if x == 10:
            draw.text((text_offset - 5, -tick_length + 4), text, font=font, fill=axis_color)
        elif x!= 0:
            draw.text((text_offset + 5, -tick_length + 4), text, font=font, fill=axis_color)

    # 绘制Y轴刻度（左侧）
    for y in range(11):  # 0到10，共11个刻度
        tick_position = int(y * (height / 10))
        average_color = get_average_color(0, tick_position, img, 15)
        axis_color = 'black' if average_color > 200 else 'white'  # 假设亮度基于R通道
        # axis_color = 'white'
        draw.line((0, height - 1 - tick_position, -tick_length, height - 1 - tick_position), fill=axis_color, width=tick_width)
        text = str(f"{y/10:.1f}")
        text_offset = -tick_length - (-4)  # 偏移量，可根据需要调整
        if y == 10:
            draw.text((text_offset, tick_position - (10)), text, font=font, fill=axis_color)
        elif y != 0:
            draw.text((text_offset, tick_position - 5), text, font=font, fill=axis_color)
    
    return img


def add_axis_pil_double(img: Image, save_path: str=''):
    width, height = img.size
    # 由于没有提供font_path，这里使用默认字体
    # font = ImageFont.load_default()

    # 创建一个用于绘图的图层
    draw = ImageDraw.Draw(img)

    # 定义坐标轴和刻度的样式
    base_color = 'white'  # 坐标轴颜色
    tick_length = -10      # 刻度线长度
    tick_width = 1        # 刻度线宽度




    # 绘制X轴（上侧）
    draw.line((0, 0, width, 0), fill='black', width=tick_width)
    # 绘制Y轴（左侧）
    draw.line((0, height - 1, 0, 0), fill='black', width=tick_width)
    # 绘制X轴（下侧）
    draw.line((0, height - 1, width, height - 1), fill='black', width=tick_width)
    # 绘制Y轴（右侧）
    draw.line((width - 1, 0, width - 1, height), fill='black', width=tick_width)

    # 绘制X轴刻度（上侧）
    for x in range(11):  # 0到10，共11个刻度
        tick_position = int(x * (width / 10))
        average_color = get_average_color(tick_position, 0, img, 15)
        axis_color = 'black' if average_color > 200 else 'white'  # 假设亮度基于R通道

        # 文本位置在刻度的正上方，根据文本宽度调整偏移
        text = str(f"{x/10:.1f}")
        text_offset = tick_position - 10  # 偏移量，可根据需要调整
        if x == 10:
            ...
        elif x!= 0:
            draw.line((tick_position-1, 0, tick_position-1, -tick_length), fill=axis_color, width=tick_width)
            draw.line((tick_position, height - 1, tick_position, height - 1 + tick_length), fill=axis_color, width=tick_width)
            draw.text((text_offset + 5, -tick_length + 4), text, font=font, fill=axis_color)
            draw.text((text_offset, height - 1 + tick_length - 15), text, font=font, fill=axis_color)

    # 绘制Y轴刻度（左侧）
    for y in range(11):  # 0到10，共11个刻度
        tick_position = int(y * (height / 10))
        average_color = get_average_color(0, tick_position, img, 15)
        axis_color = 'black' if average_color > 200 else 'white'  # 假设亮度基于R通道
        # axis_color = 'white'
        
        text = str(f"{y/10:.1f}")
        text_offset = -tick_length - (-4)  # 偏移量，可根据需要调整
        if y == 10:
            ...
            # draw.text((text_offset, tick_position - (15)), text, font=font, fill=axis_color)
        elif y != 0:
            draw.line((0, height - 1 - tick_position, -tick_length, height - 1 - tick_position), fill=axis_color, width=tick_width)
            draw.line((width - 1 + tick_length, height - 1 - tick_position, width - 1, height - 1 - tick_position), fill=axis_color, width=tick_width)
            draw.text((text_offset, tick_position - 5), text, font=font, fill=axis_color)
            draw.text((width - 1 + tick_length - 15, tick_position - 5), text, font=font, fill=axis_color)
    
    return img


def get_average_color(x, y, image, area_size=5):
    # 确保区域不超出图像边界
    left = max(0, x - area_size // 2)
    top = max(0, y - area_size // 2)
    right = min(image.width, x + area_size // 2)
    bottom = min(image.height, y + area_size // 2)
    
    # 裁剪区域
    area = image.crop((left, top, right, bottom))
    
    # 获取平均颜色
    return sum(ImageStat.Stat(area).mean)/3

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


img = expand2square(img, tuple(int(x*255) for x in image_mean))



def read_jsonl(path) -> List[dict]:
    with open(path, 'r') as f:
        json_str = f.read()

    data_list = []
    for line in json_str.splitlines():
        data = json.loads(line)
        data_list.append(data)

    return data_list


def read_json(path) -> List[dict]:
    with open(path, 'r') as f:
        json_str = f.read()

    data_list = json.loads(json_str)

    return data_list


def handle_conv_bbox(conversations: List[str]):
    coordinates = []
    for conv in conversations:
        v = conv['value']
        matches = re.findall(pattern, v)
        if matches != []:
            flag = False
            float_list = list(map(float, matches[-1].split(',')))
            if float_list not in coordinates:
                coordinates.append(float_list)
    return coordinates


def handle_conv_bbox_var(conversations: List[str]):
    human_coordinates = []
    gpt_coordinates = []
    re_coordinates = []
    for conv in conversations:
        if conv['from'] == "human":
            v = conv['value']
            
            matches = re.findall(pattern, v)
            if matches != []:
                float_list = list(map(float, matches[-1].split(',')))
                if float_list not in human_coordinates:
                    human_coordinates.append(float_list)
        elif conv['from'] == "gpt":
            v = conv['value']
            matches = re.findall(pattern, v)
            if matches != []:
                float_list = list(map(float, matches[-1].split(',')))
                if float_list not in gpt_coordinates:
                    gpt_coordinates.append(float_list)

    for coord in human_coordinates:
        if coord not in gpt_coordinates:
            re_coordinates.append(coord)
    
    if re_coordinates != []:
        return re_coordinates[0]
    return re_coordinates


def handle_conv_bbox_var_grd(conversations: List[str]):
    # 返回grounding结果里的bbox
    human_coordinates = []
    gpt_coordinates = []

    for conv in conversations:
        if conv['from'] == "human":
            v = conv['value']
            
            matches = re.findall(pattern, v)
            if matches != []:
                float_list = list(map(float, matches[-1].split(',')))
                if float_list not in human_coordinates:
                    human_coordinates.append(float_list)
        elif conv['from'] == "gpt":
            v = conv['value']
            matches = re.findall(pattern, v)
            if matches != []:
                float_list = list(map(float, matches[-1].split(',')))
                if float_list not in gpt_coordinates:
                    gpt_coordinates.append(float_list)

    
    if gpt_coordinates != []:
        return random.choice(gpt_coordinates)
    return gpt_coordinates


def handle_image_token(conversations: List[str]):
    count = 0
    for conv in conversations:
        v = conv['value']
        count += v.count('<image>')
    return count



# def add_bbox_pil(img: Image, bbox, save_path: str=''):
#     draw = ImageDraw.Draw(img)
    
#     for box in bbox:
#         # 绘制矩形边界框
#         box_ori = [x * img.width for x in box]
#         text_top_left_color = get_average_color(box_ori[0], box_ori[1], img, 30)
#         text_bottom_right_color = get_average_color(box_ori[2], box_ori[3], img, 30)
#         text_top_left_color = 'black' if text_top_left_color > 200 else 'white'  # 假设亮度基于R通道
#         text_bottom_right_color = 'black' if text_bottom_right_color > 200 else 'white'  # 假设亮度基于R通道
#         text_top_left = str([box[0], box[1]])
#         text_bottom_right = str([box[2], box[3]])
#         draw.text((box_ori[0] - 30 , box_ori[1] -20), text_top_left, font=font, fill=text_top_left_color)
#         draw.text((box_ori[2] - 30, box_ori[3] + 5), text_bottom_right, font=font, fill=text_bottom_right_color)

        
#         draw.rectangle(box_ori, outline='red', width=1)
        
#         draw.rectangle([box_ori[0] - point_size, box_ori[1] - point_size, box_ori[0] + point_size, box_ori[1] + point_size], fill='red')
#         draw.rectangle([box_ori[2] - point_size, box_ori[3] - point_size, box_ori[2] + point_size, box_ori[3] + point_size], fill='red')
#     return img


def add_bbox_scatter_pil(img: Image, bbox, save_path: str=''):
    draw = ImageDraw.Draw(img)
    for box in bbox:
        # 绘制矩形边界框
        box_ori = [x * img.width for x in box]
        text_top_left_color = get_average_color(box_ori[0], box_ori[1], img, 30)
        text_bottom_right_color = get_average_color(box_ori[2], box_ori[3], img, 30)
        text_top_left_color = 'black' if text_top_left_color > 200 else 'white'  # 假设亮度基于R通道
        text_bottom_right_color = 'black' if text_bottom_right_color > 200 else 'white'  # 假设亮度基于R通道
        text_top_left = str([box[0], box[1]])
        text_bottom_right = str([box[2], box[3]])
        draw.text((box_ori[0] - 30 , box_ori[1] -20), text_top_left, font=font, fill=text_top_left_color)
        draw.text((box_ori[2] - 30, box_ori[3] + 5), text_bottom_right, font=font, fill=text_bottom_right_color)
        draw.rectangle(box_ori, outline='red', width=1)
        
        draw.rectangle([box_ori[0] - point_size, box_ori[1] - point_size, box_ori[0] + point_size, box_ori[1] + point_size], fill='red')
        draw.rectangle([box_ori[2] - point_size, box_ori[3] - point_size, box_ori[2] + point_size, box_ori[3] + point_size], fill='red')
    return img



def preprocessing(data):
    bbox_count = 0
    total_count = 0
    new_data = []
    
    coordinates_dict = {}

    for item in data:
        flag = False
        coordinates = []
        conversations = item['conversations']
        count = handle_image_token(conversations)
        if count == 1:
            total_count += 1
        for conv in conversations:
            v = conv['value']
            matches = re.findall(pattern, v)
            if matches != []:
                flag = True
                float_list = list(map(float, matches[-1].split(',')))
                if float_list not in coordinates:
                    coordinates.append(float_list)
        if flag:
            new_data.append(item)
        else:
            coordinates_dict[item['id']] = coordinates
        
        ###################################################
        # test if bbox valid
        # bbox = handle_conv_bbox_var_grd(conversations)
        # if bbox != [] and len(bbox) == 4:
        #     bbox_count += 1
        #     add_axis_pil_double(img,'')
        #     img.save("double.jpg")
        #     return
        #     new_data.append(bbox)
        # total_count += 1
        # ####################################################

    with open("playground/data/llava_v1_5_mix665k_GRD_final.json", 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False)
        print(f"写入完成...")
    print(total_count)


if __name__ == "__main__":
    ori_path = "playground/data/llava_v1_5_mix665k.json"
    ori_data = read_json(ori_path)
    # preprocessing(data=ori_data)
    
    print("ff")
# print()
