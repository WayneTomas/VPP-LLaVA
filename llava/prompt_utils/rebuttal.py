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

def add_axis_pil_double(img: Image, save_path: str=''):
    axis_color = 'black'
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
        # average_color = get_average_color(tick_position, 0, img, 15)
        # axis_color = 'black' if average_color > 200 else 'white'  # 假设亮度基于R通道

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
        # average_color = get_average_color(0, tick_position, img, 15)
        # axis_color = 'black' if average_color > 200 else 'white'  # 假设亮度基于R通道
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


def add_axis_pil_double_var(img: Image, save_path: str=''):
    axis_color = 'black'
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
    for x in range(21):  # 0到10，共11个刻度
        tick_position = int(x * (width / 20))
        # average_color = get_average_color(tick_position, 0, img, 15)
        # axis_color = 'black' if average_color > 200 else 'white'  # 假设亮度基于R通道

        # 文本位置在刻度的正上方，根据文本宽度调整偏移
        text = str(f"{x/20:.2f}")
        text_offset = tick_position - 20  # 偏移量，可根据需要调整
        if x == 20:
            ...
        elif x == 1:
            draw.line((tick_position-1, 0, tick_position-1, -tick_length), fill=axis_color, width=tick_width)
            draw.line((tick_position, height - 1, tick_position, height - 1 + tick_length), fill=axis_color, width=tick_width)
        elif x == 19:
            draw.line((tick_position-1, 0, tick_position-1, -tick_length), fill=axis_color, width=tick_width)
            draw.line((tick_position, height - 1, tick_position, height - 1 + tick_length), fill=axis_color, width=tick_width)
        elif x!= 0:
            draw.line((tick_position-1, 0, tick_position-1, -tick_length), fill=axis_color, width=tick_width)
            draw.line((tick_position, height - 1, tick_position, height - 1 + tick_length), fill=axis_color, width=tick_width)
            draw.text((text_offset + 5, -tick_length + 4), text, font=font, fill=axis_color)
            draw.text((text_offset, height - 1 + tick_length - 20), text, font=font, fill=axis_color)

    # 绘制Y轴刻度（左侧）
    for y in range(21):  # 0到10，共11个刻度
        tick_position = int(y * (height / 20))
        # average_color = get_average_color(0, tick_position, img, 15)
        # axis_color = 'black' if average_color > 200 else 'white'  # 假设亮度基于R通道
        # axis_color = 'white'
        
        text = str(f"{y/20:.2f}")
        text_offset = -tick_length - (-4)  # 偏移量，可根据需要调整
        if y == 20:
            ...
            # draw.text((text_offset, tick_position - (15)), text, font=font, fill=axis_color)
        elif y == 1 or y == 19:
            draw.line((0, height - 1 - tick_position, -tick_length, height - 1 - tick_position), fill=axis_color, width=tick_width)
            draw.line((width - 1 + tick_length, height - 1 - tick_position, width - 1, height - 1 - tick_position), fill=axis_color, width=tick_width)
        elif y != 0:
            draw.line((0, height - 1 - tick_position, -tick_length, height - 1 - tick_position), fill=axis_color, width=tick_width)
            draw.line((width - 1 + tick_length, height - 1 - tick_position, width - 1, height - 1 - tick_position), fill=axis_color, width=tick_width)
            draw.text((text_offset, tick_position - 10), text, font=font, fill=axis_color)
            draw.text((width - 1 + tick_length - 30, tick_position - 10), text, font=font, fill=axis_color)
    
    return img


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


def add_axis_pil_cross(img: Image, save_path: str=''):
    axis_color = 'black'
    # 获取图像的宽度和高度
    width, height = img.size
    # 计算中心点坐标
    center_x = width // 2
    center_y = height // 2

    # 由于没有提供font_path，这里使用默认字体
    # font = ImageFont.load_default()

    # 创建一个用于绘图的图层
    draw = ImageDraw.Draw(img)

    # 定义坐标轴和刻度的样式
    base_color = 'white'  # 坐标轴颜色
    tick_length = -10      # 刻度线长度
    tick_width = 1        # 刻度线宽度

    # 绘制水平轴
    draw.line([(0, center_y), (width, center_y)], fill=axis_color, width=tick_width)

    # 绘制垂直轴
    draw.line([(center_x, 0), (center_x, height)], fill=axis_color, width=tick_width)

    # 绘制X轴刻度（上侧）
    for x in range(11):  # 0到10，共11个刻度
        tick_position = int(x * (width / 10))

        # 文本位置在刻度的正上方，根据文本宽度调整偏移
        text = str(f"{x/10:.1f}")
        text_offset = tick_position - 10  # 偏移量，可根据需要调整
        if x == 10:
            ...
        elif x == 5:
            ...
            # draw.text((text_offset + 15, center_y-tick_length-5), text, font=font, fill=axis_color)
        elif x!= 0:
            draw.line((tick_position, center_y - tick_length, tick_position, center_y+tick_length), fill=axis_color, width=tick_width)
            draw.text((text_offset + 5, center_y-tick_length+5), text, font=font, fill=axis_color)

    # 绘制Y轴刻度（左侧）
    for y in range(11):  # 0到10，共11个刻度
        tick_position = int(y * (height / 10))
        
        text = str(f"{y/10:.1f}")
        text_offset = -tick_length - (-4)  # 偏移量，可根据需要调整
        if y == 10:
            ...
        elif y == 5:
            draw.line((center_x+tick_length, height - tick_position, -tick_length+center_x, height - tick_position), fill=axis_color, width=tick_width)
            # draw.text((text_offset, tick_position - (15)), text, font=font, fill=axis_color)
        elif y != 0:
            draw.line((center_x+tick_length, height - tick_position, -tick_length+center_x, height - tick_position), fill=axis_color, width=tick_width)
            draw.text((center_x+15, tick_position - 5), text, font=font, fill=axis_color)
    
    return img


if __name__ == "__main__":
    add_axis_pil_cross(img,"test.jpg")
    # img = img.resize((336, 336))
    img.save("test.jpg")
