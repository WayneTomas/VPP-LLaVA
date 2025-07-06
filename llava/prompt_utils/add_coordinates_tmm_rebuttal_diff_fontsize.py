from PIL import Image, ImageDraw, ImageFont, ImageStat
import numpy as np
import json
from typing import List
import re
import random


img = Image.new('RGB', (336, 336), 'white')
font_path = "TIMES.TTF"  # 例如："arial.ttf"

point_size = 1


def add_axis_pil_double_font15(img: Image, save_path: str=''):
    font_size = 15
    font = ImageFont.truetype(font_path, font_size)
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
            draw.text((text_offset + 2, -tick_length + 2), text, font=font, fill=axis_color)
            draw.text((text_offset, height + tick_length - 20), text, font=font, fill=axis_color)

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
            draw.text((text_offset, tick_position - 8), text, font=font, fill=axis_color)
            draw.text((width + tick_length - 22, tick_position - 8), text, font=font, fill=axis_color)
    
    return img


def add_axis_pil_double_font5(img: Image, save_path: str=''):
    font_size = 5
    font = ImageFont.truetype(font_path, font_size)
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
            draw.text((text_offset + 6, -tick_length + 2), text, font=font, fill=axis_color)
            draw.text((text_offset + 6, height + tick_length - 8), text, font=font, fill=axis_color)

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
            draw.text((text_offset, tick_position - 3), text, font=font, fill=axis_color)
            draw.text((width + tick_length - 10, tick_position - 3), text, font=font, fill=axis_color)
    
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

if __name__ == "__main__":
    img = Image.new('RGB', (336, 336), 'white')
    axised_img = add_axis_pil_double_font5(img, save_path="")
    axised_img.save("axis_double_fontsize5.png")