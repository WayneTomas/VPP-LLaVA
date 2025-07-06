# import debugpy; debugpy.connect(('10.140.12.33', 5678))
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image, ImageDraw, ImageFont, ImageStat
import random

from typing import List
import re

pattern = r'\[(\d+\.\d+(?:\s*,\s*\d+\.\d+)*)\]'
    # 选择字体和大小，这里需要替换为您的字体文件路径
font_path = "TIMES.TTF"  # 例如："arial.ttf"
font_size = 15

point_size = 1
font = ImageFont.truetype(font_path, font_size)


def add_axis(img: Image):
    plt.figure()  # 创建新的图形窗口
    plt.xticks(np.linspace(0, img.width-1, 11), [f"{i/10:.1f}" for i in range(0,11,1)])  # 从0到图像宽度-1，共11个刻度
    plt.yticks(np.linspace(0, img.height-1, 11), [f"{i/10:.1f}" for i in range(0,11,1)])  # 从0到图像高度-1，共11个刻度

    plt.gca().invert_yaxis()  # 反转y轴，使得原点在左上角
    plt.gca().xaxis.tick_top()  # x轴刻度在上方
    plt.tight_layout()
    plt.imshow(img)
    
    # 引入 FigureCanvasAgg
    # 将plt转化为numpy数据
    canvas = FigureCanvasAgg(plt.gcf())
    plt.close()  # 关闭所有打开的图形窗口
    # 绘制图像
    canvas.draw()
    # 获取图像尺寸
    w, h = canvas.get_width_height()
    # 解码string 得到argb图像
    buf = np.fromstring(canvas.tostring_argb(), dtype=np.uint8)
    
        # 重构成w h 4(argb)图像
    buf.shape = (w, h, 4)
    # 转换为 RGBA
    buf = np.roll(buf, 3, axis=2)
    # 得到 Image RGBA图像对象 (需要Image对象的同学到此为止就可以了)
    image = Image.frombytes("RGBA", (w, h), buf.tostring())
    # 转换为numpy array rgba四通道数组
    image = np.asarray(image)
    # 转换为rgb图像
    rgb_image = image[:, :, :3]
    
    return Image.fromarray(rgb_image)


def add_axis_save_image(img: Image, save_path: str):
    plt.figure(figsize=(img.width/100, img.height/100), dpi=100)
    plt.xticks(np.linspace(0, img.width-1, 11), [f"{i/10:.1f}" for i in range(0,11,1)])  # 从0到图像宽度-1，共11个刻度
    plt.yticks(np.linspace(0, img.height-1, 11), [f"{i/10:.1f}" for i in range(0,11,1)])  # 从0到图像高度-1，共11个刻度

    plt.gca().invert_yaxis()  # 反转y轴，使得原点在左上角
    plt.gca().xaxis.tick_top()  # x轴刻度在上方
    plt.tight_layout()
    plt.imshow(img)
    fig = plt.gcf()  # 获取当前的图形对象
    fig.canvas.draw()  # 绘制图形
    width, height = fig.canvas.get_width_height()
    image_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_data = image_data.reshape((height, width, 3))
    image = Image.fromarray(image_data)
    plt.close()
    return image
    # # plt.savefig(save_path, bbox_inches='tight')
    
    
    # # 引入 FigureCanvasAgg
    # # 将plt转化为numpy数据
    # canvas = FigureCanvasAgg(plt.gcf())
    # plt.close()
    # # 绘制图像
    # canvas.draw()
    # # 获取图像尺寸
    # w, h = canvas.get_width_height()
    # # 解码string 得到argb图像
    # buf = np.fromstring(canvas.tostring_argb(), dtype=np.uint8)
    
    #     # 重构成w h 4(argb)图像
    # buf.shape = (w, h, 4)
    # # 转换为 RGBA
    # buf = np.roll(buf, 3, axis=2)
    # # 得到 Image RGBA图像对象 (需要Image对象的同学到此为止就可以了)
    # image = Image.frombytes("RGBA", (w, h), buf.tostring())
    # # 转换为numpy array rgba四通道数组
    # image = np.asarray(image)
    # # 转换为rgb图像
    # rgb_image = image[:, :, :3]
    
    # return Image.fromarray(rgb_image)

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


# def add_axis_pil(img: Image, save_path: str):
#     width, height = img.size
#     # 由于没有提供font_path，这里使用默认字体
#     font = ImageFont.load_default()

#     # 创建一个用于绘图的图层
#     draw = ImageDraw.Draw(img)

#     # 定义坐标轴和刻度的样式
#     base_color = 'white'  # 坐标轴颜色
#     tick_length = -10      # 刻度线长度
#     tick_width = 1        # 刻度线宽度

#     # 选择字体和大小，这里需要替换为您的字体文件路径
#     font_path = "TIMESBD.TTF"  # 例如："arial.ttf"
#     font_size = 20
#     font = ImageFont.truetype(font_path, font_size)


#     # 绘制X轴（上侧）

#     # draw.line((0, 0, width, 0), fill=base_color, width=tick_width)
#     # 绘制Y轴（左侧）
#     # draw.line((0, height - 1, 0, 0), fill=base_color, width=tick_width)
#     # draw.text((-tick_length - (-4), -tick_length + 4), str(f"{0.0/10:.0f}"), font=font, fill=base_color)

#     # 绘制X轴刻度（上侧）
#     for x in range(11):  # 0到10，共11个刻度
#         tick_position = int(x * (width / 10))
#         average_color = get_average_color(tick_position, 0, img, 20)
#         axis_color = 'black' if average_color > 200 else 'white'  # 假设亮度基于R通道
#         draw.line((tick_position-1, 0, tick_position-1, -tick_length), fill=axis_color, width=tick_width)
#         # 文本位置在刻度的正上方，根据文本宽度调整偏移
#         text = str(f"{x/10:.1f}")
#         text_offset = tick_position - 10  # 偏移量，可根据需要调整
#         if x == 10:
#             draw.text((text_offset - 16, -tick_length + 4), text, font=font, fill=axis_color)
#         elif x!= 0:
#             draw.text((text_offset, -tick_length + 4), text, font=font, fill=axis_color)

#     # 绘制Y轴刻度（左侧）
#     for y in range(11):  # 0到10，共11个刻度
#         tick_position = int(y * (height / 10))
#         average_color = get_average_color(0, tick_position, img, 20)
#         axis_color = 'black' if average_color > 200 else 'white'  # 假设亮度基于R通道
#         draw.line((0, height - 1 - tick_position, -tick_length, height - 1 - tick_position), fill=axis_color, width=tick_width)
#         text = str(f"{y/10:.1f}")
#         text_offset = -tick_length - (-4)  # 偏移量，可根据需要调整
#         if y == 10:
#             draw.text((text_offset, tick_position - (20)), text, font=font, fill=axis_color)
#         elif y != 0:
#             draw.text((text_offset, tick_position - 10), text, font=font, fill=axis_color)
    
#     return img

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
            draw.text((text_offset - 10, -tick_length + 4), text, font=font, fill=axis_color)
        elif x!= 0:
            draw.text((text_offset, -tick_length + 4), text, font=font, fill=axis_color)

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
            draw.text((text_offset, tick_position - (15)), text, font=font, fill=axis_color)
        elif y != 0:
            draw.text((text_offset, tick_position - 10), text, font=font, fill=axis_color)
    
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
    draw.line((0, 0, width, 0), fill='white', width=tick_width)
    # 绘制Y轴（左侧）
    draw.line((0, height - 1, 0, 0), fill='white', width=tick_width)
    # 绘制X轴（下侧）
    draw.line((0, height - 1, width, height - 1), fill='white', width=tick_width)
    # 绘制Y轴（右侧）
    draw.line((width - 1, 0, width - 1, height), fill='white', width=tick_width)

    # 绘制X轴刻度（上侧）
    for x in range(11):  # 0到10，共11个刻度
        tick_position = int(x * (width / 10))
        average_color = get_average_color(tick_position, 0, img, 15)
        axis_color = 'black' if average_color > 200 else 'white'  # 假设亮度基于R通道
        # axis_color = 'white'
        draw.line((tick_position-1, 0, tick_position-1, -tick_length), fill=axis_color, width=tick_width)
        draw.line((tick_position, height - 1, tick_position, height - 1 + tick_length), fill=axis_color, width=tick_width)
        # 文本位置在刻度的正上方，根据文本宽度调整偏移
        text = str(f"{x/10:.1f}")
        text_offset = tick_position - 10  # 偏移量，可根据需要调整
        if x == 10:
            ...
            # draw.text((text_offset - 10, -tick_length + 4), text, font=font, fill=axis_color)
        elif x!= 0:
            draw.text((text_offset, -tick_length + 4), text, font=font, fill=axis_color)
            draw.text((text_offset, height - 1 + tick_length - 20), text, font=font, fill=axis_color)

    # 绘制Y轴刻度（左侧）
    for y in range(11):  # 0到10，共11个刻度
        tick_position = int(y * (height / 10))
        average_color = get_average_color(0, tick_position, img, 15)
        axis_color = 'black' if average_color > 200 else 'white'  # 假设亮度基于R通道
        # axis_color = 'white'
        draw.line((0, height - 1 - tick_position, -tick_length, height - 1 - tick_position), fill=axis_color, width=tick_width)
        draw.line((width - 1 + tick_length, height - 1 - tick_position, width - 1, height - 1 - tick_position), fill=axis_color, width=tick_width)
        text = str(f"{y/10:.1f}")
        text_offset = -tick_length - (-4)  # 偏移量，可根据需要调整
        if y == 10:
            ...
            # draw.text((text_offset, tick_position - (15)), text, font=font, fill=axis_color)
        elif y != 0:
            draw.text((text_offset, tick_position - 10), text, font=font, fill=axis_color)
            draw.text((width - 1 + tick_length - 20, tick_position - 10), text, font=font, fill=axis_color)
    
    return img


def add_axis_pil_black(img: Image, save_path: str=''):
    width, height = img.size
    # 由于没有提供font_path，这里使用默认字体
    # font = ImageFont.load_default()

    # 创建一个用于绘图的图层
    draw = ImageDraw.Draw(img)

    # 定义矩形框的尺寸和位置
    # 假设矩形框的高度和宽度分别为100像素
    rect_width = 35
    rect_height = 35
    fill_color = 'black'

    # 在上边界绘制填充的矩形框
    draw.rectangle([0, 0, rect_width, width], fill=fill_color)
    # 在左边界绘制填充的矩形框
    draw.rectangle([0, 0, height, rect_height], fill=fill_color)

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
        # average_color = get_average_color(tick_position, 0, img, 15)
        # axis_color = 'black' if average_color > 200 else 'white'  # 假设亮度基于R通道
        axis_color = 'white'
        draw.line((tick_position-1, 0, tick_position-1, -tick_length), fill=axis_color, width=tick_width)
        # 文本位置在刻度的正上方，根据文本宽度调整偏移
        text = str(f"{x/10:.1f}")
        text_offset = tick_position - 10  # 偏移量，可根据需要调整
        if x == 10:
            draw.text((text_offset - 10, -tick_length + 4), text, font=font, fill=axis_color)
        elif x!= 0:
            draw.text((text_offset, -tick_length + 4), text, font=font, fill=axis_color)

    # 绘制Y轴刻度（左侧）
    for y in range(11):  # 0到10，共11个刻度
        tick_position = int(y * (height / 10))
        # average_color = get_average_color(0, tick_position, img, 15)
        # axis_color = 'black' if average_color > 200 else 'white'  # 假设亮度基于R通道
        axis_color = 'white'
        draw.line((0, height - 1 - tick_position, -tick_length, height - 1 - tick_position), fill=axis_color, width=tick_width)
        text = str(f"{y/10:.1f}")
        text_offset = -tick_length - (-4)  # 偏移量，可根据需要调整
        if y == 10:
            draw.text((text_offset, tick_position - (15)), text, font=font, fill=axis_color)
        elif y != 0:
            draw.text((text_offset, tick_position - 10), text, font=font, fill=axis_color)
    
    return img


def add_axis_pil_pad(img: Image, save_path: str=''):
    width, height = img.size
    # 由于没有提供font_path，这里使用默认字体
    # font = ImageFont.load_default()

    # 创建一个用于绘图的图层
    draw = ImageDraw.Draw(img)

    # 定义矩形框的尺寸和位置
    # 假设矩形框的高度和宽度分别为100像素
    rect_width = 35
    rect_height = 35
    fill_color = tuple(int(x*255) for x in [0.48145466, 0.4578275, 0.40821073])

    # 在上边界绘制填充的矩形框
    draw.rectangle([0, 0, rect_width, width], fill=fill_color)
    # 在左边界绘制填充的矩形框
    draw.rectangle([0, 0, height, rect_height], fill=fill_color)

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
        # average_color = get_average_color(tick_position, 0, img, 15)
        # axis_color = 'black' if average_color > 200 else 'white'  # 假设亮度基于R通道
        axis_color = 'white'
        draw.line((tick_position-1, 0, tick_position-1, -tick_length), fill=axis_color, width=tick_width)
        # 文本位置在刻度的正上方，根据文本宽度调整偏移
        text = str(f"{x/10:.1f}")
        text_offset = tick_position - 10  # 偏移量，可根据需要调整
        if x == 10:
            draw.text((text_offset - 10, -tick_length + 4), text, font=font, fill=axis_color)
        elif x!= 0:
            draw.text((text_offset, -tick_length + 4), text, font=font, fill=axis_color)

    # 绘制Y轴刻度（左侧）
    for y in range(11):  # 0到10，共11个刻度
        tick_position = int(y * (height / 10))
        # average_color = get_average_color(0, tick_position, img, 15)
        # axis_color = 'black' if average_color > 200 else 'white'  # 假设亮度基于R通道
        axis_color = 'white'
        draw.line((0, height - 1 - tick_position, -tick_length, height - 1 - tick_position), fill=axis_color, width=tick_width)
        text = str(f"{y/10:.1f}")
        text_offset = -tick_length - (-4)  # 偏移量，可根据需要调整
        if y == 10:
            draw.text((text_offset, tick_position - (15)), text, font=font, fill=axis_color)
        elif y != 0:
            draw.text((text_offset, tick_position - 10), text, font=font, fill=axis_color)
    
    return img


def add_bbox_pil(img: Image, bbox, save_path: str=''):
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
    return img


def add_bbox_pil_scatter_var(img: Image, bbox, save_path: str=''):
    draw = ImageDraw.Draw(img)
    for box in bbox:
        # 绘制矩形边界框
        box_ori = [x * img.width for x in box]
        # text_top_left_color = get_average_color(box_ori[0], box_ori[1], img, 30)
        # text_bottom_right_color = get_average_color(box_ori[2], box_ori[3], img, 30)
        # text_top_left_color = 'black' if text_top_left_color > 200 else 'white'  # 假设亮度基于R通道
        # text_bottom_right_color = 'black' if text_bottom_right_color > 200 else 'white'  # 假设亮度基于R通道
        text_top_left = str((box[0], box[1]))
        text_bottom_right = str((box[2], box[3]))
        draw.text((box_ori[0] - 30 , box_ori[1] -20), text_top_left, font=font, fill='red')
        draw.text((box_ori[2] - 30, box_ori[3] + 5), text_bottom_right, font=font, fill='red')
        draw.rectangle(box_ori, outline='red', width=1)
        
        
        draw.rectangle([box_ori[0] - point_size, box_ori[1] - point_size, box_ori[0] + point_size, box_ori[1] + point_size], fill='red')
        draw.rectangle([box_ori[2] - point_size, box_ori[3] - point_size, box_ori[2] + point_size, box_ori[3] + point_size], fill='red')
    return img

def add_bbox_pil_scatter_wotext(img: Image, bbox, save_path: str=''):
    draw = ImageDraw.Draw(img)
    for box in bbox:
        # 绘制矩形边界框
        box_ori = [x * img.width for x in box]
        draw.rectangle(box_ori, outline='red', width=1)
        
        
        draw.rectangle([box_ori[0] - point_size, box_ori[1] - point_size, box_ori[0] + point_size, box_ori[1] + point_size], fill='red')
        draw.rectangle([box_ori[2] - point_size, box_ori[3] - point_size, box_ori[2] + point_size, box_ori[3] + point_size], fill='red')
    return img


def add_bbox_scatter_pil(img: Image, bbox, save_path: str=''):
    draw = ImageDraw.Draw(img)
    for box in bbox:
        # 绘制矩形边界框
        box_ori = [x * img.width for x in box]
        # text_top_left_color = get_average_color(box_ori[0], box_ori[1], img, 30)
        # text_bottom_right_color = get_average_color(box_ori[2], box_ori[3], img, 30)
        # text_top_left_color = 'black' if text_top_left_color > 200 else 'white'  # 假设亮度基于R通道
        # text_bottom_right_color = 'black' if text_bottom_right_color > 200 else 'white'  # 假设亮度基于R通道
        # text_top_left = str([box[0], box[1]])
        # text_bottom_right = str([box[2], box[3]])
        # draw.text((box_ori[0] - 30 , box_ori[1] -20), text_top_left, font=font, fill=text_top_left_color)
        # draw.text((box_ori[2] - 30, box_ori[3] + 5), text_bottom_right, font=font, fill=text_bottom_right_color)
        draw.rectangle(box_ori, outline='red', width=1)
        
        draw.rectangle([box_ori[0] - point_size, box_ori[1] - point_size, box_ori[0] + point_size, box_ori[1] + point_size], fill='red')
        draw.rectangle([box_ori[2] - point_size, box_ori[3] - point_size, box_ori[2] + point_size, box_ori[3] + point_size], fill='red')
    return img


def add_scatter_pil(img: Image, bbox, save_path: str=''):
    draw = ImageDraw.Draw(img)
    for box in bbox:
        # 绘制矩形边界框
        box_ori = [x * img.width for x in box]

        # box_color = get_average_color(box_ori[0], box_ori[1], img, max(box_ori))
        text_top_left_color = get_average_color(box_ori[0], box_ori[1], img, 30)
        text_bottom_right_color = get_average_color(box_ori[2], box_ori[3], img, 30)
        text_top_left_color = 'black' if text_top_left_color > 200 else 'white'  # 假设亮度基于R通道
        text_bottom_right_color = 'black' if text_bottom_right_color > 200 else 'white'  # 假设亮度基于R通道
        

        # draw.rectangle(box_ori, outline=box_color, width=1)
        text_top_left = str([box[0], box[1]])
        text_bottom_right = str([box[2], box[3]])        
        draw.text((box_ori[0] - 30 , box_ori[1] -20), text_top_left, font=font, fill=text_top_left_color)
        draw.text((box_ori[2] - 30, box_ori[3] + 5), text_bottom_right, font=font, fill=text_bottom_right_color)        
        
        draw.rectangle([box_ori[0] - point_size, box_ori[1] - point_size, box_ori[0] + point_size, box_ori[1] + point_size], fill=text_top_left_color)
        draw.rectangle([box_ori[2] - point_size, box_ori[3] - point_size, box_ori[2] + point_size, box_ori[3] + point_size], fill=text_bottom_right_color)
    return img



def handle_conv_bbox(conversations: List[str]):
    coordinates = []
    for conv in conversations:
        v = conv['value']
        matches = re.findall(pattern, v)
        if matches != []:
            float_list = list(map(float, matches[-1].split(',')))
            if float_list not in coordinates:
                coordinates.append(float_list)
    return coordinates


def handle_conv_bbox_var(conversations: List[str]):
    # 返回referring里的bbox(问题)
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
        # return re_coordinates[0]
        return random.choice(re_coordinates)
    return re_coordinates


def handle_conv_bbox_var_grd(conversations: List[str]):
    # 返回grounding结果里的bbox（答案）
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


