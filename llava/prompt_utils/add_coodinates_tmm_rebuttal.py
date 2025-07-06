# import debugpy; debugpy.connect(('10.140.12.33', 5678))
# import matplotlib.pyplot as plt
# from PIL import Image
# import numpy as np
# import matplotlib.patches as patches

# if __name__ == "__main__":
#     img_path = "2391806.jpg"
#     img = Image.open(img_path)
#     img = img.resize((336, 336))
#     # 4. 设置x轴和y轴的刻度
# # 注意：这里我们使用图像的像素坐标，而不是归一化坐标
#     # plt.xticks(np.arange(0, img.width, 100))  # x轴刻度，每隔100像素一个刻度
#     # plt.yticks(np.arange(0, img.height, 100))  # y轴刻度，每隔100像素一个刻度
#     # 5. 设置归一化x轴和y轴的刻度
#     plt.figure(figsize=(336/100, 336/100))
#     plt.imshow(img)
#     plt.xticks(np.linspace(0, 336-1, 11), [f"{i/10:.1f}" for i in range(0,11,1)])  # 从0到图像宽度-1，共11个刻度
#     plt.yticks(np.linspace(0, 336-1, 11), [f"{i/10:.1f}" for i in range(0,11,1)])  # 从0到图像高度-1，共11个刻度

#     plt.gca().invert_yaxis()  # 反转y轴，使得原点在左上角
#         # 显示四周的刻度和刻度标签
#     ax = plt.gca()
#     ax.spines['top'].set_visible(True)
#     ax.spines['right'].set_visible(True)
#     ax.spines['bottom'].set_visible(True)
#     ax.spines['left'].set_visible(True)

#     # 在四个边框上显示刻度
#     ax.xaxis.set_ticks_position('both')  # 在上下边框上显示x轴刻度
#     ax.yaxis.set_ticks_position('both')  # 在左右边框上显示y轴刻度

#     # 在四个边框上显示刻度标签
#     ax.xaxis.set_label_position('top')  # 将x轴标签放在上方
#     ax.yaxis.set_label_position('right')  # 将y轴标签放在右侧

#     # 调整刻度标签的位置
#     ax.xaxis.set_tick_params(labeltop=True, labelbottom=True)
#     ax.yaxis.set_tick_params(labelright=True, labelleft=True)

#     # plt.gca().xaxis.tick_top()  # x轴刻度在上方
#     plt.tight_layout()
#     #########################################
#     # plt.imshow(img)
#     plt.savefig("test.jpg")
#     print()

    # plt.subplots_adjust(left=0.089, right=0.911, top=0.911, bottom=0.089)


from PIL import Image, ImageDraw, ImageFont, ImageStat
import numpy as np
import json
from typing import List
import re
import random


img = Image.new('RGB', (336, 336), 'white')
font_path = "TIMES.TTF"  # 例如："arial.ttf"
font_size = 10
font = ImageFont.truetype(font_path, font_size)
point_size = 1

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

    # 以下draw.line里的坐标都是x1y1x2y2的形式(start point, end point)
    # 绘制X轴（上侧）
    draw.line((30, 30, width-30, 30), fill='black', width=tick_width)
    # 绘制Y轴（左侧）
    draw.line((30, height - 30, 30, 30), fill='black', width=tick_width)
    # 绘制X轴（下侧）
    draw.line((30, height - 30, width-30, height - 30), fill='black', width=tick_width)
    # 绘制Y轴（右侧）
    draw.line((width - 30, 30, width - 30, height-30), fill='black', width=tick_width)

    # 绘制X轴刻度（上侧）
    for x in range(11):  # 0到10，共11个刻度
        tick_position = int(x * ((width-60) / 10))
        axis_color = 'black'

        # 文本位置在刻度的正上方，根据文本宽度调整偏移
        text = str(f"{x/10:.1f}")
        text_offset = tick_position - 10  # 偏移量，可根据需要调整
        
        draw.line((tick_position+30, 20, tick_position+30, -tick_length+20), fill=axis_color, width=tick_width)
        draw.line((tick_position+30, height - 30, tick_position+30, height - 30 - tick_length), fill=axis_color, width=tick_width)
        draw.text((text_offset + 30+5, -tick_length - 5), text, font=font, fill=axis_color)
        draw.text((text_offset + 30+5, height - 30 - tick_length + 5), text, font=font, fill=axis_color)

    # 绘制Y轴刻度（左侧）
    for y in range(11):  # 0到10，共11个刻度
        tick_position = int(y * ((height - 60) / 10))
        text = str(f"{y/10:.1f}")
        text_offset = tick_position - 10  # 偏移量，可根据需要调整

        # 绘制 y 轴刻度
        draw.line((30, tick_position + 30, 30 + tick_length, tick_position + 30), fill=axis_color, width=tick_width)
        draw.text((5, text_offset + 30 + 5), text, font=font, fill=axis_color)
        
        draw.line((306, tick_position + 30, 306 - tick_length , tick_position + 30), fill=axis_color, width=tick_width)
        draw.text((336-15, text_offset + 30 + 5), text, font=font, fill=axis_color)
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
    img_path = "2391806.jpg"
    img = Image.new('RGB', (336, 336), 'white')
    axised_img = add_axis_pil_double(img, save_path="")
    axised_img.save("test.png")