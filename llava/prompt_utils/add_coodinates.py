# import debugpy; debugpy.connect(('10.140.12.33', 5678))
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import matplotlib.patches as patches

if __name__ == "__main__":
    img_path = "2391806.jpg"
    img = Image.open(img_path)
    # 4. 设置x轴和y轴的刻度
# 注意：这里我们使用图像的像素坐标，而不是归一化坐标
    # plt.xticks(np.arange(0, img.width, 100))  # x轴刻度，每隔100像素一个刻度
    # plt.yticks(np.arange(0, img.height, 100))  # y轴刻度，每隔100像素一个刻度
    # 5. 设置归一化x轴和y轴的刻度
    plt.xticks(np.linspace(0, img.width-1, 11), [f"{i/10:.2f}" for i in range(0,11,1)])  # 从0到图像宽度-1，共11个刻度
    plt.yticks(np.linspace(0, img.height-1, 11), [f"{i/10:.2f}" for i in range(0,11,1)])  # 从0到图像高度-1，共11个刻度

    plt.gca().invert_yaxis()  # 反转y轴，使得原点在左上角
    plt.gca().xaxis.tick_top()  # x轴刻度在上方
    plt.tight_layout()
    #########################################
    # 假设的左上角和右下角坐标
    bbox_left = 120  # 左上角 x 坐标（归一化坐标，0 表示最左边）
    bbox_top = 100   # 左上角 y 坐标（归一化坐标，1 表示最顶部）
    bbox_right = 320  # 右下角 x 坐标
    bbox_bottom = 360  # 右下角 y 坐标


    # 创建矩形边界框
    rect = patches.Rectangle((bbox_left, bbox_top), bbox_right - bbox_left, bbox_bottom - bbox_top,
                            linewidth=2, edgecolor='w', facecolor='none')
    x = [bbox_left, bbox_right]
    y = [bbox_top, bbox_bottom]
    # 偏移量
    offset_x = 5
    offset_y = -5
    plt.scatter(x, y, c='r')
    texts = [f'{round((bbox_left-1)/img.width,2), round((bbox_top-1)/img.height,2)}', f'{round((bbox_right-1)/img.width, 2), round((bbox_bottom-1)/img.height, 2)}']
    # 在每个散点旁显示文本
    for i, txt in enumerate(texts):
        plt.text(x[i]+offset_x, y[i]+offset_y, txt, fontsize=12, ha='left', va='bottom', color='w')

    # 将矩形添加到当前坐标轴
    plt.gca().add_patch(rect)
    # 在矩形边框的左上角添加文本
    # plt.text(bbox_left, bbox_top, 'BBox Label', fontsize=12, ha='left', va='top')

    #########################################
    plt.imshow(img)
    plt.savefig("test.jpg")
    print()
