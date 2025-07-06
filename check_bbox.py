from PIL import Image, ImageDraw

def draw_bbox_on_image(image_path, bbox):
    # 读取图像
    image = Image.open(image_path)
    
    # 获取图像的原始宽度和高度
    original_width, original_height = image.size
    
    # 计算填充后的正方形尺寸
    max_dim = max(original_width, original_height)
    
    # 计算填充后的偏移量
    offset_x = (max_dim - original_width) // 2
    offset_y = (max_dim - original_height) // 2
    
    # 将归一化坐标转换为填充后的正方形图像的实际坐标
    x_min = int(bbox[0] * max_dim)
    y_min = int(bbox[1] * max_dim)
    x_max = int(bbox[2] * max_dim)
    y_max = int(bbox[3] * max_dim)
    
    # 将填充后的坐标还原为原始图像的坐标
    x_min = max(0, x_min - offset_x)
    y_min = max(0, y_min - offset_y)
    x_max = min(original_width, x_max - offset_x)
    y_max = min(original_height, y_max - offset_y)
    
    # 创建一个可绘制的对象
    draw = ImageDraw.Draw(image)
    
    # 绘制边界框
    draw.rectangle([x_min, y_min, x_max, y_max], outline=(4,255,14), width=4)
    
    return image

# 示例使用
image_path = "2389241.jpg"  # 替换为你的图像路径
bbox = [0.27, 0.59, 0.53, 0.71]  # 替换为你的归一化 bbox 坐标

# 绘制边界框并保存图像
result_image = draw_bbox_on_image(image_path, bbox)
result_image.save("image_with_bbox.png")  # 保存图像