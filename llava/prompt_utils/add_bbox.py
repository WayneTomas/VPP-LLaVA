import debugpy; debugpy.connect(('10.140.12.33', 5678))
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.patches as patches

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


if __name__ == "__main__":
    img_path = "/mnt/petrelfs/sunqiao/tangwei/codes/LLaVA/COCO_train2014_000000537553.jpg"
    img = Image.open(img_path)
    

    print()
