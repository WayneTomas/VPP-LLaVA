# import debugpy; debugpy.connect(('10.140.12.33', 5678))
import json
import torch
import torch.nn.functional as F
import argparse
from torchvision.ops.boxes import box_area
from PIL import Image
import copy


image_mean = [
    0.48145466,
    0.4578275,
    0.40821073
  ]
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

def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True, default="")
    args = parser.parse_args()
    # import pdb; pdb.set_trace()
    correct = total_cnt = 0
    for line_idx, line in enumerate(open(args.src)):
        res = json.loads(line)
        try:
            predict_bbox = [float(num) for num in res['text'].strip('[]').split(', ')]
        except:
            predict_bbox = [0.0, 0.0, 0.0, 0.0]
        gt_bbox = res['gt_bbox']

        pred_bbox = torch.tensor(predict_bbox)
        tmp = copy.deepcopy(pred_bbox)
        target_bbox = torch.tensor(gt_bbox)
        
        # pred_bbox = torch.tensor(resize_bbox(pred_bbox, res['hw'][1], res['hw'][0]))
        # import pdb; pdb.set_trace()
        h, w = res['hw']
        # orig
        if h > w:
            pred_bbox[0::2] *= h
            pred_bbox[0::2] -= (h - w) // 2
            pred_bbox[1::2] *= h
            pred_bbox = F.relu(pred_bbox)
        elif h < w:
            pred_bbox[0::2] *= w
            pred_bbox[1::2] *= w
            pred_bbox[1::2] -= (w - h) // 2
            pred_bbox = F.relu(pred_bbox)
        elif h == w:
            pred_bbox[0::2] *= w
            pred_bbox[1::2] *= h
        
        try:
            iou, _ = box_iou(pred_bbox[None], target_bbox[None])
            iou = iou.item()
        except:
            iou = 0
            print(tmp[None])
        total_cnt += 1
        if iou >= 0.5:
            correct += 1

    print(f'Precision @ 1: {correct / total_cnt} \n')
    
    
