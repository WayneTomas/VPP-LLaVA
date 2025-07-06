# import debugpy; debugpy.connect(('22.9.35.97', 5673))
import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math
import copy
from llava.model.multimodal_encoder.detr_queries.misc import NestedTensor


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
    

def process_images(images, image_processor, model_cfg):
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    new_images = []
    if image_aspect_ratio == 'pad':
        for image in images:
            image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
            image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            new_images.append(image)
    else:
        return image_processor(images, return_tensors='pt')['pixel_values']
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, extra_image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.extra_image_processor = extra_image_processor

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        
        extra_str = ". Each image is accompanied by axes. If the question pertains to the bounding box coordinates, refer to the axes for the response."
        qs = "Please provide the bounding box coordinate of the region this sentence describes: " + line["sent"] + extra_str

        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        
        output = self.extra_image_processor.preprocess(copy.deepcopy(image), target={})

        extra_image = output['image']
        extra_mask = output['mask']
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor, image.size, extra_image, extra_mask

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes, extra_image, extra_mask= zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    extra_image = torch.stack(extra_image)
    extra_mask = torch.stack(extra_mask)
    return input_ids, image_tensors, image_sizes, extra_image, extra_mask


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, extra_image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, extra_image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    model = model.to(dtype=torch.bfloat16, device='cuda')

    ##########################################################
    # written by wayne
    # add extra_vision_tower and extra_processor
    extra_vision_tower = model.get_extra_vision_tower()
    extra_vision_tower.to(dtype=torch.bfloat16, device='cuda')

    extra_image_processor = extra_vision_tower.image_processor
    ##########################################################

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, extra_image_processor, model.config)

    for (input_ids, image_tensor, image_sizes, extra_image, extra_mask), line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = line["image"].split("/")[-1]
        cur_prompt = "Please provide the bounding box coordinate of the region this sentence describes: " + line["sent"]
        gt_bbox = line["bbox"]
        w, h = line['width'], line['height']
        input_ids = input_ids.to(device='cuda', non_blocking=True)

        with torch.inference_mode():
            output_ids = model.generate_custom(
                input_ids,
                images=image_tensor.to(dtype=torch.bfloat16, device='cuda', non_blocking=True),
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                extra_image=extra_image.to(dtype=torch.bfloat16, device='cuda'),
                extra_mask=extra_mask.to(device='cuda')
                )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   'gt_bbox': gt_bbox,
                                   'hw': [h, w],
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        # ans_file.flush()
    ans_file.close()


def visual_visual_prompt(model):
    import numpy as np
    visual_prompt = (model.get_model().visual_prompt - model.get_model().visual_prompt.min()) / (model.get_model().visual_prompt.max() - model.get_model().visual_prompt.min())
    normed_mask = (visual_prompt.detach().to(torch.float32) * 255).cpu().numpy().astype('uint8')
    array = np.transpose(normed_mask, (1, 2, 0))
    image = Image.fromarray(array)
    image.save('axis_not_trained.jpg')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    eval_model(args)
