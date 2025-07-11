import os, json
from json import JSONDecodeError
import base64
from io import BytesIO
import mimetypes
from tqdm import tqdm
import numpy as np
import torch
import torchvision
from PIL import Image

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


# SYSTEM_PROMPT = (
#     "You are an image-captioning assistant. "
#     "Return *exactly* the following JSON schema and nothing else:\n"
#     '{ "brief_caption": "", "detailed_caption": "" }'
# )

USER_PROMPT_ENGLISH = (
    "Give the text caption of the right object, which is a component of the left object and highlighted by red box.\n"
    "Fill in the JSON so that:\n"
    "• brief_caption – one concise sentence (≤15 words)\n"
    "• detailed_caption – describe material, shape, structure\n"
    "Return only the JSON object."
)

# SYSTEM_PROMPT_CHINESE = (
#     "你是一个图像标注助手。"
#     "请返回*完全相同*的以下JSON格式，并且不要返回其他内容：\n"
#     '{ "brief_caption": "", "detailed_caption": "" }'
# )

USER_PROMPT_CHINESE = (
    "### 假如你是一个图像标注专家，你将根据用户提供的包含左右两个物体（右物体是左物体的一个组件且被红框突出显示）的场景，"
    "来解决为右物体生成**英文**文本描述的任务。根据以下规则一步步执行：\n"
    "1. 生成一个简洁描述，仅用一句话。\n"
    "2. 生成一个详细描述，描述其颜色、材料、形状以及属于左物体的哪一部分。\n"
    "请回答问题：用户提供的包含左右两个物体（右物体是左物体的一个组件且被红框突出显示）的场景，为右物体生成文本描述\n"
    '输出：\n要求：\n1 按照**JSON格式**输出\n2 **JSON格式**内容为：{ "brief_caption": "", "detailed_caption": "" }\n'
    '3 **但caption中不要使用“左物体”、“右物体”这种代称，而是使用具体的物体类别名称指代**\n ###'
)

def load_image(image_file):
    image = Image.open(image_file).convert('RGB')
    return image

@torch.no_grad()
def draw_bbox2d(bbox2d, raw_img, part_img, arrow_img=None):
    boxes_draw = torch.tensor(bbox2d).unsqueeze(0) # (1, 4)
    # canvas = torch.zeros((3, 512, 512), dtype=torch.uint8)
    raw_img = torch.as_tensor(np.array(raw_img)).permute(2, 0, 1)
    part_img = torch.as_tensor(np.array(part_img)).permute(2, 0, 1)
    colors = [255, 0, 0]
    bbox2d_render = torchvision.utils.draw_bounding_boxes(raw_img, boxes_draw, colors=colors, width=3) if boxes_draw[0][0] != -1 else raw_img
    img = torchvision.utils.make_grid([bbox2d_render, part_img], nrow=2, padding=4, pad_value=255)
    # img = torch.cat([bbox2d_render, part_img], dim=2)
    if arrow_img is not None:
        img = img.int()
        arrow_img = torch.as_tensor(np.array(arrow_img)).permute(2, 0, 1)
        _, H, W = img.shape
        _, h, w = arrow_img.shape
        top = (H - h) // 2
        left = (W - w) // 2
        img[:, top:top+h, left:left+w] += arrow_img.int()
        img = img.clamp(0, 255).to(torch.uint8)
    img = torchvision.transforms.ToPILImage()(img)
    return img

def encode_image_to_base64(pil_image) -> str:
    """Convert a PIL image to a base64-encoded string."""
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    mime = mimetypes.guess_type("image.png")[0] or "application/octet-stream"
    img_str = f"data:{mime};base64,{img_str}"
    return img_str

def check_string_json(s):
    """check json format string, ensure all {} and double quotes are paired"""
    if not s:
        return '{}'
    s = s.replace("```json", "").replace("```", "").strip()
    if not s.startswith('{'):
        s = '{' + s
    if not s.endswith('}'):
        s = s + '}'
    brace_count = 0
    for char in s:
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
    if brace_count > 0:
        s = s + '}' * brace_count
    elif brace_count < 0:
        s = '{' * abs(brace_count) + s
    quote_count = 0
    i = 0
    while i < len(s):
        if s[i] == '"' and (i == 0 or s[i-1] != '\\'):
            quote_count += 1
        i += 1
    if quote_count % 2 != 0:
        s = s + '"'
    return s

@torch.no_grad()
def main(args):
    raw_img_root = args.raw_img_root
    part_img_root = args.part_img_root
    info_file = args.info_file
    output_file = args.output_file
    vlm_ckpt_dir = args.vlm_ckpt_dir

    with open(info_file, 'r') as f:
        ins_infos = json.load(f)
    ins_list = list(ins_infos.keys())
    ins_list.sort()
    print('Number of instances:', len(ins_list))
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            prompt_dict = json.load(f)
    else:
        prompt_dict = {}

    arrow_img = Image.open('debug/red_arrow.png').convert('RGB')

    # init model
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        vlm_ckpt_dir,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2", # install flash-attn-2 if you want to use it
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(vlm_ckpt_dir, local_files_only=True)

    try:
        for count, ins in enumerate(tqdm(ins_list, total=len(ins_list))):
            raw_img_dir = os.path.join(raw_img_root, ins)
            part_img_ins_dir = os.path.join(part_img_root, ins)
            part_id = os.listdir(part_img_ins_dir)
            num_parts = len(part_id)
            caption_ins = {}
            for i in part_id:
                if ins in prompt_dict and i in prompt_dict[ins]:
                    caption_ins[i] = prompt_dict[ins][i]
                    continue
                max_visible_view_id = ins_infos[ins][i]['max_visible_view_id']
                raw_img_path = os.path.join(
                    raw_img_dir, '{:03d}.png'.format(max_visible_view_id))
                part_img_path = os.path.join(
                    part_img_ins_dir, i,
                    '{:03d}.png'.format(max_visible_view_id))
                if not os.path.exists(raw_img_path) or not os.path.exists(
                        part_img_path):
                    print(
                        f"Skipping instance {ins}, part {i}: missing image files."
                    )
                    continue
                # process image
                raw_img = load_image(raw_img_path)
                part_img = load_image(part_img_path)
                bbox2d = ins_infos[ins][i]['bbox2d_max_visible']
                img = draw_bbox2d(bbox2d, raw_img, part_img, arrow_img)
                # img.save(os.path.join('debug/debug_vlm', f"{ins}_{i}_max_v{max_visible_view_id}.png")) # for debugging
                img_str = encode_image_to_base64(img)

                messages = [
                    # {"role": "system", "content": SYSTEM_PROMPT_CHINESE},
                    {
                        "role":
                        "user",
                        "content": [
                            {
                                "type": "image",
                                "image": img_str,
                            },
                            {
                                "type": "text",
                                "text": USER_PROMPT_CHINESE
                            },
                        ],
                    }
                ]
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = inputs.to(model.device)

                # Inference: Generation of the output
                generated_ids = model.generate(**inputs, max_new_tokens=256)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):]
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False)
                try:
                    output_text_ = output_text[0].replace(
                        "```json", "").replace("```", "").strip().strip('\n')
                    output_text_ = check_string_json(output_text_)
                    caption = json.loads(output_text_)
                except JSONDecodeError as e:
                    print(
                        f"Error decoding JSON for instance {ins}, part {i}: \n{output_text}"
                    )
                caption_ins[i] = [
                    caption['brief_caption'], caption['detailed_caption']
                ]
            prompt_dict[ins] = caption_ins
            if count % 100 == 99:
                print(f"Processed {count} instances, saving progress...")
                with open(output_file, 'w') as f:
                    json.dump(prompt_dict, f, indent=4)
    except Exception as e:
        raise e
    finally:
        print("Saving prompts to file...")
        with open(output_file, 'w') as f:
            json.dump(prompt_dict, f, indent=4)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate captions for parts of objects.")
    parser.add_argument('--raw_img_root', type=str, required=True, help='Root directory for full object rendered images.')
    parser.add_argument('--part_img_root', type=str, required=True, help='Root directory for part images.')
    parser.add_argument('--info_file', type=str, required=True, help='Path to instance max visible information file generated by get_infos.py.')
    parser.add_argument('--output_file', type=str, required=True, help='Path to the output JSON file for captions.')
    parser.add_argument('--vlm_ckpt_dir', type=str, required=True, help='Directory containing the model checkpoint.')
    args = parser.parse_args()
    main(args)
