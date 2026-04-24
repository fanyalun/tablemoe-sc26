import ast
import base64
import hashlib
import io
import os

import pandas as pd
from PIL import Image


def _is_valid(val):
    return pd.notna(val) and str(val).strip() != ""


def _safe_text(val, default=""):
    if _is_valid(val):
        return str(val)
    return default


def dump_image(line, img_root):
    os.makedirs(img_root, exist_ok=True)
    output_paths = []

    image_paths = []
    if "image_path" in line and _is_valid(line["image_path"]):
        raw = line["image_path"]
        if isinstance(raw, list):
            image_paths = [str(x) for x in raw]
        else:
            image_paths = [str(raw)]

    def _resolve_path(path_str):
        if os.path.isabs(path_str) and os.path.exists(path_str):
            return path_str
        return os.path.join(img_root, path_str)

    if "image" in line and _is_valid(line["image"]):
        val = line["image"]

        def save_img(img_val, idx=0):
            try:
                img_bytes = None
                if isinstance(img_val, dict) and "bytes" in img_val:
                    img_bytes = img_val["bytes"]
                elif isinstance(img_val, bytes):
                    img_bytes = img_val
                elif isinstance(img_val, str):
                    raw_s = img_val
                    s = raw_s
                    if s.startswith("data:image"):
                        s = s.split(",", 1)[1]
                    try:
                        img_bytes = base64.b64decode(s)
                    except Exception:
                        return _resolve_path(raw_s)

                if img_bytes is None:
                    return None

                if idx < len(image_paths):
                    filename = os.path.basename(image_paths[idx])
                    save_path = os.path.join(img_root, filename)
                else:
                    img_hash = hashlib.md5(img_bytes).hexdigest()
                    save_path = os.path.join(img_root, f"{img_hash}.png")

                if not os.path.exists(save_path):
                    try:
                        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                        image.save(save_path)
                    except Exception:
                        if isinstance(img_val, str):
                            return _resolve_path(img_val)
                        raise
                return save_path
            except Exception as exc:
                print(f"Warning: failed to save image: {exc}")
                return None

        if isinstance(val, list):
            for idx, item in enumerate(val):
                path = save_img(item, idx)
                if path:
                    output_paths.append(path)
        else:
            path = save_img(val, 0)
            if path:
                output_paths.append(path)
    elif image_paths:
        for path in image_paths:
            output_paths.append(_resolve_path(path))

    return output_paths


def _extract_options(line):
    return {
        key: line.get(key)
        for key in ["A", "B", "C", "D", "E", "F"]
        if key in line and _is_valid(line.get(key))
    }


def _build_mcq_text(question, options=None, hint=None):
    prompt = ""
    if _is_valid(hint):
        prompt += f"Hint: {str(hint)}\n"
    prompt += f"Question: {question}\n"

    if options:
        prompt += "Options:\n"
        for key, val in options.items():
            if _is_valid(val):
                prompt += f"{key}. {str(val)}\n"
        prompt += "Answer with the option letter only."

    return prompt.rstrip()


def _build_yorn_text(question):
    return f"{question} Please answer yes or no."


def _build_vlmeval_message(image_paths, text):
    msgs = []
    if isinstance(image_paths, list):
        msgs.extend([{"type": "image", "value": p} for p in image_paths])
    elif image_paths:
        msgs.append({"type": "image", "value": image_paths})
    msgs.append({"type": "text", "value": text})
    return msgs


def _to_hf_chat_messages(vlmeval_msgs):
    content = []
    for item in vlmeval_msgs:
        if item["type"] == "image":
            content.append({"type": "image", "image": item["value"]})
        elif item["type"] == "text":
            content.append({"type": "text", "text": item["value"]})
    return [{"role": "user", "content": content}]


def build_mmbench_prompt_vlmeval(line, img_root):
    images = dump_image(line, img_root)
    text = _build_mcq_text(_safe_text(line.get("question")), options=_extract_options(line), hint=line.get("hint"))
    return _build_vlmeval_message(images, text)


def build_realworldqa_prompt_vlmeval(line, img_root):
    images = dump_image(line, img_root)
    text = _build_mcq_text(_safe_text(line.get("question")), options=_extract_options(line), hint=line.get("hint"))
    return _build_vlmeval_message(images, text)


def build_ai2d_prompt_vlmeval(line, img_root):
    images = dump_image(line, img_root)
    text = _build_mcq_text(_safe_text(line.get("question")), options=_extract_options(line))
    return _build_vlmeval_message(images, text)


def build_scienceqa_prompt_vlmeval(line, img_root):
    images = dump_image(line, img_root)
    text = _build_mcq_text(_safe_text(line.get("question")), options=_extract_options(line), hint=line.get("hint"))
    return _build_vlmeval_message(images, text)


def build_hallusionbench_prompt_vlmeval(line, img_root):
    images = dump_image(line, img_root)
    return _build_vlmeval_message(images, _build_yorn_text(_safe_text(line.get("question"))))


def build_mme_prompt_vlmeval(line, img_root):
    images = dump_image(line, img_root)
    return _build_vlmeval_message(images, _build_yorn_text(_safe_text(line.get("question"))))


def build_pope_prompt_vlmeval(line, img_root):
    images = dump_image(line, img_root)
    return _build_vlmeval_message(images, _build_yorn_text(_safe_text(line.get("question"))))


build_mmbench_prompt_qwen3vl = build_mmbench_prompt_vlmeval
build_realworldqa_prompt_qwen3vl = build_realworldqa_prompt_vlmeval
build_ai2d_prompt_qwen3vl = build_ai2d_prompt_vlmeval
build_scienceqa_prompt_qwen3vl = build_scienceqa_prompt_vlmeval
build_hallusionbench_prompt_qwen3vl = build_hallusionbench_prompt_vlmeval
build_mme_prompt_qwen3vl = build_mme_prompt_vlmeval
build_pope_prompt_qwen3vl = build_pope_prompt_vlmeval

build_mmbench_prompt_deepseekvl2 = build_mmbench_prompt_vlmeval
build_realworldqa_prompt_deepseekvl2 = build_realworldqa_prompt_vlmeval
build_ai2d_prompt_deepseekvl2 = build_ai2d_prompt_vlmeval
build_scienceqa_prompt_deepseekvl2 = build_scienceqa_prompt_vlmeval
build_hallusionbench_prompt_deepseekvl2 = build_hallusionbench_prompt_vlmeval
build_mme_prompt_deepseekvl2 = build_mme_prompt_vlmeval
build_pope_prompt_deepseekvl2 = build_pope_prompt_vlmeval


def build_mmbench_prompt(line, img_root):
    return _to_hf_chat_messages(build_mmbench_prompt_vlmeval(line, img_root))


def build_realworldqa_prompt(line, img_root):
    return _to_hf_chat_messages(build_realworldqa_prompt_vlmeval(line, img_root))


def build_ai2d_prompt(line, img_root):
    return _to_hf_chat_messages(build_ai2d_prompt_vlmeval(line, img_root))


def build_scienceqa_prompt(line, img_root):
    return _to_hf_chat_messages(build_scienceqa_prompt_vlmeval(line, img_root))


def build_hallusionbench_prompt(line, img_root):
    return _to_hf_chat_messages(build_hallusionbench_prompt_vlmeval(line, img_root))


def build_mme_prompt(line, img_root):
    return _to_hf_chat_messages(build_mme_prompt_vlmeval(line, img_root))


def build_pope_prompt(line, img_root):
    return _to_hf_chat_messages(build_pope_prompt_vlmeval(line, img_root))


def build_mathvision_prompt(line, img_root):
    images = dump_image(line, img_root)
    question = _safe_text(line.get("question"))
    options = {}
    if "choices" in line and _is_valid(line["choices"]):
        try:
            parsed = ast.literal_eval(str(line["choices"]))
            if isinstance(parsed, (list, tuple)):
                for idx, value in enumerate(parsed[:6]):
                    options[chr(ord("A") + idx)] = value
        except Exception:
            pass
    if not options:
        options = _extract_options(line)
    text = _build_mcq_text(question, options=options, hint=None)
    return _to_hf_chat_messages(_build_vlmeval_message(images, text))
