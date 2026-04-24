from __future__ import annotations

import warnings

import torch
from PIL import Image

from .base import BaseModel
from ._tablemoe_common import (
    ensure_deepseek_vl2_repo,
    get_model_device,
    move_batch_to_device,
    resolve_torch_dtype,
)
from ..smp import *


class DeepSeekVL2(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = True

    def check_install(self):
        ensure_deepseek_vl2_repo()

    def _init_common(self, model_path='deepseek-ai/deepseek-vl2', **kwargs):
        self.check_install()
        assert model_path is not None
        self.model_path = model_path

        DeepseekVLV2Processor, _, load_pil_images = ensure_deepseek_vl2_repo()
        self.vl_chat_processor = DeepseekVLV2Processor.from_pretrained(model_path)
        self.tokenizer = self.vl_chat_processor.tokenizer
        self._load_pil_images = load_pil_images

        torch.cuda.empty_cache()
        default_kwargs = dict(max_new_tokens=2048, do_sample=False, use_cache=True)
        default_kwargs.update(kwargs)
        self.kwargs = default_kwargs
        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as generation config. ')

    def _set_model(self, model, requested_dtype=None):
        self.model = model.eval()
        self.model_device = get_model_device(self.model)
        inferred_dtype = getattr(self.model, 'dtype', None)
        self.model_dtype = requested_dtype if requested_dtype is not None else inferred_dtype
        torch.cuda.empty_cache()

    def __init__(
        self,
        model_path='deepseek-ai/deepseek-vl2',
        torch_dtype='bf16',
        attn_implementation=None,
        device_map=None,
        **kwargs,
    ):
        self._init_common(model_path=model_path, **kwargs)

        _, DeepseekVLV2ForCausalLM, _ = ensure_deepseek_vl2_repo()
        resolved_dtype = resolve_torch_dtype(torch_dtype)

        model_kwargs = {}
        if resolved_dtype is not None:
            model_kwargs['torch_dtype'] = resolved_dtype
        if attn_implementation is not None:
            model_kwargs['attn_implementation'] = attn_implementation
        if device_map is not None:
            model_kwargs['device_map'] = device_map
        elif torch.cuda.is_available():
            model_kwargs['device_map'] = 'auto'

        model = DeepseekVLV2ForCausalLM.from_pretrained(model_path, **model_kwargs)
        self._set_model(model, requested_dtype=resolved_dtype)

    def prepare_inputs(self, message, dataset=None):

        if dataset == 'MMMU_DEV_VAL':

            def prepare_itlist(msgs):
                content, images = '', []
                image_idx = 1
                for s in msgs:
                    if s['type'] == 'image':
                        images.append(s['value'])
                        content += f'<image {image_idx}>'
                        image_idx += 1
                    elif s['type'] == 'text':
                        content += s['value']
                content = '<image>' * (image_idx - 1) + '\n' + content
                return content, images

            conversation = []
            if 'role' not in message[0]:
                content, images = prepare_itlist(message)
                content = content.replace(
                    'Please select the correct answer from the options above.',
                    "Answer with the option's letter from the given choices directly. Answer the question using a single word or phrase.\n"  # noqa
                )
                content = content.replace('Question:', "")
                content = content.replace('Options:\n', "")
                conversation.append(dict(role='<|User|>', content=content, images=images))
            else:
                role_map = {'user': '<|User|>', 'assistant': '<|Assistant|>'}
                for msgs in message:
                    role = role_map[msgs['role']]
                    content, images = prepare_itlist(msgs['content'])
                    content = content.replace(
                        'Please select the correct answer from the options above.',
                        "Answer with the option's letter from the given choices directly. Answer the question using a single word or phrase.\n"  # noqa
                    )
                    content = content.replace('Question:', "")
                    content = content.replace('Options:\n', "")
                    conversation.append(dict(role=role, content=content, images=images))
            conversation.append(dict(role='<|Assistant|>', content=''))

        else:

            def prepare_itlist(msgs):
                content, images = '', []
                for s in msgs:
                    if s['type'] == 'image':
                        images.append(s['value'])
                        content += '<image>\n'
                    elif s['type'] == 'text':
                        content += s['value']
                return content, images

            conversation = []
            if 'role' not in message[0]:
                content, images = prepare_itlist(message)
                conversation.append(dict(role='<|User|>', content=content, images=images))
            else:
                role_map = {'user': '<|User|>', 'assistant': '<|Assistant|>'}
                for msgs in message:
                    role = role_map[msgs['role']]
                    content, images = prepare_itlist(msgs['content'])
                    conversation.append(dict(role=role, content=content, images=images))
            conversation.append(dict(role='<|Assistant|>', content=''))

        return conversation

    def generate_inner(self, message, dataset=None):
        conversation = self.prepare_inputs(message, dataset)
        pil_images = self._load_pil_images(conversation)

        if dataset == 'MMMU_DEV_VAL' and len(pil_images):
            h, w = pil_images[0].size
            pil_images[0] = pil_images[0].resize((2 * h, 2 * w), Image.BILINEAR)

        prepare_inputs = self.vl_chat_processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True,
            system_prompt=""
        )
        prepare_inputs = move_batch_to_device(
            prepare_inputs,
            self.model_device,
            float_dtype=self.model_dtype,
        )
        inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)
        past_key_values = None

        outputs = self.model.generate(
            inputs_embeds=inputs_embeds,
            input_ids=prepare_inputs.input_ids,
            images=prepare_inputs.images,
            images_seq_mask=prepare_inputs.images_seq_mask,
            images_spatial_crop=prepare_inputs.images_spatial_crop,
            attention_mask=prepare_inputs.attention_mask,
            past_key_values=past_key_values,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **self.kwargs
        )

        answer = self.tokenizer.decode(
            outputs[0][len(prepare_inputs.input_ids[0]):].cpu().tolist(),
            skip_special_tokens=True
        )
        answer = answer.rstrip('.')

        return answer

    def chat_inner(self, message, dataset=None):
        return self.generate_inner(message, dataset=dataset)
