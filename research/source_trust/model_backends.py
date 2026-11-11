"""Frozen local Transformers backend with auditable candidate scoring."""
import hashlib
import json
from pathlib import Path
import time

import torch
import transformers
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor, GenerationConfig


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


INTERNVL_SYSTEM = ('你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。')
INTERNVL_SHORT_SUFFIX = 'Answer the question using a single word or phrase.'


def local_preprocessing(model_dir, pixels):
    """Resolve only the audited native architectures; no weights are loaded."""
    config = AutoConfig.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=False)
    if pixels not in (224, 448):
        raise ValueError('Supported image sizes are 224 and 448.')
    if config.model_type in ('qwen2_5_vl', 'qwen3_vl', 'qwen3_vl_moe'):
        settings = dict(min_pixels=pixels*pixels, max_pixels=pixels*pixels)
    elif config.model_type == 'internvl':
        if pixels != 448:
            raise ValueError('Audited native InternVL requires --pixels 448 and one tile.')
        settings = dict(crop_to_patches=False, min_patches=1, max_patches=1,
                        size=dict(height=448, width=448))
    else:
        raise ValueError('Unaudited model_type: ' + config.model_type)
    chat_policy = (dict(system=INTERNVL_SYSTEM, categorical_suffix=INTERNVL_SHORT_SUFFIX,
                        profile_suffix=None, categorical_short_answer=True,
                        profile_short_answer=False)
                   if config.model_type == 'internvl' else
                   dict(system='native_template_default', categorical_suffix=None,
                        profile_suffix=None, enable_thinking=False))
    return dict(model_type=config.model_type, architectures=config.architectures,
                pixels=pixels, image_processor_settings=settings, chat_policy=chat_policy)


class FrozenVLM:
    """Batch-one inference on one idle 3090; never downloads or trains."""
    def __init__(self, model_dir, model_id, pixels=224):
        preprocessing = local_preprocessing(model_dir, pixels)
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError('Expose exactly one idle RTX3090 with CUDA_VISIBLE_DEVICES.')
        if '3090' not in torch.cuda.get_device_name(0):
            raise RuntimeError('This development protocol requires an RTX3090.')
        free, total = torch.cuda.mem_get_info()
        if free < 16*1024**3:
            raise RuntimeError('Leave occupied GPUs alone; selected card needs16GiB free.')
        torch.manual_seed(20261111)
        self.processor = AutoProcessor.from_pretrained(
            str(model_dir), local_files_only=True, trust_remote_code=False,
            use_fast=False, **preprocessing['image_processor_settings'])
        self.model = AutoModelForImageTextToText.from_pretrained(
            str(model_dir), local_files_only=True, trust_remote_code=False,
            torch_dtype=torch.bfloat16, device_map={'': 0}, attn_implementation='sdpa').eval()
        head = self.model.get_output_embeddings()
        if not isinstance(head, torch.nn.Linear):
            raise RuntimeError('Audit the output head before scoring this architecture.')
        tied = head.weight.data_ptr() == self.model.get_input_embeddings().weight.data_ptr()
        projection = torch.nn.Linear(head.in_features, head.out_features,
                                     bias=head.bias is not None, device=head.weight.device,
                                     dtype=torch.float32)
        with torch.no_grad():
            projection.weight.copy_(head.weight)
            if head.bias is not None:
                projection.bias.copy_(head.bias)
        self.model.set_output_embeddings(projection)
        self.model.config.tie_word_embeddings = False
        projection.register_forward_pre_hook(lambda module, inputs: (inputs[0].float(),))
        del head
        # Checkpoint decoding defaults can penalize words already present in
        # history. Use neutral greedy decoding and score raw, unprocessed logits.
        original_generation = self.model.generation_config
        self.generation_config = GenerationConfig(
            do_sample=False, use_cache=True,
            bos_token_id=original_generation.bos_token_id,
            eos_token_id=original_generation.eos_token_id,
            pad_token_id=(original_generation.pad_token_id
                          if original_generation.pad_token_id is not None else
                          (original_generation.eos_token_id[0]
                           if isinstance(original_generation.eos_token_id, list)
                           else original_generation.eos_token_id)))
        self.template_kwargs = {'enable_thinking': False}
        self.metadata = dict(model_id=model_id, model_revision=Path(model_dir).name,
                             model_type=self.model.config.model_type,
                             model_class=type(self.model).__name__,
                             backbone_dtype='bfloat16', output_projection_dtype='float32',
                             original_output_head_tied=tied, pixels=pixels,
                             preprocessing=preprocessing,
                             native_prompt_policy=preprocessing['chat_policy'],
                             candidate_scoring='raw_generation_logits_before_processors',
                             repetition_penalty=self.generation_config.repetition_penalty,
                             chat_template_kwargs=self.template_kwargs,
                             torch=torch.__version__, transformers=transformers.__version__,
                             device=torch.cuda.get_device_name(0), total_vram_bytes=total)

    def prepare(self, content, images, short_answer=True):
        prompt = self.render_prompt(content, short_answer=short_answer)
        kwargs = dict(text=[prompt], return_tensors='pt', padding=True)
        if images:
            kwargs['images'] = images
        if images and self.metadata['model_type'] == 'internvl':
            kwargs.update(self.metadata['preprocessing']['image_processor_settings'])
        inputs = self.processor(**kwargs).to('cuda:0')
        if self.metadata['model_type'] == 'internvl':
            for key, value in inputs.items():
                if torch.is_floating_point(value):
                    inputs[key] = value.to(dtype=torch.bfloat16)
        if images and self.metadata['model_type'] == 'internvl':
            shape = tuple(inputs['pixel_values'].shape)
            if shape != (len(images), 3, 448, 448):
                raise RuntimeError('InternVL requires exactly one 448px tile per image; got ' + str(shape))
        return prompt, inputs

    def candidate_ids(self, labels):
        tokens = {label:self.processor.tokenizer.encode(label, add_special_tokens=False)
                  for label in labels}
        if any(len(ids) != 1 for ids in tokens.values()):
            raise ValueError('This protocol needs single-token candidates; do not approximate multi-token scores.')
        return {label:ids[0] for label,ids in tokens.items()}

    def render_prompt(self, content, short_answer=True):
        messages = []
        if self.metadata['model_type'] == 'internvl':
            messages.append({'role': 'system', 'content': INTERNVL_SYSTEM})
            if short_answer:
                content = list(content) + [{'type': 'text', 'text': '\n' + INTERNVL_SHORT_SUFFIX}]
        messages.append({'role': 'user', 'content': content})
        return self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **self.template_kwargs)

    @torch.inference_mode()
    def score(self, rendered, labels, max_new_tokens=8):
        ids = self.candidate_ids(labels)
        prompt, inputs = self.prepare(rendered['content'], rendered['images'])
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.monotonic()
        result = self.model.generate(**inputs, generation_config=self.generation_config,
                                     max_new_tokens=max_new_tokens,
                                     return_dict_in_generate=True, output_logits=True)
        torch.cuda.synchronize()
        elapsed = time.monotonic()-started
        first_scores = result.logits[0][0]
        logits = first_scores.float()
        full_prob = logits.softmax(-1)
        candidate_logits = torch.stack([logits[ids[label]] for label in labels])
        probabilities = candidate_logits.softmax(-1).tolist()
        generation = result.sequences[0, inputs['input_ids'].shape[1]:]
        answer = self.processor.tokenizer.decode(generation, skip_special_tokens=True).strip()
        row = dict(probabilities=dict(zip(labels,probabilities)),
                   logits={label:float(logits[ids[label]]) for label in labels},
                   canonical_candidate_mass=float(sum(full_prob[ids[label]] for label in labels)),
                   greedy_answer=answer, generated_tokens=int(generation.numel()),
                   input_tokens=int(inputs['input_ids'].shape[1]),
                   rendered_prompt=prompt, prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                   score_tensor_dtype=str(first_scores.dtype), elapsed_seconds=elapsed,
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                   peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                   candidate_token_ids=ids)
        del result, inputs
        return row

    @torch.inference_mode()
    def generate_text(self, prompt_text, max_new_tokens=96, short_answer=False):
        prompt, inputs = self.prepare([{'type':'text','text':prompt_text}], [],
                                     short_answer=short_answer)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.monotonic()
        result = self.model.generate(**inputs, generation_config=self.generation_config,
                                     max_new_tokens=max_new_tokens)
        torch.cuda.synchronize()
        generation = result[0, inputs['input_ids'].shape[1]:]
        row = dict(text=self.processor.tokenizer.decode(generation, skip_special_tokens=True).strip(),
                   generated_tokens=int(generation.numel()), input_tokens=int(inputs['input_ids'].shape[1]),
                   rendered_prompt=prompt, prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                   elapsed_seconds=time.monotonic()-started,
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                   peak_reserved_bytes=torch.cuda.max_memory_reserved())
        del result, inputs
        return row
