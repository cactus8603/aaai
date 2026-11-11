"""Frozen local Transformers backend with auditable candidate scoring."""
import hashlib
import json
import re
from pathlib import Path
import time

import torch
import transformers
from transformers import (AutoConfig, AutoImageProcessor, AutoModelForImageTextToText, AutoProcessor,
                          AutoTokenizer, GenerationConfig)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


INTERNVL_SYSTEM = ('你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。')
INTERNVL_SHORT_SUFFIX = 'Answer the question using a single word or phrase.'
# Gemma-3: SigLIP encoder at a fixed 896x896 (256 soft tokens per image). Pan-and-scan (extra crops for
# large/non-square images) is disabled so every image costs exactly one 896px view; our 224/448 renders
# are upsampled (bilinear, processor default) to 896. --pixels therefore sets the render resolution only.
GEMMA3_IMAGE = 896
COT_SUFFIX = "Think step by step, then give the final answer as 'Answer: <word>'."


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
    elif config.model_type == 'gemma3':
        settings = dict(do_pan_and_scan=False, size=dict(height=GEMMA3_IMAGE, width=GEMMA3_IMAGE))
        if getattr(config.get_text_config(), 'final_logit_softcapping', None) is not None:
            raise ValueError('Gemma-3 checkpoint with final logit soft-capping: audit raw-logit scoring first.')
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


def load_processor(model_dir, preprocessing):
    """Slow (reference) image processors everywhere. Gemma-3 ships a SentencePiece slow tokenizer that
    needs the sentencepiece package; its official tokenizer.json (fast) is used instead (same vocab)."""
    settings = preprocessing['image_processor_settings']
    if preprocessing['model_type'] == 'gemma3':
        image_processor = AutoImageProcessor.from_pretrained(
            str(model_dir), local_files_only=True, use_fast=False, **settings)
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, use_fast=True)
        processor = AutoProcessor.from_pretrained(
            str(model_dir), local_files_only=True, trust_remote_code=False,
            image_processor=image_processor, tokenizer=tokenizer)
        if processor.image_processor.do_pan_and_scan:
            raise RuntimeError('Gemma-3 pan-and-scan must stay disabled.')
        return processor
    return AutoProcessor.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=False, use_fast=False, **settings)


def split_reasoning(text):
    """Return (reasoning, final, closed). Thinking checkpoints may omit the opening <think> because the
    chat template already emitted it; everything before the last </think> is reasoning."""
    if '</think>' in text:
        head, _, final = text.rpartition('</think>')
        return head.replace('<think>', '').strip(), final, True
    return text.replace('<think>', '').strip(), '', False


# 'Answer:' marker, tolerant of markdown ('**Answer:**', 'Final answer :', full-width colon).
ANSWER_MARKER = re.compile(r'answer\W{0,3}\s*[:\uff1a]', flags=re.I)
_BOXED = re.compile(r'\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}')
_DECOR = ' \t\r\n*_"\'`.,!;:()[]<>'
PARSE_FAILED = 'parse_failed'


def parse_label(text, labels, cot=False, truncated=False):
    """Strictly map a free-text final answer to one label.

    Returns (label or None, status, char offset of the label inside `text` or None, failure reason or None).
    status is 'answer_marker', 'boxed', 'bare_label' or 'parse_failed'. Accepted forms:
      * an explicit 'Answer: <label>' marker whose clause (rest of that line) starts with the label and
        mentions no other label ('Answer: red, not blue' fails);
      * a unique \\boxed{<label>} answer (several boxes must all hold the same single label);
      * the whole text being a single bare label (decoration/punctuation allowed), not in combination.
    Every marker / box that yields a label must agree, the last marker must yield one, and any truncated
    generation fails. Nothing is ever guessed from the last mention in a sentence."""
    def fail(reason):
        return None, PARSE_FAILED, None, reason
    if truncated:
        return fail('truncated')
    if not text or not text.strip(_DECOR):
        return fail('no_answer')
    lower = {l.lower(): l for l in labels}
    pat = re.compile(r'(?<![A-Za-z])(' + '|'.join(re.escape(l) for l in labels) + r')(?![A-Za-z])', flags=re.I)

    def kinds(s):
        return {lower[m.group(1).lower()] for m in pat.finditer(s)}

    found = []                                   # (label, offset, source) of every well-formed answer
    for b in _BOXED.finditer(text):
        inner = re.sub(r'\\text\w*\{([^{}]*)\}', r'\1', b.group(1)).strip(_DECOR)
        if inner.lower() not in lower:
            return fail('boxed_not_single_label')
        found.append((lower[inner.lower()], b.start(1) + pat.search(b.group(1)).start(1), 'boxed'))
    markers = list(ANSWER_MARKER.finditer(text))
    for j, m in enumerate(markers):
        rest = text[m.end():]
        lead = len(rest) - len(rest.lstrip(' \t\r\n*_"\'`'))   # label may sit on the next line
        clause = rest[lead:].split('\n', 1)[0]
        first = pat.match(clause)
        if first and kinds(clause) == {lower[first.group(1).lower()]}:
            found.append((lower[first.group(1).lower()], m.end() + lead, 'answer_marker'))
        elif j == len(markers) - 1:
            return fail('conflicting_mentions' if len(kinds(clause)) > 1 else 'marker_without_label')
    if found:
        if len({l for l, _, _ in found}) > 1:
            return fail('conflicting_answers')
        pick = ([f for f in found if f[2] == 'answer_marker'] or found)[-1]
        return pick[0], pick[2], pick[1], None
    stripped = text.strip(_DECOR)
    if stripped.lower() in lower:
        return lower[stripped.lower()], 'bare_label', text.index(stripped), None
    mentioned = kinds(text)
    if len(mentioned) > 1:
        return fail('conflicting_mentions')
    if not mentioned:
        return fail('no_answer')
    return fail('no_answer_marker')


def parse_generation(raw_text, labels, mode, truncated):
    """Parse one decoded generation (special tokens kept, final EOS removed). Pure; no model needed.
    thinking: the answer is parsed from the text after the last </think> (no </think> -> parse_failed).
    cot     : the answer is parsed from the whole generation and needs an 'Answer:' marker / box / bare label.
    `answer_offset` is relative to raw_text. `closed`: the reasoning was closed by the model itself
    (</think> emitted / an 'Answer:' marker emitted)."""
    if mode == 'thinking':
        reasoning, final, closed = split_reasoning(raw_text)
        final_offset = len(raw_text) - len(final)
        if closed:
            answer, status, off, reason = parse_label(final, labels, truncated=truncated)
        else:
            answer, status, off, reason = None, PARSE_FAILED, None, (
                'truncated' if truncated else 'reasoning_not_closed')
        close_at = raw_text.rindex('</think>') if closed else None
    elif mode == 'cot':
        reasoning, final, final_offset = raw_text, raw_text, 0
        answer, status, off, reason = parse_label(raw_text, labels, cot=True, truncated=truncated)
        markers = list(ANSWER_MARKER.finditer(raw_text))
        closed = bool(markers)
        close_at = markers[-1].start() if closed else None
    else:
        raise ValueError(mode)
    return dict(reasoning=reasoning, final=final, final_offset=final_offset, closed=closed,
                close_at=close_at, answer=answer, answer_status=status, answer_parse_reason=reason,
                answer_offset=None if off is None else final_offset + off)


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
        self.processor = load_processor(model_dir, preprocessing)
        self.model = AutoModelForImageTextToText.from_pretrained(
            str(model_dir), local_files_only=True, trust_remote_code=False,
            torch_dtype=torch.bfloat16, device_map={'': 0}, attn_implementation='sdpa').eval()
        head = self.model.get_output_embeddings()
        if not isinstance(head, torch.nn.Linear):
            raise RuntimeError('Audit the output head before scoring this architecture.')
        tied = head.weight.data_ptr() == self.model.get_input_embeddings().weight.data_ptr()
        device = head.weight.device
        # Stage the fp32 copy on the CPU and free an untied bf16 head first, so the peak is
        # weights + one fp32 head (lets an 8B model + fp32 head fit next to other users' processes).
        with torch.no_grad():
            weight32 = head.weight.detach().to('cpu', torch.float32)
            bias32 = None if head.bias is None else head.bias.detach().to('cpu', torch.float32)
            if not tied:
                head.weight.data = torch.empty(0, dtype=head.weight.dtype, device=device)
                torch.cuda.empty_cache()
            projection = torch.nn.Linear(head.in_features, head.out_features,
                                         bias=bias32 is not None, device=device, dtype=torch.float32)
            projection.weight.copy_(weight32)
            if bias32 is not None:
                projection.bias.copy_(bias32)
            del weight32, bias32
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

    def prepare(self, content, images, short_answer=True, template_kwargs=None):
        prompt = self.render_prompt(content, short_answer=short_answer, template_kwargs=template_kwargs)
        kwargs = dict(text=[prompt], return_tensors='pt', padding=True)
        if images:
            kwargs['images'] = images
        if images and self.metadata['model_type'] == 'internvl':
            kwargs.update(self.metadata['preprocessing']['image_processor_settings'])
        inputs = self.processor(**kwargs).to('cuda:0')
        if self.metadata['model_type'] in ('internvl', 'gemma3'):
            for key, value in inputs.items():
                if torch.is_floating_point(value):
                    inputs[key] = value.to(dtype=torch.bfloat16)
        if images and self.metadata['model_type'] == 'internvl':
            shape = tuple(inputs['pixel_values'].shape)
            if shape != (len(images), 3, 448, 448):
                raise RuntimeError('InternVL requires exactly one 448px tile per image; got ' + str(shape))
        if images and self.metadata['model_type'] == 'gemma3':
            shape = tuple(inputs['pixel_values'].shape)
            if shape != (len(images), 3, GEMMA3_IMAGE, GEMMA3_IMAGE):
                raise RuntimeError('Gemma-3 requires one 896px view per image (no pan-and-scan); got ' + str(shape))
        return prompt, inputs

    def candidate_ids(self, labels, leading_space=False):
        tokens = {label:self.processor.tokenizer.encode((' ' if leading_space else '') + label,
                                                        add_special_tokens=False)
                  for label in labels}
        if any(len(ids) != 1 for ids in tokens.values()):
            raise ValueError('This protocol needs single-token candidates; do not approximate multi-token scores.')
        return {label:ids[0] for label,ids in tokens.items()}

    def render_prompt(self, content, short_answer=True, template_kwargs=None):
        messages = []
        if self.metadata['model_type'] == 'internvl':
            messages.append({'role': 'system', 'content': INTERNVL_SYSTEM})
            if short_answer:
                content = list(content) + [{'type': 'text', 'text': '\n' + INTERNVL_SHORT_SUFFIX}]
        messages.append({'role': 'user', 'content': content})
        return self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            **(self.template_kwargs if template_kwargs is None else template_kwargs))

    @torch.inference_mode()
    def score(self, rendered, labels, max_new_tokens=8):
        ids = self.candidate_ids(labels)
        prompt, inputs = self.prepare(rendered['content'], rendered['images'])
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.monotonic()
        # transformers>=4.5x merges model-specific defaults (e.g. Qwen3-VL do_sample=True) into any
        # value equal to the global default; explicit kwargs are applied last, so force greedy here.
        result = self.model.generate(**inputs, generation_config=self.generation_config,
                                     max_new_tokens=max_new_tokens, do_sample=False,
                                     repetition_penalty=1.0, top_k=None, top_p=None, temperature=None,
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
                   argmax_label=labels[int(candidate_logits.argmax())],
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

    def _answer_logits(self, inputs, prefix_ids):
        """Teacher-force prompt + prefix_ids (generated tokens up to the answer) in one fresh prefill and
        return the raw fp32 next-token logits. No cache, no processors; same fp32 head as score()."""
        ids = inputs['input_ids']
        prefix = prefix_ids.view(1, -1).to(ids.device)
        kw = dict(inputs)
        kw['input_ids'] = torch.cat([ids, prefix], 1)
        if 'attention_mask' in kw:
            kw['attention_mask'] = torch.cat([kw['attention_mask'], torch.ones_like(prefix)], 1)
        for key in ('token_type_ids', 'mm_token_type_ids'):
            if key in kw:
                kw[key] = torch.cat([kw[key], torch.zeros_like(prefix)], 1)
        try:
            out = self.model(**kw, use_cache=False, logits_to_keep=1)
        except TypeError:
            out = self.model(**kw, use_cache=False)
        return out.logits[0, -1].float()

    def _locate_answer(self, generation, char_offset):
        """Index k of the generated token that contains character `char_offset` of the decoded text, and
        whether that token carries a leading space (then the ' label' token ids are the candidates)."""
        tok = self.processor.tokenizer
        ids = generation.tolist()

        def covered(k):   # decoded length of the first k+1 tokens exceeds the offset
            return len(tok.decode(ids[:k + 1], skip_special_tokens=False)) > char_offset
        if not ids or not covered(len(ids) - 1):
            return None, False
        lo, hi = 0, len(ids) - 1          # binary search for the first covering token
        while lo < hi:
            mid = (lo + hi) // 2
            if covered(mid):
                hi = mid
            else:
                lo = mid + 1
        piece = tok.decode(ids[lo:lo + 1], skip_special_tokens=False)
        return lo, piece[:1].isspace()

    @torch.inference_mode()
    def score_with_reasoning(self, rendered, labels, max_new_tokens=4096, mode='thinking'):
        """Free reasoning, then a strictly parsed answer and ONE candidate readout, kept apart by kind.
        mode='thinking': native thinking template (enable_thinking=True); final answer = text after </think>.
        mode='cot'     : non-thinking template + COT_SUFFIX; answer needs 'Answer: <word>' (or a box / bare word).
        Decoding is greedy (do_sample=False, no repetition penalty), identical to score().

        Free answer: `answer` (None unless parse_label accepted it; `answer_valid`, `answer_status`,
        `answer_parse_reason`). Truncation (budget exhausted without EOS) always makes the answer invalid.
        Readout (`readout`):
          'natural' -- only when the free answer is valid: raw fp32 logits at the position where the model
                       generated that answer word (teacher-forced prompt + generated prefix).
          'forced'  -- otherwise (`answer_forced`=True): if the model closed its reasoning, the logits at the
                       slot where the answer starts (after </think> / the last 'Answer:'); if not, the closer
                       ('\n</think>\n\n' / '\nAnswer:') is appended to the generation and read there.
        `probabilities` / `logits` describe that one readout; consumers must split by `readout` and never
        treat a forced readout as a natural answer."""
        if mode not in ('thinking', 'cot'):
            raise ValueError(mode)
        content = list(rendered['content'])
        if mode == 'cot':
            content.append({'type': 'text', 'text': '\n' + COT_SUFFIX})
            template_kwargs = dict(self.template_kwargs)
        else:
            template_kwargs = dict(self.template_kwargs, enable_thinking=True)
        prompt, inputs = self.prepare(content, rendered['images'], short_answer=False,
                                      template_kwargs=template_kwargs)
        tok = self.processor.tokenizer
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.monotonic()
        sequences = self.model.generate(**inputs, generation_config=self.generation_config,
                                        max_new_tokens=max_new_tokens, do_sample=False,
                                        repetition_penalty=1.0, top_k=None, top_p=None, temperature=None)
        torch.cuda.synchronize()
        gen_seconds = time.monotonic() - started
        generation = sequences[0, inputs['input_ids'].shape[1]:]
        eos = self.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos])
        hit_eos = bool(generation.numel()) and int(generation[-1]) in eos
        body = generation[:-1] if hit_eos else generation
        raw_text = tok.decode(body, skip_special_tokens=False)
        truncated = (not hit_eos and generation.numel() >= max_new_tokens)
        parsed = parse_generation(raw_text, labels, mode, truncated)
        closed = parsed['closed']
        # tokens generated before the reasoning was closed (the whole generation if it never was)
        reasoning_tokens = (self._locate_answer(body, parsed['close_at'])[0] if closed else None)
        if reasoning_tokens is None:
            reasoning_tokens = int(body.numel())
        generated_answer_token, k, spaced = None, None, False
        if parsed['answer'] is not None:
            k, spaced = self._locate_answer(body, parsed['answer_offset'])
        if k is not None:
            readout, prefix = 'natural', body[:k]
            generated_answer_token = int(body[k])
        elif closed:
            # Model closed its reasoning but gave no valid answer: read the slot where the answer starts.
            readout = 'forced'
            if mode == 'thinking':
                final = parsed['final']
                k0, spaced = self._locate_answer(body, parsed['final_offset'] + len(final) - len(final.lstrip()))
            else:
                k0, spaced = self._locate_answer(body, list(ANSWER_MARKER.finditer(raw_text))[-1].end())
            prefix = body[:k0] if k0 is not None else body
            spaced = spaced if k0 is not None else (mode == 'cot')
        else:
            # Reasoning never closed (truncated, or EOS without closing): append the closer and read there.
            readout = 'forced'
            closer = '\n</think>\n\n' if mode == 'thinking' else '\nAnswer:'
            prefix = torch.cat([body, torch.tensor(tok.encode(closer, add_special_tokens=False),
                                                   device=body.device, dtype=body.dtype)])
            spaced = mode == 'cot'
        ids = self.candidate_ids(labels, leading_space=spaced)
        logits = self._answer_logits(inputs, prefix)
        full_prob = logits.softmax(-1)
        candidate_logits = torch.stack([logits[ids[label]] for label in labels])
        probabilities = candidate_logits.softmax(-1).tolist()
        torch.cuda.synchronize()
        answer = parsed['answer'] if readout == 'natural' else None
        status, reason = parsed['answer_status'], parsed['answer_parse_reason']
        if parsed['answer'] is not None and answer is None:      # parsed, but its token could not be located
            status, reason = PARSE_FAILED, 'answer_token_not_located'
        row = dict(probabilities=dict(zip(labels, probabilities)),
                   argmax_label=labels[int(candidate_logits.argmax())],
                   logits={label: float(logits[ids[label]]) for label in labels},
                   canonical_candidate_mass=float(sum(full_prob[ids[label]] for label in labels)),
                   readout=readout, answer_forced=readout == 'forced',
                   answer=answer, answer_valid=answer is not None,
                   answer_status=status, answer_parse_reason=reason,
                   final_text=parsed['final'].strip(), reasoning=parsed['reasoning'],
                   reasoning_tokens=reasoning_tokens,
                   answer_tokens=int(body.numel()) - reasoning_tokens,
                   generated_tokens=int(generation.numel()), truncated=bool(truncated),
                   reasoning_closed=bool(closed), hit_eos=hit_eos,
                   answer_token_index=int(prefix.numel()),
                   answer_leading_space=bool(spaced),
                   teacher_forced_argmax_is_generated=(None if generated_answer_token is None else
                                                       int(logits.argmax()) == generated_answer_token),
                   input_tokens=int(inputs['input_ids'].shape[1]), reasoning_mode=mode,
                   rendered_prompt=prompt, prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                   generation_seconds=gen_seconds, elapsed_seconds=time.monotonic() - started,
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                   peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                   candidate_token_ids=ids)
        del sequences, inputs
        return row

    @torch.inference_mode()
    def generate_text(self, prompt_text, max_new_tokens=96, short_answer=False):
        prompt, inputs = self.prepare([{'type':'text','text':prompt_text}], [],
                                     short_answer=short_answer)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.monotonic()
        result = self.model.generate(**inputs, generation_config=self.generation_config,
                                     max_new_tokens=max_new_tokens, do_sample=False,
                                     repetition_penalty=1.0, top_k=None, top_p=None, temperature=None)
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
