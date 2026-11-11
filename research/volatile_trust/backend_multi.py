"""Multi-GPU variant of FrozenVLM for >24GB models (identical scoring path, sharded weights)."""
import sys
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForImageTextToText, GenerationConfig

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'research/source_trust'))
import model_backends as mb  # noqa: E402


class ShardedVLM(mb.FrozenVLM):
    def __init__(self, model_dir, model_id, pixels=224, max_gib_per_gpu=None):
        preprocessing = mb.local_preprocessing(model_dir, pixels)
        n = torch.cuda.device_count()
        if n < 1:
            raise RuntimeError('No CUDA devices visible')
        torch.manual_seed(20261111)
        self.processor = mb.load_processor(model_dir, preprocessing)   # Gemma-3: fast tokenizer, slow image proc
        # Default: 90% of each visible device (works for 24GB 3090s and 192GB MI300X alike).
        import os
        if max_gib_per_gpu is None and os.environ.get('VT_MAX_GIB'):
            max_gib_per_gpu = float(os.environ['VT_MAX_GIB'])
        def free_gib(i):   # currently free memory (respects other users' processes), minus headroom
            free, total = torch.cuda.mem_get_info(i)
            return max(1, int(min(free - 2.5 * 2**30, 0.9 * total) / 2**30))
        max_memory = {i: (f'{max_gib_per_gpu}GiB' if max_gib_per_gpu else f'{free_gib(i)}GiB')
                      for i in range(n)}
        self.model = AutoModelForImageTextToText.from_pretrained(
            str(model_dir), local_files_only=True, trust_remote_code=False,
            torch_dtype=torch.bfloat16, device_map='auto', max_memory=max_memory,
            attn_implementation='sdpa').eval()
        head = self.model.get_output_embeddings()
        tied = head.weight.data_ptr() == self.model.get_input_embeddings().weight.data_ptr()
        projection = torch.nn.Linear(head.in_features, head.out_features, bias=head.bias is not None,
                                     device=head.weight.device, dtype=torch.float32)
        with torch.no_grad():
            projection.weight.copy_(head.weight)
            if head.bias is not None:
                projection.bias.copy_(head.bias)
        # keep accelerate hooks: replace weights in-place on the existing module path
        self.model.set_output_embeddings(projection)
        self.model.config.tie_word_embeddings = False
        projection.register_forward_pre_hook(
            lambda module, inputs: (inputs[0].to(projection.weight.device).float(),))
        del head
        g = self.model.generation_config
        eos = g.eos_token_id
        self.generation_config = GenerationConfig(
            do_sample=False, use_cache=True, bos_token_id=g.bos_token_id, eos_token_id=eos,
            pad_token_id=g.pad_token_id if g.pad_token_id is not None else (eos[0] if isinstance(eos, list) else eos))
        self.template_kwargs = {'enable_thinking': False}
        self.metadata = dict(model_id=model_id, model_revision=Path(model_dir).name,
                             model_type=self.model.config.model_type, model_class=type(self.model).__name__,
                             backbone_dtype='bfloat16', output_projection_dtype='float32',
                             original_output_head_tied=tied, pixels=pixels, preprocessing=preprocessing,
                             native_prompt_policy=preprocessing['chat_policy'],
                             candidate_scoring='raw_generation_logits_before_processors',
                             chat_template_kwargs=self.template_kwargs, torch=torch.__version__,
                             transformers=transformers.__version__, devices=n,
                             device=torch.cuda.get_device_name(0), sharded=True,
                             hip=getattr(torch.version, 'hip', None), cuda=torch.version.cuda,
                             max_memory=max_memory)
