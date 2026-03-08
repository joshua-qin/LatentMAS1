import os
import csv
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from vllm import LLM, SamplingParams
    _HAS_VLLM = True
except ImportError:
    _HAS_VLLM = False

try:
    from transformers.cache_utils import Cache
except ImportError:
    Cache = None


def _ensure_pad_token(tokenizer: AutoTokenizer) -> None:
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})


def _past_length(past_key_values) -> int:
    if past_key_values is None:
        return 0

    if Cache is not None and isinstance(past_key_values, Cache):
        return past_key_values.get_seq_length()

    # legacy tuple format
    k = past_key_values[0][0]
    return k.shape[-2]


# --- RoPE helpers for hierarchical_fixed cache fusion ---
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims (for RoPE)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _get_rope_config(hf_model) -> Optional[dict]:
    """Get rope_theta, head_dim, num_heads from model config. Returns None if not found."""
    config = getattr(hf_model, "config", None)
    if config is None:
        return None
    rope_theta = getattr(config, "rope_theta", None)
    if rope_theta is None:
        rp = getattr(config, "rope_parameters", None)
        if isinstance(rp, dict):
            rope_theta = rp.get("rope_theta", 10000.0)
        elif rp is not None:
            rope_theta = getattr(rp, "rope_theta", 10000.0)
        else:
            rope_theta = 10000.0
    num_heads = getattr(config, "num_attention_heads", None)
    head_dim = getattr(config, "head_dim", None) or (
        getattr(config, "hidden_size", 0) // num_heads if num_heads else None
    )
    if head_dim is None or num_heads is None:
        return None
    return {"rope_theta": float(rope_theta), "head_dim": int(head_dim), "num_heads": int(num_heads)}


def _rope_cos_sin(key: torch.Tensor, position_start: int, rope_config: dict, device: torch.device, dtype: torch.dtype):
    """Compute cos/sin for RoPE for positions [position_start, position_start+seq_len). Shape for key [..., seq_len, head_dim]."""
    seq_len = key.shape[-2]
    head_dim = rope_config["head_dim"]
    base = rope_config["rope_theta"]
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    positions = torch.arange(position_start, position_start + seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    cos = freqs.cos().to(dtype).unsqueeze(0).unsqueeze(0)
    sin = freqs.sin().to(dtype).unsqueeze(0).unsqueeze(0)
    # expand to [1, 1, seq_len, head_dim] (pairs duplicated for RoPE)
    cos = torch.cat([cos, cos], dim=-1)
    sin = torch.cat([sin, sin], dim=-1)
    return cos, sin


def _apply_rotary(key: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to key: key_rot = key * cos + rotate_half(key) * sin."""
    return (key * cos) + (_rotate_half(key) * sin)


def _unapply_rotary(key_rot: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Inverse RoPE: key = key_rot * cos - rotate_half(key_rot) * sin."""
    return (key_rot * cos) - (_rotate_half(key_rot) * sin)


class ModelWrapper:
    def __init__(self, model_name: str, device: torch.device, use_vllm: bool = False, args = None):
        self.model_name = model_name
        self.device = device
        self.use_vllm = use_vllm and _HAS_VLLM
        self.vllm_engine = None
        self.latent_space_realign = bool(getattr(args, "latent_space_realign", False)) if args else False
        self._latent_realign_matrices: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.args = args

        # for ablation
        self.pre_aligned = None

        if self.use_vllm:
            
            tp_size = max(1, int(getattr(args, "tensor_parallel_size", 1)))
            gpu_util = float(getattr(args, "gpu_memory_utilization", 0.9))
            
            print(f"[vLLM] Using vLLM backend for model {model_name}")
            if args.enable_prefix_caching and args.method == "latent_mas": 
                self.vllm_engine = LLM(model=model_name, tensor_parallel_size=tp_size, gpu_memory_utilization=gpu_util, enable_prefix_caching=True, enable_prompt_embeds=True)
            else:
                self.vllm_engine = LLM(model=model_name, tensor_parallel_size=tp_size, gpu_memory_utilization=gpu_util)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            
            use_second_hf = bool(getattr(args, "use_second_HF_model", False)) if args else False
            if use_second_hf:
                self.HF_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
                ).to(args.device2).eval() 
                self.embedding_layer = self.HF_model.get_input_embeddings()
                self.HF_device = args.device2
                # if self.latent_space_realign:
                self._ensure_latent_realign_matrix(self.HF_model, torch.device(self.HF_device), args)
            elif self.latent_space_realign:
                raise ValueError("latent_space_realign requires --use_second_HF_model when using vLLM backend.")
            _ensure_pad_token(self.tokenizer)
            return  # skip loading transformers model

        # fallback: normal transformers path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        _ensure_pad_token(self.tokenizer)
        with torch.no_grad():
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
            )
        if len(self.tokenizer) != self.model.get_input_embeddings().weight.shape[0]:
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(device)
        self.model.eval()
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = True
        if self.latent_space_realign:
            self._ensure_latent_realign_matrix(self.model, self.device, args)

    def render_chat(self, messages: List[Dict], add_generation_prompt: bool = True) -> str:
        tpl = getattr(self.tokenizer, "chat_template", None)
        if tpl:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt
            )
        segments = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            segments.append(f"<|{role}|>\n{content}\n</|{role}|>")
        if add_generation_prompt:
            segments.append("<|assistant|>")
        return "\n".join(segments)

    def prepare_chat_input(
        self, messages: List[Dict], add_generation_prompt: bool = True
    ) -> Tuple[str, torch.Tensor, torch.Tensor, List[str]]:
        prompt_text = self.render_chat(messages, add_generation_prompt=add_generation_prompt)
        encoded = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        active_ids = input_ids[0][attention_mask[0].bool()].tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(active_ids)
        return prompt_text, input_ids, attention_mask, tokens

    def prepare_chat_batch(
        self,
        batch_messages: List[List[Dict]],
        add_generation_prompt: bool = True,
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor, List[List[str]]]:
        prompts: List[str] = []
        for messages in batch_messages:
            prompts.append(self.render_chat(messages, add_generation_prompt=add_generation_prompt))
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        tokens_batch: List[List[str]] = []
        for ids_row, mask_row in zip(input_ids, attention_mask):
            active_ids = ids_row[mask_row.bool()].tolist()
            tokens_batch.append(self.tokenizer.convert_ids_to_tokens(active_ids))
        return prompts, input_ids, attention_mask, tokens_batch

    def vllm_generate_text_batch(
        self,
        prompts: List[str],
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> List[str]:
        if not self.vllm_engine:
            raise RuntimeError("vLLM engine not initialized. Pass use_vllm=True to ModelWrapper.")
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
        )
        outputs = self.vllm_engine.generate(prompts, sampling_params)
        generations = [out.outputs[0].text.strip() for out in outputs]
        return generations
    
    def _build_latent_realign_matrix(self, model, device, args) -> Tuple[torch.Tensor, torch.Tensor]:
        input_embeds = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
        output_embeds = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
        if output_embeds is None:
            output_embeds = getattr(model, "lm_head", None)
        if (
            input_embeds is None
            or output_embeds is None
            or not hasattr(input_embeds, "weight")
            or not hasattr(output_embeds, "weight")
        ):
            raise RuntimeError("Cannot build latent realignment matrix: embedding weights not accessible.")
        input_weight = input_embeds.weight.detach().to(device=device, dtype=torch.float32)
        output_weight = output_embeds.weight.detach().to(device=device, dtype=torch.float32)
        gram = torch.matmul(output_weight.T, output_weight)
        reg = 1e-5 * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        gram = gram + reg
        rhs = torch.matmul(output_weight.T, input_weight)
        realign_matrix = torch.linalg.solve(gram, rhs)
        target_norm = input_weight.norm(dim=1).mean().detach()

        if self.args.latent_space_realign:
            pass
        else:
            # keep the matrix, for further normalization
            realign_matrix = torch.eye(realign_matrix.shape[0], device=realign_matrix.device, dtype=realign_matrix.dtype)

        return realign_matrix, target_norm

    def _ensure_latent_realign_matrix(self, model, device, args) -> Tuple[torch.Tensor, torch.Tensor]:
        key = id(model)
        info = self._latent_realign_matrices.get(key)
        target_device = torch.device(device)

        if info is None:
            matrix, target_norm = self._build_latent_realign_matrix(model, target_device, args)
        else:
            matrix, target_norm = info
            if matrix.device != target_device:
                matrix = matrix.to(target_device)

        target_norm = target_norm.to(device=target_device, dtype=matrix.dtype) if isinstance(target_norm, torch.Tensor) else torch.as_tensor(target_norm, device=target_device, dtype=matrix.dtype)
        self._latent_realign_matrices[key] = (matrix, target_norm)

        return matrix, target_norm

    def _apply_latent_realignment(self, hidden: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
        matrix, target_norm = self._ensure_latent_realign_matrix(model, hidden.device, self.args)
        hidden_fp32 = hidden.to(torch.float32)
        aligned = torch.matmul(hidden_fp32, matrix)

        aligned_norm = aligned.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        pre_aligned = aligned.detach().clone()
        self.pre_aligned = pre_aligned
        aligned = aligned * (target_norm / aligned_norm)
        return aligned.to(hidden.dtype)

    @torch.no_grad()
    def generate_text_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        past_key_values: Optional[Tuple] = None,
    ) -> Tuple[List[str], Optional[Tuple]]:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D with shape [batch, seq_len]")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.device)
        prompt_lengths = attention_mask.sum(dim=1).tolist()
        cache_position = None
        if past_key_values is not None:
            past_len = _past_length(past_key_values)
            cache_position = torch.arange(
                past_len,
                past_len + input_ids.shape[-1],
                dtype=torch.long,
                device=self.device,
            )
            if past_len > 0:
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=False,
            past_key_values=past_key_values,
            cache_position=cache_position,
        )
        sequences = outputs.sequences
        generations: List[str] = []
        for idx, length in enumerate(prompt_lengths):
            length = int(length)
            generated_ids = sequences[idx, length:]
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            generations.append(text)
        return generations, outputs.past_key_values

    def unrotate_legacy_cache(self, cache_legacy: Optional[Tuple]) -> Optional[Tuple]:
        """Unrotate keys in legacy cache (inverse RoPE) so cache can be fused. Keys assumed rotated with positions 0..L-1."""
        if cache_legacy is None:
            return None
        hf_model = self.model if hasattr(self, "model") else getattr(self, "HF_model", None)
        if hf_model is None:
            print("[hierarchical_fixed_rope] No HF model available; skipping unrotate (cache returned as-is).")
            return cache_legacy
        rope_config = _get_rope_config(hf_model)
        if rope_config is None:
            print("[hierarchical_fixed_rope] Could not get RoPE config (rope_theta/head_dim); skipping unrotate (cache returned as-is).")
            return cache_legacy
        device = next(hf_model.parameters()).device
        dtype = next(hf_model.parameters()).dtype
        out = []
        try:
            for k, v in cache_legacy:
                seq_len = k.shape[-2]
                cos, sin = _rope_cos_sin(k, 0, rope_config, device, dtype)
                cos = cos.to(k.device)
                sin = sin.to(k.device)
                k_unrot = _unapply_rotary(k.float(), cos, sin).to(k.dtype)
                out.append((k_unrot, v))
            return tuple(out)
        except Exception as e:
            print(f"[hierarchical_fixed_rope] unrotate_legacy_cache failed: {e}; returning cache as-is.")
            return cache_legacy

    def rotate_legacy_cache(self, cache_legacy: Optional[Tuple], position_start: int = 0) -> Optional[Tuple]:
        """Apply RoPE to keys in legacy cache with positions [position_start, ...]. Use after fusing caches."""
        if cache_legacy is None:
            return None
        hf_model = self.model if hasattr(self, "model") else getattr(self, "HF_model", None)
        if hf_model is None:
            print("[hierarchical_fixed_rope] No HF model available; skipping rotate (cache returned as-is).")
            return cache_legacy
        rope_config = _get_rope_config(hf_model)
        if rope_config is None:
            print("[hierarchical_fixed_rope] Could not get RoPE config; skipping rotate (cache returned as-is).")
            return cache_legacy
        device = next(hf_model.parameters()).device
        dtype = next(hf_model.parameters()).dtype
        out = []
        try:
            for k, v in cache_legacy:
                cos, sin = _rope_cos_sin(k, position_start, rope_config, device, dtype)
                cos = cos.to(k.device)
                sin = sin.to(k.device)
                k_rot = _apply_rotary(k.float(), cos, sin).to(k.dtype)
                out.append((k_rot, v))
            return tuple(out)
        except Exception as e:
            print(f"[hierarchical_fixed_rope] rotate_legacy_cache failed: {e}; returning cache as-is.")
            return cache_legacy

    def tokenize_text(self, text: str) -> torch.Tensor:
        return self.tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].to(self.device)

    @torch.no_grad()
    def generate_latent_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        latent_steps: int,
        past_key_values: Optional[Tuple] = None,
    ) -> Tuple:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D with shape [batch, seq_len]")

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.device)
        else:
            attention_mask = attention_mask.to(self.device)

        if past_key_values is not None:
            past_len = _past_length(past_key_values)
            if past_len > 0:
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past = outputs.past_key_values

        e_t = outputs.hidden_states[0][:, -1, :]          # [B, D]
        last_hidden = outputs.hidden_states[-1][:, -1, :] # [B, D]
        h_t = last_hidden.detach().clone()

        e_t_plus_1 = None
        latent_vecs_all: List[torch.Tensor] = []
        latent_vecs_all.append(e_t.detach().clone())

        for step in range(latent_steps):

            source_model = self.HF_model if hasattr(self, "HF_model") else self.model
            latent_vec = self._apply_latent_realignment(last_hidden, source_model)

            latent_vecs_all.append(latent_vec.detach().clone())

            if step == 0:
                e_t_plus_1 = latent_vec.detach().clone()
            
            latent_embed = latent_vec.unsqueeze(1)

            past_len = _past_length(past)
            latent_mask = torch.ones(
                (latent_embed.shape[0], past_len + 1),
                dtype=torch.long,
                device=self.device,
            )
            outputs = self.model(
                inputs_embeds=latent_embed,
                attention_mask=latent_mask,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past = outputs.past_key_values
            last_hidden = outputs.hidden_states[-1][:, -1, :]

        return past
    
    @torch.no_grad()
    def generate_latent_batch_hidden_state(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        latent_steps: int,
        past_key_values: Optional[Tuple] = None,
    ) -> Tuple:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D with shape [batch, seq_len]")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.HF_device)
        else:
            attention_mask = attention_mask.to(self.HF_device)
        if past_key_values is not None:
            past_len = _past_length(past_key_values)
            if past_len > 0:
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)
        outputs = self.HF_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past = outputs.past_key_values
        last_hidden = outputs.hidden_states[-1][:, -1, :]
        
        curr_output_embedding = [] 
        curr_output_embedding.append(outputs.hidden_states[0])  # input embedding
        
        
        for _ in range(latent_steps):

            source_model = self.HF_model if hasattr(self, "HF_model") else self.model
            latent_vec = self._apply_latent_realignment(last_hidden, source_model)
            latent_embed = latent_vec.unsqueeze(1)
            past_len = _past_length(past)
            latent_mask = torch.ones(
                (latent_embed.shape[0], past_len + 1),
                dtype=torch.long,
                device=latent_embed.device,
            )
            outputs = self.HF_model(
                inputs_embeds=latent_embed,
                attention_mask=latent_mask,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past = outputs.past_key_values
            last_hidden = outputs.hidden_states[-1][:, -1, :]

            curr_output_embedding.append(latent_embed.detach())

        return past, torch.cat(curr_output_embedding, dim=1) # Output input embeddings
