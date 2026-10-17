import math

import torch
import torch.nn.functional as F
from torch import nn
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.qwen3.modeling_qwen3 import Qwen3Model, Qwen3PreTrainedModel, Qwen3RMSNorm


DEFAULT_CONFIG = {
    "loop_num_layers": 3,
    "loop_train_max": 8,
    "loop_variable_depth": True,
    "loop_seed": 42,
    "loop_train_mean": 4.0,
    "loop_train_sigma": 0.5,
    "loop_infer_count": 2,
    "loop_memory_window": 3,
    "loop_attn_num_heads": 4,
    "loop_attn_head_dim": None,
    "loop_distance_bias": True,
    "loop_qk_norm": True,
    "loop_gate_init": 1.0,
    "loop_token_gate_init_bias": -3.0,
    "loop_input_injection_init": 0.0,
}

LOOP_PARAMETER_PREFIXES = (
    "model.loop_attn.",
    "model.loop_input_injection.",
)


def configure_model(config, **overrides):
    if config.model_type != "qwen3":
        raise ValueError("RecurTrace requires a Qwen3 configuration")
    unknown = set(overrides) - set(DEFAULT_CONFIG) - {"loop_start_layer"}
    if unknown:
        raise ValueError(f"unknown loop settings: {sorted(unknown)}")
    values = {key: getattr(config, key, default) for key, default in DEFAULT_CONFIG.items()}
    values["loop_start_layer"] = getattr(
        config, "loop_start_layer", {28: 12, 36: 15}.get(config.num_hidden_layers)
    )
    values.update({key: value for key, value in overrides.items() if value is not None})
    if values["loop_start_layer"] is None:
        raise ValueError("set loop_start_layer for this model's layer count")
    for key in (
        "loop_num_layers",
        "loop_train_max",
        "loop_infer_count",
        "loop_memory_window",
        "loop_attn_num_heads",
    ):
        if not isinstance(values[key], int) or values[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    start = values["loop_start_layer"]
    if (
        not isinstance(start, int)
        or start < 0
        or start + values["loop_num_layers"] > config.num_hidden_layers
    ):
        raise ValueError("the loop block must fit inside the base model")
    head_dim = values["loop_attn_head_dim"]
    if head_dim is not None and (not isinstance(head_dim, int) or head_dim < 1):
        raise ValueError("loop_attn_head_dim must be a positive integer")
    if head_dim is None and config.hidden_size % values["loop_attn_num_heads"]:
        raise ValueError("hidden_size must be divisible by loop_attn_num_heads")
    if not math.isfinite(values["loop_train_mean"]) or values["loop_train_mean"] < 1:
        raise ValueError("loop_train_mean must be finite and at least one")
    if not math.isfinite(values["loop_train_sigma"]) or values["loop_train_sigma"] < 0:
        raise ValueError("loop_train_sigma must be finite and nonnegative")
    for key, value in values.items():
        setattr(config, key, value)
    return config


def _alibi_slopes(num_heads):
    def power_of_two_slopes(count):
        start = 2 ** (-(2 ** -(math.log2(count) - 3)))
        return [start * start**index for index in range(count)]

    if math.log2(num_heads).is_integer():
        values = power_of_two_slopes(num_heads)
    else:
        closest = 2 ** math.floor(math.log2(num_heads))
        values = power_of_two_slopes(closest)
        values += power_of_two_slopes(2 * closest)[0::2][: num_heads - closest]
    return torch.tensor(values, dtype=torch.float32)


class LoopMemoryAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.loop_attn_num_heads
        self.head_dim = config.loop_attn_head_dim or self.hidden_size // self.num_heads
        self.inner_dim = self.num_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.use_distance_bias = config.loop_distance_bias
        self.use_qk_norm = config.loop_qk_norm
        self.norm = Qwen3RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.q_proj = nn.Linear(self.hidden_size, self.inner_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.inner_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.inner_dim, bias=False)
        self.o_proj = nn.Linear(self.inner_dim, self.hidden_size, bias=False)
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.gate_norm = Qwen3RMSNorm(2 * self.hidden_size, eps=config.rms_norm_eps)
        self.gate_proj = nn.Linear(2 * self.hidden_size, 1)
        self.gate = nn.Parameter(torch.tensor(float(config.loop_gate_init)))
        if self.use_distance_bias:
            self.loop_bias_slope = nn.Parameter(_alibi_slopes(self.num_heads))
        self.scalar_gate_init = float(config.loop_gate_init)
        self.token_gate_bias = float(config.loop_token_gate_init_bias)

    def reset_gate(self):
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, self.token_gate_bias)
        self.gate.data.fill_(self.scalar_gate_init)
        if self.use_distance_bias:
            self.loop_bias_slope.data.copy_(_alibi_slopes(self.num_heads))

    def forward(self, query_hidden, memory):
        batch_size, seq_len, _ = query_hidden.shape
        if memory is None or memory.shape[2] == 0:
            return torch.zeros_like(query_hidden)
        memory_len = memory.shape[2]
        query_input = self.norm(query_hidden)
        memory_input = self.norm(memory)
        query = self.q_proj(query_input).view(batch_size, seq_len, self.num_heads, self.head_dim)
        key = self.k_proj(memory_input).view(
            batch_size, seq_len, memory_len, self.num_heads, self.head_dim
        )
        value = self.v_proj(memory_input).view(
            batch_size, seq_len, memory_len, self.num_heads, self.head_dim
        )
        if self.use_qk_norm:
            query = self.q_norm(query)
            key = self.k_norm(key)
        logits = torch.einsum("bthd,btmhd->bthm", query, key) * self.scaling
        if self.use_distance_bias:
            distance = torch.arange(memory_len, 0, -1, device=logits.device, dtype=logits.dtype)
            slopes = self.loop_bias_slope.to(device=logits.device, dtype=logits.dtype)
            logits = logits - slopes[None, None, :, None] * distance[None, None, None, :]
        attention = F.softmax(logits, dim=-1, dtype=torch.float32).to(query.dtype)
        output = torch.einsum("bthm,btmhd->bthd", attention, value)
        output = self.o_proj(output.reshape(batch_size, seq_len, self.inner_dim))
        memory_mean = memory_input.mean(dim=2)
        gate_input = self.gate_norm(torch.cat([query_input, memory_mean], dim=-1))
        token_gate = torch.sigmoid(self.gate_proj(gate_input).float())
        return (self.gate * token_gate.to(output.dtype) * output).to(query_hidden.dtype)


class LoopInputInjection(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.alpha = nn.Parameter(torch.tensor(float(config.loop_input_injection_init)))

    def forward(self, hidden_states, block_input):
        injection = self.alpha * self.norm(block_input)
        return hidden_states + injection.to(hidden_states.dtype)


class RecurTraceModel(Qwen3Model):
    def __init__(self, config):
        super().__init__(configure_model(config))
        start = int(config.loop_start_layer)
        end = start + int(config.loop_num_layers)
        if start < 0 or end > config.num_hidden_layers:
            raise ValueError(
                f"loop block [{start}, {end}) is outside {config.num_hidden_layers} layers"
            )
        self.loop_layer_ids = list(range(start, end))
        self.loop_attn = nn.ModuleDict(
            {str(index): LoopMemoryAttention(config) for index in self.loop_layer_ids}
        )
        self.loop_input_injection = LoopInputInjection(config)
        self.register_buffer(
            "_loop_forward_counter", torch.zeros((), dtype=torch.long), persistent=False
        )
        self.collected_depth_hidden = {}
        self.collected_decision_features = {}
        self.last_actual_depth = 1
        self.last_continue_probabilities = []

    def reset_recurtrace_parameters(self):
        for module in self.loop_attn.values():
            module.reset_gate()
        self.loop_input_injection.alpha.data.fill_(float(self.config.loop_input_injection_init))

    def _resolve_loop_count(self):
        if not self.training or not self.config.loop_variable_depth:
            return max(1, int(self.config.loop_infer_count))
        step = int(self._loop_forward_counter.item())
        self._loop_forward_counter += 1
        generator = torch.Generator(device="cpu").manual_seed(int(self.config.loop_seed) + step)
        maximum = max(1, int(self.config.loop_train_max))
        mean = float(self.config.loop_train_mean) - 1.0
        if mean <= 0:
            return 1
        sigma = float(self.config.loop_train_sigma)
        mu = math.log(mean) - 0.5 * sigma**2
        rate = torch.exp(torch.randn((), generator=generator) * sigma + mu)
        count = int(torch.poisson(rate, generator=generator).item()) + 1
        return min(maximum, max(1, count))

    @staticmethod
    def _pool(hidden_states, prompt_mask, attention_mask):
        mask = prompt_mask if prompt_mask is not None else attention_mask
        if mask is None:
            return hidden_states.mean(dim=1)
        mask = mask.to(device=hidden_states.device, dtype=hidden_states.dtype)
        return (hidden_states * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(
            dim=1, keepdim=True
        ).clamp_min(1)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        cache_position=None,
        loop_count=None,
        collect_depths=None,
        prompt_mask=None,
        adaptive=False,
        halting_head=None,
        halting_threshold=0.5,
        halting_min_depth=2,
        **kwargs,
    ):
        kwargs.pop("return_dict", None)
        kwargs.pop("output_attentions", None)
        kwargs.pop("output_hidden_states", None)
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("provide exactly one of input_ids and inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        requested_depths = sorted(set(int(value) for value in (collect_depths or [])))
        if requested_depths and requested_depths[0] < 1:
            raise ValueError("collected depths must be positive")
        if loop_count is None:
            loop_count = max(requested_depths) if requested_depths else self._resolve_loop_count()
        if not isinstance(loop_count, int) or loop_count < 1:
            raise ValueError("loop_count must be a positive integer")
        if requested_depths and requested_depths[-1] > loop_count:
            raise ValueError("collected depths exceed loop_count")
        if adaptive and (halting_min_depth < 1 or halting_min_depth > loop_count):
            raise ValueError("halting_min_depth must lie within the loop budget")
        if adaptive and halting_head is None:
            raise ValueError("adaptive inference requires a halting head")
        use_cache = self.config.use_cache if use_cache is None else use_cache
        if loop_count > 1 or requested_depths or adaptive:
            use_cache = False
            past_key_values = None
            cache_position = None
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        if cache_position is None:
            seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                seen, seen + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        if not isinstance(attention_mask, dict):
            mask_args = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            masks = {"full_attention": create_causal_mask(**mask_args)}
            if self.has_sliding_layers:
                masks["sliding_attention"] = create_sliding_window_causal_mask(**mask_args)
        else:
            masks = attention_mask
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        def run_layer(index, states):
            return self.layers[index](
                states,
                attention_mask=masks[self.config.layer_types[index]],
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        start = self.loop_layer_ids[0]
        final_loop_layer = self.loop_layer_ids[-1]
        for index in range(start):
            hidden_states = run_layer(index, hidden_states)
        block_input = hidden_states
        memory = {index: [] for index in self.loop_layer_ids}
        block_states = {}
        decision_features = {}
        continue_probabilities = []
        actual_depth = 0
        for loop_index in range(loop_count):
            if loop_index:
                hidden_states = self.loop_input_injection(hidden_states, block_input)
            for index in self.loop_layer_ids:
                if memory[index]:
                    memory_tensor = torch.stack(memory[index], dim=2)
                    hidden_states = hidden_states + self.loop_attn[str(index)](
                        memory[index][-1], memory_tensor
                    )
                hidden_states = run_layer(index, hidden_states)
                memory[index].append(hidden_states)
                window = int(self.config.loop_memory_window)
                if window > 0 and len(memory[index]) > window:
                    memory[index] = memory[index][-window:]
            depth = loop_index + 1
            actual_depth = depth
            if depth in requested_depths:
                block_states[depth] = hidden_states
                decision_features[depth] = self._pool(hidden_states, prompt_mask, attention_mask)
            if adaptive and depth < loop_count:
                pooled = self._pool(hidden_states, prompt_mask, attention_mask)
                pooled = pooled.to(halting_head.weight.dtype)
                probability = torch.sigmoid(halting_head(pooled).squeeze(-1).float())
                continue_probabilities.append(float(probability.item()))
                if depth >= int(halting_min_depth) and bool(
                    (probability < float(halting_threshold)).all().item()
                ):
                    break

        final_block_states = hidden_states
        for index in range(final_loop_layer + 1, self.config.num_hidden_layers):
            final_block_states = run_layer(index, final_block_states)
        final_hidden = self.norm(final_block_states)
        depth_hidden = {}
        for depth, states in block_states.items():
            if depth == actual_depth:
                depth_hidden[depth] = final_hidden
                continue
            for index in range(final_loop_layer + 1, self.config.num_hidden_layers):
                states = run_layer(index, states)
            depth_hidden[depth] = self.norm(states)
        self.collected_depth_hidden = depth_hidden
        self.collected_decision_features = decision_features
        self.last_actual_depth = actual_depth
        self.last_continue_probabilities = continue_probabilities
        return BaseModelOutputWithPast(
            last_hidden_state=final_hidden,
            past_key_values=past_key_values if use_cache else None,
        )


class RecurTraceForCausalLM(Qwen3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _keys_to_ignore_on_load_missing = [
        r"model\.loop_attn\.",
        r"model\.loop_input_injection\.",
        r"halting_head\.",
    ]
    _tp_plan = {"lm_head": "colwise_gather_output"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(configure_model(config))
        self.model = RecurTraceModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.halting_head = nn.Linear(config.hidden_size, 1)
        self.post_init()
        self.model.reset_recurtrace_parameters()
        nn.init.zeros_(self.halting_head.bias)
        self.halting_trained = False

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, LoopMemoryAttention):
            if not getattr(module.gate_proj.weight, "_is_hf_initialized", False):
                nn.init.zeros_(module.gate_proj.weight)
            if not getattr(module.gate_proj.bias, "_is_hf_initialized", False):
                nn.init.constant_(module.gate_proj.bias, module.token_gate_bias)
            if not getattr(module.gate, "_is_hf_initialized", False):
                module.gate.data.fill_(module.scalar_gate_init)
            if module.use_distance_bias and not getattr(
                module.loop_bias_slope, "_is_hf_initialized", False
            ):
                module.loop_bias_slope.data.copy_(_alibi_slopes(module.num_heads))
        if isinstance(module, LoopInputInjection):
            if not getattr(module.alpha, "_is_hf_initialized", False):
                module.alpha.data.fill_(float(self.config.loop_input_injection_init))

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        **kwargs,
    ):
        loop_count = kwargs.get("loop_count", self.config.loop_infer_count)
        if loop_count > 1:
            if inputs_embeds is not None:
                raise ValueError("multi-loop generation requires input_ids")
            past_key_values = None
            cache_position = None
            kwargs["use_cache"] = False
        return super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values: Cache | None = None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        cache_position=None,
        logits_to_keep=0,
        loop_count=None,
        prompt_mask=None,
        collect_depths=None,
        **kwargs,
    ):
        num_items_in_batch = kwargs.pop("num_items_in_batch", None)
        if labels is not None and not (isinstance(logits_to_keep, int) and logits_to_keep == 0):
            raise ValueError("training requires logits_to_keep=0")
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            loop_count=loop_count,
            prompt_mask=prompt_mask,
            collect_depths=collect_depths,
            **kwargs,
        )
        indices = (
            slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        )
        logits = self.lm_head(outputs.last_hidden_state[:, indices, :])
        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                num_items_in_batch=num_items_in_batch,
            )
            loss = loss + self.model.loop_input_injection.alpha.to(loss.dtype) * 0
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
        )

    @torch.no_grad()
    def choose_depth(
        self, input_ids, attention_mask=None, max_depth=16, min_depth=2, threshold=0.5
    ):
        if not self.halting_trained:
            raise ValueError("load an adapter produced by train_halting.py before choosing depth")
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("choose_depth accepts one prompt at a time")
        if not 1 <= min_depth <= max_depth:
            raise ValueError("depths must satisfy 1 <= min_depth <= max_depth")
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must lie between zero and one")
        was_training = self.training
        self.eval()
        try:
            self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                loop_count=int(max_depth),
                use_cache=False,
                adaptive=True,
                halting_head=self.halting_head,
                halting_threshold=threshold,
                halting_min_depth=int(min_depth),
                prompt_mask=attention_mask,
            )
            return self.model.last_actual_depth, self.model.last_continue_probabilities
        finally:
            self.train(was_training)
