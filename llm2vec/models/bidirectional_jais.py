import torch
from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, CausalLMOutputWithCrossAttentions
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from typing import Optional, Tuple, Union

from .configuration_jais import JAISConfig
from .modeling_jais import (
    JAISPreTrainedModel,
    JAISModel,
    JAISLMHeadModel,
    JAISAttention,
    JAISMLP,
    JAISBlock,
    AlibiPositionEmbeddingLayer,
)

from peft import PeftModel

logger = logging.get_logger(__name__)


class ModifiedJAISAttention(JAISAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False  # Disable causal masking

    def _attn(self, query, key, value, attention_mask=None, head_mask=None, position_bias=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / torch.full(
                [], value.size(-1) ** self.attn_scale_power, dtype=attn_weights.dtype, device=attn_weights.device
            )

        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)


        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        if position_bias is not None:
            attn_weights += position_bias.type_as(attn_weights).unsqueeze(0)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def _upcast_and_reordered_attn(
        self, query, key, value, attention_mask=None, head_mask=None, position_bias=None
    ):
        bsz, num_heads, q_seq_len, dk = query.size()
        _, _, k_seq_len, _ = key.size()

        attn_weights = torch.empty(
            bsz * num_heads, q_seq_len, k_seq_len, dtype=torch.float32, device=query.device
        )

        scale_factor = 1.0
        if self.scale_attn_weights:
            scale_factor /= float(value.size(-1)) ** self.attn_scale_power

        if self.scale_attn_by_inverse_layer_idx:
            scale_factor /= float(self.layer_idx + 1)

        with torch.cuda.amp.autocast(enabled=False):
            q = query.reshape(-1, q_seq_len, dk)
            k = key.transpose(-1, -2).reshape(-1, dk, k_seq_len)
            attn_weights = torch.baddbmm(
                attn_weights, q.float(), k.float(), beta=0, alpha=scale_factor
            )
            attn_weights = attn_weights.reshape(bsz, num_heads, q_seq_len, k_seq_len)


        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        if position_bias is not None:
            attn_weights += position_bias.type_as(attn_weights).unsqueeze(0)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if attn_weights.dtype != torch.float32:
            raise RuntimeError(
                "Error with upcasting, attn_weights does not have dtype torch.float32"
            )
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights


class ModifiedJAISBlock(JAISBlock):
    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx=layer_idx)
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = ModifiedJAISAttention(config, layer_idx=layer_idx)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        if config.add_cross_attention:
            self.crossattention = ModifiedJAISAttention(
                config, is_cross_attention=True, layer_idx=layer_idx
            )
            self.ln_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = JAISMLP(inner_dim, config)


class JaisBiModel(JAISPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [
        r"h\.\d+\.attn\.bias",
        r"h\.\d+\.attn\.masked_bias",
    ]
    _keys_to_ignore_on_load_missing = [
        r"attn.masked_bias",
        r"h\.\d+\.attn\.masked_bias",
        r"h\.\d+\.attn\.bias",
    ]

    def __init__(self, config):
        super().__init__(config)

        self.embed_dim = config.hidden_size

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = (
            nn.Embedding(config.max_position_embeddings, self.embed_dim)
            if config.position_embedding_type != "alibi"
            else None
        )
        self.embeddings_scale = config.mup_embeddings_scale

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList(
            [ModifiedJAISBlock(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        self.relative_pe = (
            AlibiPositionEmbeddingLayer(config.num_attention_heads, config.alibi_scaling)
            if config.position_embedding_type == "alibi"
            else None
        )

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

def forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    token_type_ids: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds")
        
    if input_ids is not None:
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        batch_size = input_ids.shape[0]
    elif inputs_embeds is not None:
        input_shape = inputs_embeds.size()[:-1]
        batch_size = inputs_embeds.shape[0]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    device = input_ids.device if input_ids is not None else inputs_embeds.device

    if token_type_ids is not None:
        token_type_ids = token_type_ids.view(-1, input_shape[-1])
    if position_ids is not None:
        position_ids = position_ids.view(-1, input_shape[-1])

    if past_key_values is None:
        past_length = 0
        past_key_values = tuple([None] * len(self.h))
    else:
        past_length = past_key_values[0][0].size(-2)

    if inputs_embeds is None:
        inputs_embeds = self.wte(input_ids)

    if self.wpe is not None and position_ids is not None:
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
    else:
        hidden_states = inputs_embeds

    if self.embeddings_scale:
        hidden_states = hidden_states * (self.embed_dim ** 0.5)

    hidden_states = self.drop(hidden_states)

    if self.relative_pe is not None:
        length = input_shape[-1]
        cached_kv_length = 0
        cached_kv = past_key_values[0]
        if cached_kv is not None:
            cached_kv_length = cached_kv[0].shape[-2]
        position_bias = self.relative_pe(length, length, cached_kv_length)
    else:
        position_bias = None

    output_shape = input_shape + (hidden_states.size(-1),)

    presents = () if use_cache else None
    all_self_attentions = () if output_attentions else None
    all_hidden_states = () if output_hidden_states else None
    
    for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = block(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=None if head_mask is None else head_mask[i],
            use_cache=use_cache,
            output_attentions=output_attentions,
            position_bias=position_bias,
        )

        hidden_states = outputs[0]
        if use_cache is True:
            presents = presents + (outputs[1],)

        if output_attentions:
            all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

    hidden_states = self.ln_f(hidden_states)
    hidden_states = hidden_states.view(output_shape)

    if not return_dict:
        return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

    return BaseModelOutputWithPastAndCrossAttentions(
        last_hidden_state=hidden_states,
        past_key_values=presents,
        hidden_states=all_hidden_states,
        attentions=all_self_attentions,
    )


class JaisBiForMNTP(JAISLMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = JaisBiModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.output_logits_scale = config.mup_output_alpha * config.mup_width_scale

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()

    # Getter for PEFT model
    def get_model_for_peft(self):
        return self.transformer

    # get the base model
    def get_base_model(self):
        return self.transformer
    
    # Setter for PEFT model
    def set_model_for_peft(self, model: PeftModel):
        self.transformer = model

    # Save the PEFT model
    def save_peft_model(self, path):
        self.transformer.save_pretrained(path)

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)
