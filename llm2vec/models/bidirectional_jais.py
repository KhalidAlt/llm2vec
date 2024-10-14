import torch
from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, CausalLMOutputWithCrossAttentions
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

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