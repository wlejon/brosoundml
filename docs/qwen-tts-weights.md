# Qwen3-TTS-12Hz-0.6B-CustomVoice -- weight map

Generated reference for the brosoundml build-out (the weights themselves
are not checked in). Tensor name, shape, dtype. Source repo:
`Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` (model.safetensors) +
`speech_tokenizer/model.safetensors` (the bundled 12 Hz codec).


## Talker model (`model.safetensors`) -- 402 tensors

```
[2048, 1024]           BF16  talker.code_predictor.lm_head.0.weight
[2048, 1024]           BF16  talker.code_predictor.lm_head.1.weight
[2048, 1024]           BF16  talker.code_predictor.lm_head.10.weight
[2048, 1024]           BF16  talker.code_predictor.lm_head.11.weight
[2048, 1024]           BF16  talker.code_predictor.lm_head.12.weight
[2048, 1024]           BF16  talker.code_predictor.lm_head.13.weight
[2048, 1024]           BF16  talker.code_predictor.lm_head.14.weight
[2048, 1024]           BF16  talker.code_predictor.lm_head.2.weight
[2048, 1024]           BF16  talker.code_predictor.lm_head.3.weight
[2048, 1024]           BF16  talker.code_predictor.lm_head.4.weight
[2048, 1024]           BF16  talker.code_predictor.lm_head.5.weight
[2048, 1024]           BF16  talker.code_predictor.lm_head.6.weight
[2048, 1024]           BF16  talker.code_predictor.lm_head.7.weight
[2048, 1024]           BF16  talker.code_predictor.lm_head.8.weight
[2048, 1024]           BF16  talker.code_predictor.lm_head.9.weight
[2048, 1024]           BF16  talker.code_predictor.model.codec_embedding.0.weight
[2048, 1024]           BF16  talker.code_predictor.model.codec_embedding.1.weight
[2048, 1024]           BF16  talker.code_predictor.model.codec_embedding.10.weight
[2048, 1024]           BF16  talker.code_predictor.model.codec_embedding.11.weight
[2048, 1024]           BF16  talker.code_predictor.model.codec_embedding.12.weight
[2048, 1024]           BF16  talker.code_predictor.model.codec_embedding.13.weight
[2048, 1024]           BF16  talker.code_predictor.model.codec_embedding.14.weight
[2048, 1024]           BF16  talker.code_predictor.model.codec_embedding.2.weight
[2048, 1024]           BF16  talker.code_predictor.model.codec_embedding.3.weight
[2048, 1024]           BF16  talker.code_predictor.model.codec_embedding.4.weight
[2048, 1024]           BF16  talker.code_predictor.model.codec_embedding.5.weight
[2048, 1024]           BF16  talker.code_predictor.model.codec_embedding.6.weight
[2048, 1024]           BF16  talker.code_predictor.model.codec_embedding.7.weight
[2048, 1024]           BF16  talker.code_predictor.model.codec_embedding.8.weight
[2048, 1024]           BF16  talker.code_predictor.model.codec_embedding.9.weight
[1024]                 BF16  talker.code_predictor.model.layers.0.input_layernorm.weight
[1024, 3072]           BF16  talker.code_predictor.model.layers.0.mlp.down_proj.weight
[3072, 1024]           BF16  talker.code_predictor.model.layers.0.mlp.gate_proj.weight
[3072, 1024]           BF16  talker.code_predictor.model.layers.0.mlp.up_proj.weight
[1024]                 BF16  talker.code_predictor.model.layers.0.post_attention_layernorm.weight
[128]                  BF16  talker.code_predictor.model.layers.0.self_attn.k_norm.weight
[1024, 1024]           BF16  talker.code_predictor.model.layers.0.self_attn.k_proj.weight
[1024, 2048]           BF16  talker.code_predictor.model.layers.0.self_attn.o_proj.weight
[128]                  BF16  talker.code_predictor.model.layers.0.self_attn.q_norm.weight
[2048, 1024]           BF16  talker.code_predictor.model.layers.0.self_attn.q_proj.weight
[1024, 1024]           BF16  talker.code_predictor.model.layers.0.self_attn.v_proj.weight
[1024]                 BF16  talker.code_predictor.model.layers.1.input_layernorm.weight
[1024, 3072]           BF16  talker.code_predictor.model.layers.1.mlp.down_proj.weight
[3072, 1024]           BF16  talker.code_predictor.model.layers.1.mlp.gate_proj.weight
[3072, 1024]           BF16  talker.code_predictor.model.layers.1.mlp.up_proj.weight
[1024]                 BF16  talker.code_predictor.model.layers.1.post_attention_layernorm.weight
[128]                  BF16  talker.code_predictor.model.layers.1.self_attn.k_norm.weight
[1024, 1024]           BF16  talker.code_predictor.model.layers.1.self_attn.k_proj.weight
[1024, 2048]           BF16  talker.code_predictor.model.layers.1.self_attn.o_proj.weight
[128]                  BF16  talker.code_predictor.model.layers.1.self_attn.q_norm.weight
[2048, 1024]           BF16  talker.code_predictor.model.layers.1.self_attn.q_proj.weight
[1024, 1024]           BF16  talker.code_predictor.model.layers.1.self_attn.v_proj.weight
[1024]                 BF16  talker.code_predictor.model.layers.2.input_layernorm.weight
[1024, 3072]           BF16  talker.code_predictor.model.layers.2.mlp.down_proj.weight
[3072, 1024]           BF16  talker.code_predictor.model.layers.2.mlp.gate_proj.weight
[3072, 1024]           BF16  talker.code_predictor.model.layers.2.mlp.up_proj.weight
[1024]                 BF16  talker.code_predictor.model.layers.2.post_attention_layernorm.weight
[128]                  BF16  talker.code_predictor.model.layers.2.self_attn.k_norm.weight
[1024, 1024]           BF16  talker.code_predictor.model.layers.2.self_attn.k_proj.weight
[1024, 2048]           BF16  talker.code_predictor.model.layers.2.self_attn.o_proj.weight
[128]                  BF16  talker.code_predictor.model.layers.2.self_attn.q_norm.weight
[2048, 1024]           BF16  talker.code_predictor.model.layers.2.self_attn.q_proj.weight
[1024, 1024]           BF16  talker.code_predictor.model.layers.2.self_attn.v_proj.weight
[1024]                 BF16  talker.code_predictor.model.layers.3.input_layernorm.weight
[1024, 3072]           BF16  talker.code_predictor.model.layers.3.mlp.down_proj.weight
[3072, 1024]           BF16  talker.code_predictor.model.layers.3.mlp.gate_proj.weight
[3072, 1024]           BF16  talker.code_predictor.model.layers.3.mlp.up_proj.weight
[1024]                 BF16  talker.code_predictor.model.layers.3.post_attention_layernorm.weight
[128]                  BF16  talker.code_predictor.model.layers.3.self_attn.k_norm.weight
[1024, 1024]           BF16  talker.code_predictor.model.layers.3.self_attn.k_proj.weight
[1024, 2048]           BF16  talker.code_predictor.model.layers.3.self_attn.o_proj.weight
[128]                  BF16  talker.code_predictor.model.layers.3.self_attn.q_norm.weight
[2048, 1024]           BF16  talker.code_predictor.model.layers.3.self_attn.q_proj.weight
[1024, 1024]           BF16  talker.code_predictor.model.layers.3.self_attn.v_proj.weight
[1024]                 BF16  talker.code_predictor.model.layers.4.input_layernorm.weight
[1024, 3072]           BF16  talker.code_predictor.model.layers.4.mlp.down_proj.weight
[3072, 1024]           BF16  talker.code_predictor.model.layers.4.mlp.gate_proj.weight
[3072, 1024]           BF16  talker.code_predictor.model.layers.4.mlp.up_proj.weight
[1024]                 BF16  talker.code_predictor.model.layers.4.post_attention_layernorm.weight
[128]                  BF16  talker.code_predictor.model.layers.4.self_attn.k_norm.weight
[1024, 1024]           BF16  talker.code_predictor.model.layers.4.self_attn.k_proj.weight
[1024, 2048]           BF16  talker.code_predictor.model.layers.4.self_attn.o_proj.weight
[128]                  BF16  talker.code_predictor.model.layers.4.self_attn.q_norm.weight
[2048, 1024]           BF16  talker.code_predictor.model.layers.4.self_attn.q_proj.weight
[1024, 1024]           BF16  talker.code_predictor.model.layers.4.self_attn.v_proj.weight
[1024]                 BF16  talker.code_predictor.model.norm.weight
[3072, 1024]           BF16  talker.codec_head.weight
[3072, 1024]           BF16  talker.model.codec_embedding.weight
[1024]                 BF16  talker.model.layers.0.input_layernorm.weight
[1024, 3072]           BF16  talker.model.layers.0.mlp.down_proj.weight
[3072, 1024]           BF16  talker.model.layers.0.mlp.gate_proj.weight
[3072, 1024]           BF16  talker.model.layers.0.mlp.up_proj.weight
[1024]                 BF16  talker.model.layers.0.post_attention_layernorm.weight
[128]                  BF16  talker.model.layers.0.self_attn.k_norm.weight
[1024, 1024]           BF16  talker.model.layers.0.self_attn.k_proj.weight
[1024, 2048]           BF16  talker.model.layers.0.self_attn.o_proj.weight
[128]                  BF16  talker.model.layers.0.self_attn.q_norm.weight
[2048, 1024]           BF16  talker.model.layers.0.self_attn.q_proj.weight
[1024, 1024]           BF16  talker.model.layers.0.self_attn.v_proj.weight
[1024]                 BF16  talker.model.layers.1.input_layernorm.weight
[1024, 3072]           BF16  talker.model.layers.1.mlp.down_proj.weight
[3072, 1024]           BF16  talker.model.layers.1.mlp.gate_proj.weight
[3072, 1024]           BF16  talker.model.layers.1.mlp.up_proj.weight
[1024]                 BF16  talker.model.layers.1.post_attention_layernorm.weight
[128]                  BF16  talker.model.layers.1.self_attn.k_norm.weight
[1024, 1024]           BF16  talker.model.layers.1.self_attn.k_proj.weight
[1024, 2048]           BF16  talker.model.layers.1.self_attn.o_proj.weight
[128]                  BF16  talker.model.layers.1.self_attn.q_norm.weight
[2048, 1024]           BF16  talker.model.layers.1.self_attn.q_proj.weight
[1024, 1024]           BF16  talker.model.layers.1.self_attn.v_proj.weight
[1024]                 BF16  talker.model.layers.10.input_layernorm.weight
[1024, 3072]           BF16  talker.model.layers.10.mlp.down_proj.weight
[3072, 1024]           BF16  talker.model.layers.10.mlp.gate_proj.weight
[3072, 1024]           BF16  talker.model.layers.10.mlp.up_proj.weight
[1024]                 BF16  talker.model.layers.10.post_attention_layernorm.weight
[128]                  BF16  talker.model.layers.10.self_attn.k_norm.weight
[1024, 1024]           BF16  talker.model.layers.10.self_attn.k_proj.weight
[1024, 2048]           BF16  talker.model.layers.10.self_attn.o_proj.weight
[128]                  BF16  talker.model.layers.10.self_attn.q_norm.weight
[2048, 1024]           BF16  talker.model.layers.10.self_attn.q_proj.weight
[1024, 1024]           BF16  talker.model.layers.10.self_attn.v_proj.weight
[1024]                 BF16  talker.model.layers.11.input_layernorm.weight
[1024, 3072]           BF16  talker.model.layers.11.mlp.down_proj.weight
[3072, 1024]           BF16  talker.model.layers.11.mlp.gate_proj.weight
[3072, 1024]           BF16  talker.model.layers.11.mlp.up_proj.weight
[1024]                 BF16  talker.model.layers.11.post_attention_layernorm.weight
[128]                  BF16  talker.model.layers.11.self_attn.k_norm.weight
[1024, 1024]           BF16  talker.model.layers.11.self_attn.k_proj.weight
[1024, 2048]           BF16  talker.model.layers.11.self_attn.o_proj.weight
[128]                  BF16  talker.model.layers.11.self_attn.q_norm.weight
[2048, 1024]           BF16  talker.model.layers.11.self_attn.q_proj.weight
[1024, 1024]           BF16  talker.model.layers.11.self_attn.v_proj.weight
[1024]                 BF16  talker.model.layers.12.input_layernorm.weight
[1024, 3072]           BF16  talker.model.layers.12.mlp.down_proj.weight
[3072, 1024]           BF16  talker.model.layers.12.mlp.gate_proj.weight
[3072, 1024]           BF16  talker.model.layers.12.mlp.up_proj.weight
[1024]                 BF16  talker.model.layers.12.post_attention_layernorm.weight
[128]                  BF16  talker.model.layers.12.self_attn.k_norm.weight
[1024, 1024]           BF16  talker.model.layers.12.self_attn.k_proj.weight
[1024, 2048]           BF16  talker.model.layers.12.self_attn.o_proj.weight
[128]                  BF16  talker.model.layers.12.self_attn.q_norm.weight
[2048, 1024]           BF16  talker.model.layers.12.self_attn.q_proj.weight
[1024, 1024]           BF16  talker.model.layers.12.self_attn.v_proj.weight
[1024]                 BF16  talker.model.layers.13.input_layernorm.weight
[1024, 3072]           BF16  talker.model.layers.13.mlp.down_proj.weight
[3072, 1024]           BF16  talker.model.layers.13.mlp.gate_proj.weight
[3072, 1024]           BF16  talker.model.layers.13.mlp.up_proj.weight
[1024]                 BF16  talker.model.layers.13.post_attention_layernorm.weight
[128]                  BF16  talker.model.layers.13.self_attn.k_norm.weight
[1024, 1024]           BF16  talker.model.layers.13.self_attn.k_proj.weight
[1024, 2048]           BF16  talker.model.layers.13.self_attn.o_proj.weight
[128]                  BF16  talker.model.layers.13.self_attn.q_norm.weight
[2048, 1024]           BF16  talker.model.layers.13.self_attn.q_proj.weight
[1024, 1024]           BF16  talker.model.layers.13.self_attn.v_proj.weight
[1024]                 BF16  talker.model.layers.14.input_layernorm.weight
[1024, 3072]           BF16  talker.model.layers.14.mlp.down_proj.weight
[3072, 1024]           BF16  talker.model.layers.14.mlp.gate_proj.weight
[3072, 1024]           BF16  talker.model.layers.14.mlp.up_proj.weight
[1024]                 BF16  talker.model.layers.14.post_attention_layernorm.weight
[128]                  BF16  talker.model.layers.14.self_attn.k_norm.weight
[1024, 1024]           BF16  talker.model.layers.14.self_attn.k_proj.weight
[1024, 2048]           BF16  talker.model.layers.14.self_attn.o_proj.weight
[128]                  BF16  talker.model.layers.14.self_attn.q_norm.weight
[2048, 1024]           BF16  talker.model.layers.14.self_attn.q_proj.weight
[1024, 1024]           BF16  talker.model.layers.14.self_attn.v_proj.weight
[1024]                 BF16  talker.model.layers.15.input_layernorm.weight
[1024, 3072]           BF16  talker.model.layers.15.mlp.down_proj.weight
[3072, 1024]           BF16  talker.model.layers.15.mlp.gate_proj.weight
[3072, 1024]           BF16  talker.model.layers.15.mlp.up_proj.weight
[1024]                 BF16  talker.model.layers.15.post_attention_layernorm.weight
[128]                  BF16  talker.model.layers.15.self_attn.k_norm.weight
[1024, 1024]           BF16  talker.model.layers.15.self_attn.k_proj.weight
[1024, 2048]           BF16  talker.model.layers.15.self_attn.o_proj.weight
[128]                  BF16  talker.model.layers.15.self_attn.q_norm.weight
[2048, 1024]           BF16  talker.model.layers.15.self_attn.q_proj.weight
[1024, 1024]           BF16  talker.model.layers.15.self_attn.v_proj.weight
[1024]                 BF16  talker.model.layers.16.input_layernorm.weight
[1024, 3072]           BF16  talker.model.layers.16.mlp.down_proj.weight
[3072, 1024]           BF16  talker.model.layers.16.mlp.gate_proj.weight
[3072, 1024]           BF16  talker.model.layers.16.mlp.up_proj.weight
[1024]                 BF16  talker.model.layers.16.post_attention_layernorm.weight
[128]                  BF16  talker.model.layers.16.self_attn.k_norm.weight
[1024, 1024]           BF16  talker.model.layers.16.self_attn.k_proj.weight
[1024, 2048]           BF16  talker.model.layers.16.self_attn.o_proj.weight
[128]                  BF16  talker.model.layers.16.self_attn.q_norm.weight
[2048, 1024]           BF16  talker.model.layers.16.self_attn.q_proj.weight
[1024, 1024]           BF16  talker.model.layers.16.self_attn.v_proj.weight
[1024]                 BF16  talker.model.layers.17.input_layernorm.weight
[1024, 3072]           BF16  talker.model.layers.17.mlp.down_proj.weight
[3072, 1024]           BF16  talker.model.layers.17.mlp.gate_proj.weight
[3072, 1024]           BF16  talker.model.layers.17.mlp.up_proj.weight
[1024]                 BF16  talker.model.layers.17.post_attention_layernorm.weight
[128]                  BF16  talker.model.layers.17.self_attn.k_norm.weight
[1024, 1024]           BF16  talker.model.layers.17.self_attn.k_proj.weight
[1024, 2048]           BF16  talker.model.layers.17.self_attn.o_proj.weight
[128]                  BF16  talker.model.layers.17.self_attn.q_norm.weight
[2048, 1024]           BF16  talker.model.layers.17.self_attn.q_proj.weight
[1024, 1024]           BF16  talker.model.layers.17.self_attn.v_proj.weight
[1024]                 BF16  talker.model.layers.18.input_layernorm.weight
[1024, 3072]           BF16  talker.model.layers.18.mlp.down_proj.weight
[3072, 1024]           BF16  talker.model.layers.18.mlp.gate_proj.weight
[3072, 1024]           BF16  talker.model.layers.18.mlp.up_proj.weight
[1024]                 BF16  talker.model.layers.18.post_attention_layernorm.weight
[128]                  BF16  talker.model.layers.18.self_attn.k_norm.weight
[1024, 1024]           BF16  talker.model.layers.18.self_attn.k_proj.weight
[1024, 2048]           BF16  talker.model.layers.18.self_attn.o_proj.weight
[128]                  BF16  talker.model.layers.18.self_attn.q_norm.weight
[2048, 1024]           BF16  talker.model.layers.18.self_attn.q_proj.weight
[1024, 1024]           BF16  talker.model.layers.18.self_attn.v_proj.weight
[1024]                 BF16  talker.model.layers.19.input_layernorm.weight
[1024, 3072]           BF16  talker.model.layers.19.mlp.down_proj.weight
[3072, 1024]           BF16  talker.model.layers.19.mlp.gate_proj.weight
[3072, 1024]           BF16  talker.model.layers.19.mlp.up_proj.weight
[1024]                 BF16  talker.model.layers.19.post_attention_layernorm.weight
[128]                  BF16  talker.model.layers.19.self_attn.k_norm.weight
[1024, 1024]           BF16  talker.model.layers.19.self_attn.k_proj.weight
[1024, 2048]           BF16  talker.model.layers.19.self_attn.o_proj.weight
[128]                  BF16  talker.model.layers.19.self_attn.q_norm.weight
[2048, 1024]           BF16  talker.model.layers.19.self_attn.q_proj.weight
[1024, 1024]           BF16  talker.model.layers.19.self_attn.v_proj.weight
[1024]                 BF16  talker.model.layers.2.input_layernorm.weight
[1024, 3072]           BF16  talker.model.layers.2.mlp.down_proj.weight
[3072, 1024]           BF16  talker.model.layers.2.mlp.gate_proj.weight
[3072, 1024]           BF16  talker.model.layers.2.mlp.up_proj.weight
[1024]                 BF16  talker.model.layers.2.post_attention_layernorm.weight
[128]                  BF16  talker.model.layers.2.self_attn.k_norm.weight
[1024, 1024]           BF16  talker.model.layers.2.self_attn.k_proj.weight
[1024, 2048]           BF16  talker.model.layers.2.self_attn.o_proj.weight
[128]                  BF16  talker.model.layers.2.self_attn.q_norm.weight
[2048, 1024]           BF16  talker.model.layers.2.self_attn.q_proj.weight
[1024, 1024]           BF16  talker.model.layers.2.self_attn.v_proj.weight
[1024]                 BF16  talker.model.layers.20.input_layernorm.weight
[1024, 3072]           BF16  talker.model.layers.20.mlp.down_proj.weight
[3072, 1024]           BF16  talker.model.layers.20.mlp.gate_proj.weight
[3072, 1024]           BF16  talker.model.layers.20.mlp.up_proj.weight
[1024]                 BF16  talker.model.layers.20.post_attention_layernorm.weight
[128]                  BF16  talker.model.layers.20.self_attn.k_norm.weight
[1024, 1024]           BF16  talker.model.layers.20.self_attn.k_proj.weight
[1024, 2048]           BF16  talker.model.layers.20.self_attn.o_proj.weight
[128]                  BF16  talker.model.layers.20.self_attn.q_norm.weight
[2048, 1024]           BF16  talker.model.layers.20.self_attn.q_proj.weight
[1024, 1024]           BF16  talker.model.layers.20.self_attn.v_proj.weight
[1024]                 BF16  talker.model.layers.21.input_layernorm.weight
[1024, 3072]           BF16  talker.model.layers.21.mlp.down_proj.weight
[3072, 1024]           BF16  talker.model.layers.21.mlp.gate_proj.weight
[3072, 1024]           BF16  talker.model.layers.21.mlp.up_proj.weight
[1024]                 BF16  talker.model.layers.21.post_attention_layernorm.weight
[128]                  BF16  talker.model.layers.21.self_attn.k_norm.weight
[1024, 1024]           BF16  talker.model.layers.21.self_attn.k_proj.weight
[1024, 2048]           BF16  talker.model.layers.21.self_attn.o_proj.weight
[128]                  BF16  talker.model.layers.21.self_attn.q_norm.weight
[2048, 1024]           BF16  talker.model.layers.21.self_attn.q_proj.weight
[1024, 1024]           BF16  talker.model.layers.21.self_attn.v_proj.weight
[1024]                 BF16  talker.model.layers.22.input_layernorm.weight
[1024, 3072]           BF16  talker.model.layers.22.mlp.down_proj.weight
[3072, 1024]           BF16  talker.model.layers.22.mlp.gate_proj.weight
[3072, 1024]           BF16  talker.model.layers.22.mlp.up_proj.weight
[1024]                 BF16  talker.model.layers.22.post_attention_layernorm.weight
[128]                  BF16  talker.model.layers.22.self_attn.k_norm.weight
[1024, 1024]           BF16  talker.model.layers.22.self_attn.k_proj.weight
[1024, 2048]           BF16  talker.model.layers.22.self_attn.o_proj.weight
[128]                  BF16  talker.model.layers.22.self_attn.q_norm.weight
[2048, 1024]           BF16  talker.model.layers.22.self_attn.q_proj.weight
[1024, 1024]           BF16  talker.model.layers.22.self_attn.v_proj.weight
[1024]                 BF16  talker.model.layers.23.input_layernorm.weight
[1024, 3072]           BF16  talker.model.layers.23.mlp.down_proj.weight
[3072, 1024]           BF16  talker.model.layers.23.mlp.gate_proj.weight
[3072, 1024]           BF16  talker.model.layers.23.mlp.up_proj.weight
[1024]                 BF16  talker.model.layers.23.post_attention_layernorm.weight
[128]                  BF16  talker.model.layers.23.self_attn.k_norm.weight
[1024, 1024]           BF16  talker.model.layers.23.self_attn.k_proj.weight
[1024, 2048]           BF16  talker.model.layers.23.self_attn.o_proj.weight
[128]                  BF16  talker.model.layers.23.self_attn.q_norm.weight
[2048, 1024]           BF16  talker.model.layers.23.self_attn.q_proj.weight
[1024, 1024]           BF16  talker.model.layers.23.self_attn.v_proj.weight
[1024]                 BF16  talker.model.layers.24.input_layernorm.weight
[1024, 3072]           BF16  talker.model.layers.24.mlp.down_proj.weight
[3072, 1024]           BF16  talker.model.layers.24.mlp.gate_proj.weight
[3072, 1024]           BF16  talker.model.layers.24.mlp.up_proj.weight
[1024]                 BF16  talker.model.layers.24.post_attention_layernorm.weight
[128]                  BF16  talker.model.layers.24.self_attn.k_norm.weight
[1024, 1024]           BF16  talker.model.layers.24.self_attn.k_proj.weight
[1024, 2048]           BF16  talker.model.layers.24.self_attn.o_proj.weight
[128]                  BF16  talker.model.layers.24.self_attn.q_norm.weight
[2048, 1024]           BF16  talker.model.layers.24.self_attn.q_proj.weight
[1024, 1024]           BF16  talker.model.layers.24.self_attn.v_proj.weight
[1024]                 BF16  talker.model.layers.25.input_layernorm.weight
[1024, 3072]           BF16  talker.model.layers.25.mlp.down_proj.weight
[3072, 1024]           BF16  talker.model.layers.25.mlp.gate_proj.weight
[3072, 1024]           BF16  talker.model.layers.25.mlp.up_proj.weight
[1024]                 BF16  talker.model.layers.25.post_attention_layernorm.weight
[128]                  BF16  talker.model.layers.25.self_attn.k_norm.weight
[1024, 1024]           BF16  talker.model.layers.25.self_attn.k_proj.weight
[1024, 2048]           BF16  talker.model.layers.25.self_attn.o_proj.weight
[128]                  BF16  talker.model.layers.25.self_attn.q_norm.weight
[2048, 1024]           BF16  talker.model.layers.25.self_attn.q_proj.weight
[1024, 1024]           BF16  talker.model.layers.25.self_attn.v_proj.weight
[1024]                 BF16  talker.model.layers.26.input_layernorm.weight
[1024, 3072]           BF16  talker.model.layers.26.mlp.down_proj.weight
[3072, 1024]           BF16  talker.model.layers.26.mlp.gate_proj.weight
[3072, 1024]           BF16  talker.model.layers.26.mlp.up_proj.weight
[1024]                 BF16  talker.model.layers.26.post_attention_layernorm.weight
[128]                  BF16  talker.model.layers.26.self_attn.k_norm.weight
[1024, 1024]           BF16  talker.model.layers.26.self_attn.k_proj.weight
[1024, 2048]           BF16  talker.model.layers.26.self_attn.o_proj.weight
[128]                  BF16  talker.model.layers.26.self_attn.q_norm.weight
[2048, 1024]           BF16  talker.model.layers.26.self_attn.q_proj.weight
[1024, 1024]           BF16  talker.model.layers.26.self_attn.v_proj.weight
[1024]                 BF16  talker.model.layers.27.input_layernorm.weight
[1024, 3072]           BF16  talker.model.layers.27.mlp.down_proj.weight
[3072, 1024]           BF16  talker.model.layers.27.mlp.gate_proj.weight
[3072, 1024]           BF16  talker.model.layers.27.mlp.up_proj.weight
[1024]                 BF16  talker.model.layers.27.post_attention_layernorm.weight
[128]                  BF16  talker.model.layers.27.self_attn.k_norm.weight
[1024, 1024]           BF16  talker.model.layers.27.self_attn.k_proj.weight
[1024, 2048]           BF16  talker.model.layers.27.self_attn.o_proj.weight
[128]                  BF16  talker.model.layers.27.self_attn.q_norm.weight
[2048, 1024]           BF16  talker.model.layers.27.self_attn.q_proj.weight
[1024, 1024]           BF16  talker.model.layers.27.self_attn.v_proj.weight
[1024]                 BF16  talker.model.layers.3.input_layernorm.weight
[1024, 3072]           BF16  talker.model.layers.3.mlp.down_proj.weight
[3072, 1024]           BF16  talker.model.layers.3.mlp.gate_proj.weight
[3072, 1024]           BF16  talker.model.layers.3.mlp.up_proj.weight
[1024]                 BF16  talker.model.layers.3.post_attention_layernorm.weight
[128]                  BF16  talker.model.layers.3.self_attn.k_norm.weight
[1024, 1024]           BF16  talker.model.layers.3.self_attn.k_proj.weight
[1024, 2048]           BF16  talker.model.layers.3.self_attn.o_proj.weight
[128]                  BF16  talker.model.layers.3.self_attn.q_norm.weight
[2048, 1024]           BF16  talker.model.layers.3.self_attn.q_proj.weight
[1024, 1024]           BF16  talker.model.layers.3.self_attn.v_proj.weight
[1024]                 BF16  talker.model.layers.4.input_layernorm.weight
[1024, 3072]           BF16  talker.model.layers.4.mlp.down_proj.weight
[3072, 1024]           BF16  talker.model.layers.4.mlp.gate_proj.weight
[3072, 1024]           BF16  talker.model.layers.4.mlp.up_proj.weight
[1024]                 BF16  talker.model.layers.4.post_attention_layernorm.weight
[128]                  BF16  talker.model.layers.4.self_attn.k_norm.weight
[1024, 1024]           BF16  talker.model.layers.4.self_attn.k_proj.weight
[1024, 2048]           BF16  talker.model.layers.4.self_attn.o_proj.weight
[128]                  BF16  talker.model.layers.4.self_attn.q_norm.weight
[2048, 1024]           BF16  talker.model.layers.4.self_attn.q_proj.weight
[1024, 1024]           BF16  talker.model.layers.4.self_attn.v_proj.weight
[1024]                 BF16  talker.model.layers.5.input_layernorm.weight
[1024, 3072]           BF16  talker.model.layers.5.mlp.down_proj.weight
[3072, 1024]           BF16  talker.model.layers.5.mlp.gate_proj.weight
[3072, 1024]           BF16  talker.model.layers.5.mlp.up_proj.weight
[1024]                 BF16  talker.model.layers.5.post_attention_layernorm.weight
[128]                  BF16  talker.model.layers.5.self_attn.k_norm.weight
[1024, 1024]           BF16  talker.model.layers.5.self_attn.k_proj.weight
[1024, 2048]           BF16  talker.model.layers.5.self_attn.o_proj.weight
[128]                  BF16  talker.model.layers.5.self_attn.q_norm.weight
[2048, 1024]           BF16  talker.model.layers.5.self_attn.q_proj.weight
[1024, 1024]           BF16  talker.model.layers.5.self_attn.v_proj.weight
[1024]                 BF16  talker.model.layers.6.input_layernorm.weight
[1024, 3072]           BF16  talker.model.layers.6.mlp.down_proj.weight
[3072, 1024]           BF16  talker.model.layers.6.mlp.gate_proj.weight
[3072, 1024]           BF16  talker.model.layers.6.mlp.up_proj.weight
[1024]                 BF16  talker.model.layers.6.post_attention_layernorm.weight
[128]                  BF16  talker.model.layers.6.self_attn.k_norm.weight
[1024, 1024]           BF16  talker.model.layers.6.self_attn.k_proj.weight
[1024, 2048]           BF16  talker.model.layers.6.self_attn.o_proj.weight
[128]                  BF16  talker.model.layers.6.self_attn.q_norm.weight
[2048, 1024]           BF16  talker.model.layers.6.self_attn.q_proj.weight
[1024, 1024]           BF16  talker.model.layers.6.self_attn.v_proj.weight
[1024]                 BF16  talker.model.layers.7.input_layernorm.weight
[1024, 3072]           BF16  talker.model.layers.7.mlp.down_proj.weight
[3072, 1024]           BF16  talker.model.layers.7.mlp.gate_proj.weight
[3072, 1024]           BF16  talker.model.layers.7.mlp.up_proj.weight
[1024]                 BF16  talker.model.layers.7.post_attention_layernorm.weight
[128]                  BF16  talker.model.layers.7.self_attn.k_norm.weight
[1024, 1024]           BF16  talker.model.layers.7.self_attn.k_proj.weight
[1024, 2048]           BF16  talker.model.layers.7.self_attn.o_proj.weight
[128]                  BF16  talker.model.layers.7.self_attn.q_norm.weight
[2048, 1024]           BF16  talker.model.layers.7.self_attn.q_proj.weight
[1024, 1024]           BF16  talker.model.layers.7.self_attn.v_proj.weight
[1024]                 BF16  talker.model.layers.8.input_layernorm.weight
[1024, 3072]           BF16  talker.model.layers.8.mlp.down_proj.weight
[3072, 1024]           BF16  talker.model.layers.8.mlp.gate_proj.weight
[3072, 1024]           BF16  talker.model.layers.8.mlp.up_proj.weight
[1024]                 BF16  talker.model.layers.8.post_attention_layernorm.weight
[128]                  BF16  talker.model.layers.8.self_attn.k_norm.weight
[1024, 1024]           BF16  talker.model.layers.8.self_attn.k_proj.weight
[1024, 2048]           BF16  talker.model.layers.8.self_attn.o_proj.weight
[128]                  BF16  talker.model.layers.8.self_attn.q_norm.weight
[2048, 1024]           BF16  talker.model.layers.8.self_attn.q_proj.weight
[1024, 1024]           BF16  talker.model.layers.8.self_attn.v_proj.weight
[1024]                 BF16  talker.model.layers.9.input_layernorm.weight
[1024, 3072]           BF16  talker.model.layers.9.mlp.down_proj.weight
[3072, 1024]           BF16  talker.model.layers.9.mlp.gate_proj.weight
[3072, 1024]           BF16  talker.model.layers.9.mlp.up_proj.weight
[1024]                 BF16  talker.model.layers.9.post_attention_layernorm.weight
[128]                  BF16  talker.model.layers.9.self_attn.k_norm.weight
[1024, 1024]           BF16  talker.model.layers.9.self_attn.k_proj.weight
[1024, 2048]           BF16  talker.model.layers.9.self_attn.o_proj.weight
[128]                  BF16  talker.model.layers.9.self_attn.q_norm.weight
[2048, 1024]           BF16  talker.model.layers.9.self_attn.q_proj.weight
[1024, 1024]           BF16  talker.model.layers.9.self_attn.v_proj.weight
[1024]                 BF16  talker.model.norm.weight
[151936, 2048]         BF16  talker.model.text_embedding.weight
[2048]                 BF16  talker.text_projection.linear_fc1.bias
[2048, 2048]           BF16  talker.text_projection.linear_fc1.weight
[1024]                 BF16  talker.text_projection.linear_fc2.bias
[1024, 2048]           BF16  talker.text_projection.linear_fc2.weight
```

## Codec (`speech_tokenizer/model.safetensors`) -- 496 tensors

```
[1536]                 F32   decoder.decoder.0.conv.bias
[1536, 1024, 7]        F32   decoder.decoder.0.conv.weight
[1536]                 F32   decoder.decoder.1.block.0.alpha
[1536]                 F32   decoder.decoder.1.block.0.beta
[768]                  F32   decoder.decoder.1.block.1.conv.bias
[1536, 768, 16]        F32   decoder.decoder.1.block.1.conv.weight
[768]                  F32   decoder.decoder.1.block.2.act1.alpha
[768]                  F32   decoder.decoder.1.block.2.act1.beta
[768]                  F32   decoder.decoder.1.block.2.act2.alpha
[768]                  F32   decoder.decoder.1.block.2.act2.beta
[768]                  F32   decoder.decoder.1.block.2.conv1.conv.bias
[768, 768, 7]          F32   decoder.decoder.1.block.2.conv1.conv.weight
[768]                  F32   decoder.decoder.1.block.2.conv2.conv.bias
[768, 768, 1]          F32   decoder.decoder.1.block.2.conv2.conv.weight
[768]                  F32   decoder.decoder.1.block.3.act1.alpha
[768]                  F32   decoder.decoder.1.block.3.act1.beta
[768]                  F32   decoder.decoder.1.block.3.act2.alpha
[768]                  F32   decoder.decoder.1.block.3.act2.beta
[768]                  F32   decoder.decoder.1.block.3.conv1.conv.bias
[768, 768, 7]          F32   decoder.decoder.1.block.3.conv1.conv.weight
[768]                  F32   decoder.decoder.1.block.3.conv2.conv.bias
[768, 768, 1]          F32   decoder.decoder.1.block.3.conv2.conv.weight
[768]                  F32   decoder.decoder.1.block.4.act1.alpha
[768]                  F32   decoder.decoder.1.block.4.act1.beta
[768]                  F32   decoder.decoder.1.block.4.act2.alpha
[768]                  F32   decoder.decoder.1.block.4.act2.beta
[768]                  F32   decoder.decoder.1.block.4.conv1.conv.bias
[768, 768, 7]          F32   decoder.decoder.1.block.4.conv1.conv.weight
[768]                  F32   decoder.decoder.1.block.4.conv2.conv.bias
[768, 768, 1]          F32   decoder.decoder.1.block.4.conv2.conv.weight
[768]                  F32   decoder.decoder.2.block.0.alpha
[768]                  F32   decoder.decoder.2.block.0.beta
[384]                  F32   decoder.decoder.2.block.1.conv.bias
[768, 384, 10]         F32   decoder.decoder.2.block.1.conv.weight
[384]                  F32   decoder.decoder.2.block.2.act1.alpha
[384]                  F32   decoder.decoder.2.block.2.act1.beta
[384]                  F32   decoder.decoder.2.block.2.act2.alpha
[384]                  F32   decoder.decoder.2.block.2.act2.beta
[384]                  F32   decoder.decoder.2.block.2.conv1.conv.bias
[384, 384, 7]          F32   decoder.decoder.2.block.2.conv1.conv.weight
[384]                  F32   decoder.decoder.2.block.2.conv2.conv.bias
[384, 384, 1]          F32   decoder.decoder.2.block.2.conv2.conv.weight
[384]                  F32   decoder.decoder.2.block.3.act1.alpha
[384]                  F32   decoder.decoder.2.block.3.act1.beta
[384]                  F32   decoder.decoder.2.block.3.act2.alpha
[384]                  F32   decoder.decoder.2.block.3.act2.beta
[384]                  F32   decoder.decoder.2.block.3.conv1.conv.bias
[384, 384, 7]          F32   decoder.decoder.2.block.3.conv1.conv.weight
[384]                  F32   decoder.decoder.2.block.3.conv2.conv.bias
[384, 384, 1]          F32   decoder.decoder.2.block.3.conv2.conv.weight
[384]                  F32   decoder.decoder.2.block.4.act1.alpha
[384]                  F32   decoder.decoder.2.block.4.act1.beta
[384]                  F32   decoder.decoder.2.block.4.act2.alpha
[384]                  F32   decoder.decoder.2.block.4.act2.beta
[384]                  F32   decoder.decoder.2.block.4.conv1.conv.bias
[384, 384, 7]          F32   decoder.decoder.2.block.4.conv1.conv.weight
[384]                  F32   decoder.decoder.2.block.4.conv2.conv.bias
[384, 384, 1]          F32   decoder.decoder.2.block.4.conv2.conv.weight
[384]                  F32   decoder.decoder.3.block.0.alpha
[384]                  F32   decoder.decoder.3.block.0.beta
[192]                  F32   decoder.decoder.3.block.1.conv.bias
[384, 192, 8]          F32   decoder.decoder.3.block.1.conv.weight
[192]                  F32   decoder.decoder.3.block.2.act1.alpha
[192]                  F32   decoder.decoder.3.block.2.act1.beta
[192]                  F32   decoder.decoder.3.block.2.act2.alpha
[192]                  F32   decoder.decoder.3.block.2.act2.beta
[192]                  F32   decoder.decoder.3.block.2.conv1.conv.bias
[192, 192, 7]          F32   decoder.decoder.3.block.2.conv1.conv.weight
[192]                  F32   decoder.decoder.3.block.2.conv2.conv.bias
[192, 192, 1]          F32   decoder.decoder.3.block.2.conv2.conv.weight
[192]                  F32   decoder.decoder.3.block.3.act1.alpha
[192]                  F32   decoder.decoder.3.block.3.act1.beta
[192]                  F32   decoder.decoder.3.block.3.act2.alpha
[192]                  F32   decoder.decoder.3.block.3.act2.beta
[192]                  F32   decoder.decoder.3.block.3.conv1.conv.bias
[192, 192, 7]          F32   decoder.decoder.3.block.3.conv1.conv.weight
[192]                  F32   decoder.decoder.3.block.3.conv2.conv.bias
[192, 192, 1]          F32   decoder.decoder.3.block.3.conv2.conv.weight
[192]                  F32   decoder.decoder.3.block.4.act1.alpha
[192]                  F32   decoder.decoder.3.block.4.act1.beta
[192]                  F32   decoder.decoder.3.block.4.act2.alpha
[192]                  F32   decoder.decoder.3.block.4.act2.beta
[192]                  F32   decoder.decoder.3.block.4.conv1.conv.bias
[192, 192, 7]          F32   decoder.decoder.3.block.4.conv1.conv.weight
[192]                  F32   decoder.decoder.3.block.4.conv2.conv.bias
[192, 192, 1]          F32   decoder.decoder.3.block.4.conv2.conv.weight
[192]                  F32   decoder.decoder.4.block.0.alpha
[192]                  F32   decoder.decoder.4.block.0.beta
[96]                   F32   decoder.decoder.4.block.1.conv.bias
[192, 96, 6]           F32   decoder.decoder.4.block.1.conv.weight
[96]                   F32   decoder.decoder.4.block.2.act1.alpha
[96]                   F32   decoder.decoder.4.block.2.act1.beta
[96]                   F32   decoder.decoder.4.block.2.act2.alpha
[96]                   F32   decoder.decoder.4.block.2.act2.beta
[96]                   F32   decoder.decoder.4.block.2.conv1.conv.bias
[96, 96, 7]            F32   decoder.decoder.4.block.2.conv1.conv.weight
[96]                   F32   decoder.decoder.4.block.2.conv2.conv.bias
[96, 96, 1]            F32   decoder.decoder.4.block.2.conv2.conv.weight
[96]                   F32   decoder.decoder.4.block.3.act1.alpha
[96]                   F32   decoder.decoder.4.block.3.act1.beta
[96]                   F32   decoder.decoder.4.block.3.act2.alpha
[96]                   F32   decoder.decoder.4.block.3.act2.beta
[96]                   F32   decoder.decoder.4.block.3.conv1.conv.bias
[96, 96, 7]            F32   decoder.decoder.4.block.3.conv1.conv.weight
[96]                   F32   decoder.decoder.4.block.3.conv2.conv.bias
[96, 96, 1]            F32   decoder.decoder.4.block.3.conv2.conv.weight
[96]                   F32   decoder.decoder.4.block.4.act1.alpha
[96]                   F32   decoder.decoder.4.block.4.act1.beta
[96]                   F32   decoder.decoder.4.block.4.act2.alpha
[96]                   F32   decoder.decoder.4.block.4.act2.beta
[96]                   F32   decoder.decoder.4.block.4.conv1.conv.bias
[96, 96, 7]            F32   decoder.decoder.4.block.4.conv1.conv.weight
[96]                   F32   decoder.decoder.4.block.4.conv2.conv.bias
[96, 96, 1]            F32   decoder.decoder.4.block.4.conv2.conv.weight
[96]                   F32   decoder.decoder.5.alpha
[96]                   F32   decoder.decoder.5.beta
[1]                    F32   decoder.decoder.6.conv.bias
[1, 96, 7]             F32   decoder.decoder.6.conv.weight
[1024]                 F32   decoder.pre_conv.conv.bias
[1024, 512, 3]         F32   decoder.pre_conv.conv.weight
[512]                  F32   decoder.pre_transformer.input_proj.bias
[512, 1024]            F32   decoder.pre_transformer.input_proj.weight
[512]                  F32   decoder.pre_transformer.layers.0.input_layernorm.weight
[512, 1024]            F32   decoder.pre_transformer.layers.0.mlp.down_proj.weight
[1024, 512]            F32   decoder.pre_transformer.layers.0.mlp.gate_proj.weight
[1024, 512]            F32   decoder.pre_transformer.layers.0.mlp.up_proj.weight
[512]                  F32   decoder.pre_transformer.layers.0.mlp_layer_scale.scale
[512]                  F32   decoder.pre_transformer.layers.0.post_attention_layernorm.weight
[1024, 512]            F32   decoder.pre_transformer.layers.0.self_attn.k_proj.weight
[512, 1024]            F32   decoder.pre_transformer.layers.0.self_attn.o_proj.weight
[1024, 512]            F32   decoder.pre_transformer.layers.0.self_attn.q_proj.weight
[1024, 512]            F32   decoder.pre_transformer.layers.0.self_attn.v_proj.weight
[512]                  F32   decoder.pre_transformer.layers.0.self_attn_layer_scale.scale
[512]                  F32   decoder.pre_transformer.layers.1.input_layernorm.weight
[512, 1024]            F32   decoder.pre_transformer.layers.1.mlp.down_proj.weight
[1024, 512]            F32   decoder.pre_transformer.layers.1.mlp.gate_proj.weight
[1024, 512]            F32   decoder.pre_transformer.layers.1.mlp.up_proj.weight
[512]                  F32   decoder.pre_transformer.layers.1.mlp_layer_scale.scale
[512]                  F32   decoder.pre_transformer.layers.1.post_attention_layernorm.weight
[1024, 512]            F32   decoder.pre_transformer.layers.1.self_attn.k_proj.weight
[512, 1024]            F32   decoder.pre_transformer.layers.1.self_attn.o_proj.weight
[1024, 512]            F32   decoder.pre_transformer.layers.1.self_attn.q_proj.weight
[1024, 512]            F32   decoder.pre_transformer.layers.1.self_attn.v_proj.weight
[512]                  F32   decoder.pre_transformer.layers.1.self_attn_layer_scale.scale
[512]                  F32   decoder.pre_transformer.layers.2.input_layernorm.weight
[512, 1024]            F32   decoder.pre_transformer.layers.2.mlp.down_proj.weight
[1024, 512]            F32   decoder.pre_transformer.layers.2.mlp.gate_proj.weight
[1024, 512]            F32   decoder.pre_transformer.layers.2.mlp.up_proj.weight
[512]                  F32   decoder.pre_transformer.layers.2.mlp_layer_scale.scale
[512]                  F32   decoder.pre_transformer.layers.2.post_attention_layernorm.weight
[1024, 512]            F32   decoder.pre_transformer.layers.2.self_attn.k_proj.weight
[512, 1024]            F32   decoder.pre_transformer.layers.2.self_attn.o_proj.weight
[1024, 512]            F32   decoder.pre_transformer.layers.2.self_attn.q_proj.weight
[1024, 512]            F32   decoder.pre_transformer.layers.2.self_attn.v_proj.weight
[512]                  F32   decoder.pre_transformer.layers.2.self_attn_layer_scale.scale
[512]                  F32   decoder.pre_transformer.layers.3.input_layernorm.weight
[512, 1024]            F32   decoder.pre_transformer.layers.3.mlp.down_proj.weight
[1024, 512]            F32   decoder.pre_transformer.layers.3.mlp.gate_proj.weight
[1024, 512]            F32   decoder.pre_transformer.layers.3.mlp.up_proj.weight
[512]                  F32   decoder.pre_transformer.layers.3.mlp_layer_scale.scale
[512]                  F32   decoder.pre_transformer.layers.3.post_attention_layernorm.weight
[1024, 512]            F32   decoder.pre_transformer.layers.3.self_attn.k_proj.weight
[512, 1024]            F32   decoder.pre_transformer.layers.3.self_attn.o_proj.weight
[1024, 512]            F32   decoder.pre_transformer.layers.3.self_attn.q_proj.weight
[1024, 512]            F32   decoder.pre_transformer.layers.3.self_attn.v_proj.weight
[512]                  F32   decoder.pre_transformer.layers.3.self_attn_layer_scale.scale
[512]                  F32   decoder.pre_transformer.layers.4.input_layernorm.weight
[512, 1024]            F32   decoder.pre_transformer.layers.4.mlp.down_proj.weight
[1024, 512]            F32   decoder.pre_transformer.layers.4.mlp.gate_proj.weight
[1024, 512]            F32   decoder.pre_transformer.layers.4.mlp.up_proj.weight
[512]                  F32   decoder.pre_transformer.layers.4.mlp_layer_scale.scale
[512]                  F32   decoder.pre_transformer.layers.4.post_attention_layernorm.weight
[1024, 512]            F32   decoder.pre_transformer.layers.4.self_attn.k_proj.weight
[512, 1024]            F32   decoder.pre_transformer.layers.4.self_attn.o_proj.weight
[1024, 512]            F32   decoder.pre_transformer.layers.4.self_attn.q_proj.weight
[1024, 512]            F32   decoder.pre_transformer.layers.4.self_attn.v_proj.weight
[512]                  F32   decoder.pre_transformer.layers.4.self_attn_layer_scale.scale
[512]                  F32   decoder.pre_transformer.layers.5.input_layernorm.weight
[512, 1024]            F32   decoder.pre_transformer.layers.5.mlp.down_proj.weight
[1024, 512]            F32   decoder.pre_transformer.layers.5.mlp.gate_proj.weight
[1024, 512]            F32   decoder.pre_transformer.layers.5.mlp.up_proj.weight
[512]                  F32   decoder.pre_transformer.layers.5.mlp_layer_scale.scale
[512]                  F32   decoder.pre_transformer.layers.5.post_attention_layernorm.weight
[1024, 512]            F32   decoder.pre_transformer.layers.5.self_attn.k_proj.weight
[512, 1024]            F32   decoder.pre_transformer.layers.5.self_attn.o_proj.weight
[1024, 512]            F32   decoder.pre_transformer.layers.5.self_attn.q_proj.weight
[1024, 512]            F32   decoder.pre_transformer.layers.5.self_attn.v_proj.weight
[512]                  F32   decoder.pre_transformer.layers.5.self_attn_layer_scale.scale
[512]                  F32   decoder.pre_transformer.layers.6.input_layernorm.weight
[512, 1024]            F32   decoder.pre_transformer.layers.6.mlp.down_proj.weight
[1024, 512]            F32   decoder.pre_transformer.layers.6.mlp.gate_proj.weight
[1024, 512]            F32   decoder.pre_transformer.layers.6.mlp.up_proj.weight
[512]                  F32   decoder.pre_transformer.layers.6.mlp_layer_scale.scale
[512]                  F32   decoder.pre_transformer.layers.6.post_attention_layernorm.weight
[1024, 512]            F32   decoder.pre_transformer.layers.6.self_attn.k_proj.weight
[512, 1024]            F32   decoder.pre_transformer.layers.6.self_attn.o_proj.weight
[1024, 512]            F32   decoder.pre_transformer.layers.6.self_attn.q_proj.weight
[1024, 512]            F32   decoder.pre_transformer.layers.6.self_attn.v_proj.weight
[512]                  F32   decoder.pre_transformer.layers.6.self_attn_layer_scale.scale
[512]                  F32   decoder.pre_transformer.layers.7.input_layernorm.weight
[512, 1024]            F32   decoder.pre_transformer.layers.7.mlp.down_proj.weight
[1024, 512]            F32   decoder.pre_transformer.layers.7.mlp.gate_proj.weight
[1024, 512]            F32   decoder.pre_transformer.layers.7.mlp.up_proj.weight
[512]                  F32   decoder.pre_transformer.layers.7.mlp_layer_scale.scale
[512]                  F32   decoder.pre_transformer.layers.7.post_attention_layernorm.weight
[1024, 512]            F32   decoder.pre_transformer.layers.7.self_attn.k_proj.weight
[512, 1024]            F32   decoder.pre_transformer.layers.7.self_attn.o_proj.weight
[1024, 512]            F32   decoder.pre_transformer.layers.7.self_attn.q_proj.weight
[1024, 512]            F32   decoder.pre_transformer.layers.7.self_attn.v_proj.weight
[512]                  F32   decoder.pre_transformer.layers.7.self_attn_layer_scale.scale
[512]                  F32   decoder.pre_transformer.norm.weight
[1024]                 F32   decoder.pre_transformer.output_proj.bias
[1024, 512]            F32   decoder.pre_transformer.output_proj.weight
[256, 512, 1]          F32   decoder.quantizer.rvq_first.input_proj.weight
[512, 256, 1]          F32   decoder.quantizer.rvq_first.output_proj.weight
[2048]                 F32   decoder.quantizer.rvq_first.vq.layers.0._codebook.cluster_usage
[2048, 256]            F32   decoder.quantizer.rvq_first.vq.layers.0._codebook.embedding_sum
[256, 512, 1]          F32   decoder.quantizer.rvq_rest.input_proj.weight
[512, 256, 1]          F32   decoder.quantizer.rvq_rest.output_proj.weight
[2048]                 F32   decoder.quantizer.rvq_rest.vq.layers.0._codebook.cluster_usage
[2048, 256]            F32   decoder.quantizer.rvq_rest.vq.layers.0._codebook.embedding_sum
[2048]                 F32   decoder.quantizer.rvq_rest.vq.layers.1._codebook.cluster_usage
[2048, 256]            F32   decoder.quantizer.rvq_rest.vq.layers.1._codebook.embedding_sum
[2048]                 F32   decoder.quantizer.rvq_rest.vq.layers.10._codebook.cluster_usage
[2048, 256]            F32   decoder.quantizer.rvq_rest.vq.layers.10._codebook.embedding_sum
[2048]                 F32   decoder.quantizer.rvq_rest.vq.layers.11._codebook.cluster_usage
[2048, 256]            F32   decoder.quantizer.rvq_rest.vq.layers.11._codebook.embedding_sum
[2048]                 F32   decoder.quantizer.rvq_rest.vq.layers.12._codebook.cluster_usage
[2048, 256]            F32   decoder.quantizer.rvq_rest.vq.layers.12._codebook.embedding_sum
[2048]                 F32   decoder.quantizer.rvq_rest.vq.layers.13._codebook.cluster_usage
[2048, 256]            F32   decoder.quantizer.rvq_rest.vq.layers.13._codebook.embedding_sum
[2048]                 F32   decoder.quantizer.rvq_rest.vq.layers.14._codebook.cluster_usage
[2048, 256]            F32   decoder.quantizer.rvq_rest.vq.layers.14._codebook.embedding_sum
[2048]                 F32   decoder.quantizer.rvq_rest.vq.layers.2._codebook.cluster_usage
[2048, 256]            F32   decoder.quantizer.rvq_rest.vq.layers.2._codebook.embedding_sum
[2048]                 F32   decoder.quantizer.rvq_rest.vq.layers.3._codebook.cluster_usage
[2048, 256]            F32   decoder.quantizer.rvq_rest.vq.layers.3._codebook.embedding_sum
[2048]                 F32   decoder.quantizer.rvq_rest.vq.layers.4._codebook.cluster_usage
[2048, 256]            F32   decoder.quantizer.rvq_rest.vq.layers.4._codebook.embedding_sum
[2048]                 F32   decoder.quantizer.rvq_rest.vq.layers.5._codebook.cluster_usage
[2048, 256]            F32   decoder.quantizer.rvq_rest.vq.layers.5._codebook.embedding_sum
[2048]                 F32   decoder.quantizer.rvq_rest.vq.layers.6._codebook.cluster_usage
[2048, 256]            F32   decoder.quantizer.rvq_rest.vq.layers.6._codebook.embedding_sum
[2048]                 F32   decoder.quantizer.rvq_rest.vq.layers.7._codebook.cluster_usage
[2048, 256]            F32   decoder.quantizer.rvq_rest.vq.layers.7._codebook.embedding_sum
[2048]                 F32   decoder.quantizer.rvq_rest.vq.layers.8._codebook.cluster_usage
[2048, 256]            F32   decoder.quantizer.rvq_rest.vq.layers.8._codebook.embedding_sum
[2048]                 F32   decoder.quantizer.rvq_rest.vq.layers.9._codebook.cluster_usage
[2048, 256]            F32   decoder.quantizer.rvq_rest.vq.layers.9._codebook.embedding_sum
[1024]                 F32   decoder.upsample.0.0.conv.bias
[1024, 1024, 2]        F32   decoder.upsample.0.0.conv.weight
[1024]                 F32   decoder.upsample.0.1.dwconv.conv.bias
[1024, 1, 7]           F32   decoder.upsample.0.1.dwconv.conv.weight
[1024]                 F32   decoder.upsample.0.1.gamma
[1024]                 F32   decoder.upsample.0.1.norm.bias
[1024]                 F32   decoder.upsample.0.1.norm.weight
[4096]                 F32   decoder.upsample.0.1.pwconv1.bias
[4096, 1024]           F32   decoder.upsample.0.1.pwconv1.weight
[1024]                 F32   decoder.upsample.0.1.pwconv2.bias
[1024, 4096]           F32   decoder.upsample.0.1.pwconv2.weight
[1024]                 F32   decoder.upsample.1.0.conv.bias
[1024, 1024, 2]        F32   decoder.upsample.1.0.conv.weight
[1024]                 F32   decoder.upsample.1.1.dwconv.conv.bias
[1024, 1, 7]           F32   decoder.upsample.1.1.dwconv.conv.weight
[1024]                 F32   decoder.upsample.1.1.gamma
[1024]                 F32   decoder.upsample.1.1.norm.bias
[1024]                 F32   decoder.upsample.1.1.norm.weight
[4096]                 F32   decoder.upsample.1.1.pwconv1.bias
[4096, 1024]           F32   decoder.upsample.1.1.pwconv1.weight
[1024]                 F32   decoder.upsample.1.1.pwconv2.bias
[1024, 4096]           F32   decoder.upsample.1.1.pwconv2.weight
[512, 512, 4]          F32   encoder.downsample.conv.weight
[64]                   F32   encoder.encoder.layers.0.conv.bias
[64, 1, 7]             F32   encoder.encoder.layers.0.conv.weight
[32]                   F32   encoder.encoder.layers.1.block.1.conv.bias
[32, 64, 3]            F32   encoder.encoder.layers.1.block.1.conv.weight
[64]                   F32   encoder.encoder.layers.1.block.3.conv.bias
[64, 32, 1]            F32   encoder.encoder.layers.1.block.3.conv.weight
[256]                  F32   encoder.encoder.layers.10.block.1.conv.bias
[256, 512, 3]          F32   encoder.encoder.layers.10.block.1.conv.weight
[512]                  F32   encoder.encoder.layers.10.block.3.conv.bias
[512, 256, 1]          F32   encoder.encoder.layers.10.block.3.conv.weight
[1024]                 F32   encoder.encoder.layers.12.conv.bias
[1024, 512, 16]        F32   encoder.encoder.layers.12.conv.weight
[512]                  F32   encoder.encoder.layers.14.conv.bias
[512, 1024, 3]         F32   encoder.encoder.layers.14.conv.weight
[128]                  F32   encoder.encoder.layers.3.conv.bias
[128, 64, 8]           F32   encoder.encoder.layers.3.conv.weight
[64]                   F32   encoder.encoder.layers.4.block.1.conv.bias
[64, 128, 3]           F32   encoder.encoder.layers.4.block.1.conv.weight
[128]                  F32   encoder.encoder.layers.4.block.3.conv.bias
[128, 64, 1]           F32   encoder.encoder.layers.4.block.3.conv.weight
[256]                  F32   encoder.encoder.layers.6.conv.bias
[256, 128, 10]         F32   encoder.encoder.layers.6.conv.weight
[128]                  F32   encoder.encoder.layers.7.block.1.conv.bias
[128, 256, 3]          F32   encoder.encoder.layers.7.block.1.conv.weight
[256]                  F32   encoder.encoder.layers.7.block.3.conv.bias
[256, 128, 1]          F32   encoder.encoder.layers.7.block.3.conv.weight
[512]                  F32   encoder.encoder.layers.9.conv.bias
[512, 256, 12]         F32   encoder.encoder.layers.9.conv.weight
[512]                  F32   encoder.encoder_transformer.layers.0.input_layernorm.bias
[512]                  F32   encoder.encoder_transformer.layers.0.input_layernorm.weight
[2048, 512]            F32   encoder.encoder_transformer.layers.0.mlp.fc1.weight
[512, 2048]            F32   encoder.encoder_transformer.layers.0.mlp.fc2.weight
[512]                  F32   encoder.encoder_transformer.layers.0.mlp_layer_scale.scale
[512]                  F32   encoder.encoder_transformer.layers.0.post_attention_layernorm.bias
[512]                  F32   encoder.encoder_transformer.layers.0.post_attention_layernorm.weight
[512, 512]             F32   encoder.encoder_transformer.layers.0.self_attn.k_proj.weight
[512, 512]             F32   encoder.encoder_transformer.layers.0.self_attn.o_proj.weight
[512, 512]             F32   encoder.encoder_transformer.layers.0.self_attn.q_proj.weight
[512, 512]             F32   encoder.encoder_transformer.layers.0.self_attn.v_proj.weight
[512]                  F32   encoder.encoder_transformer.layers.0.self_attn_layer_scale.scale
[512]                  F32   encoder.encoder_transformer.layers.1.input_layernorm.bias
[512]                  F32   encoder.encoder_transformer.layers.1.input_layernorm.weight
[2048, 512]            F32   encoder.encoder_transformer.layers.1.mlp.fc1.weight
[512, 2048]            F32   encoder.encoder_transformer.layers.1.mlp.fc2.weight
[512]                  F32   encoder.encoder_transformer.layers.1.mlp_layer_scale.scale
[512]                  F32   encoder.encoder_transformer.layers.1.post_attention_layernorm.bias
[512]                  F32   encoder.encoder_transformer.layers.1.post_attention_layernorm.weight
[512, 512]             F32   encoder.encoder_transformer.layers.1.self_attn.k_proj.weight
[512, 512]             F32   encoder.encoder_transformer.layers.1.self_attn.o_proj.weight
[512, 512]             F32   encoder.encoder_transformer.layers.1.self_attn.q_proj.weight
[512, 512]             F32   encoder.encoder_transformer.layers.1.self_attn.v_proj.weight
[512]                  F32   encoder.encoder_transformer.layers.1.self_attn_layer_scale.scale
[512]                  F32   encoder.encoder_transformer.layers.2.input_layernorm.bias
[512]                  F32   encoder.encoder_transformer.layers.2.input_layernorm.weight
[2048, 512]            F32   encoder.encoder_transformer.layers.2.mlp.fc1.weight
[512, 2048]            F32   encoder.encoder_transformer.layers.2.mlp.fc2.weight
[512]                  F32   encoder.encoder_transformer.layers.2.mlp_layer_scale.scale
[512]                  F32   encoder.encoder_transformer.layers.2.post_attention_layernorm.bias
[512]                  F32   encoder.encoder_transformer.layers.2.post_attention_layernorm.weight
[512, 512]             F32   encoder.encoder_transformer.layers.2.self_attn.k_proj.weight
[512, 512]             F32   encoder.encoder_transformer.layers.2.self_attn.o_proj.weight
[512, 512]             F32   encoder.encoder_transformer.layers.2.self_attn.q_proj.weight
[512, 512]             F32   encoder.encoder_transformer.layers.2.self_attn.v_proj.weight
[512]                  F32   encoder.encoder_transformer.layers.2.self_attn_layer_scale.scale
[512]                  F32   encoder.encoder_transformer.layers.3.input_layernorm.bias
[512]                  F32   encoder.encoder_transformer.layers.3.input_layernorm.weight
[2048, 512]            F32   encoder.encoder_transformer.layers.3.mlp.fc1.weight
[512, 2048]            F32   encoder.encoder_transformer.layers.3.mlp.fc2.weight
[512]                  F32   encoder.encoder_transformer.layers.3.mlp_layer_scale.scale
[512]                  F32   encoder.encoder_transformer.layers.3.post_attention_layernorm.bias
[512]                  F32   encoder.encoder_transformer.layers.3.post_attention_layernorm.weight
[512, 512]             F32   encoder.encoder_transformer.layers.3.self_attn.k_proj.weight
[512, 512]             F32   encoder.encoder_transformer.layers.3.self_attn.o_proj.weight
[512, 512]             F32   encoder.encoder_transformer.layers.3.self_attn.q_proj.weight
[512, 512]             F32   encoder.encoder_transformer.layers.3.self_attn.v_proj.weight
[512]                  F32   encoder.encoder_transformer.layers.3.self_attn_layer_scale.scale
[512]                  F32   encoder.encoder_transformer.layers.4.input_layernorm.bias
[512]                  F32   encoder.encoder_transformer.layers.4.input_layernorm.weight
[2048, 512]            F32   encoder.encoder_transformer.layers.4.mlp.fc1.weight
[512, 2048]            F32   encoder.encoder_transformer.layers.4.mlp.fc2.weight
[512]                  F32   encoder.encoder_transformer.layers.4.mlp_layer_scale.scale
[512]                  F32   encoder.encoder_transformer.layers.4.post_attention_layernorm.bias
[512]                  F32   encoder.encoder_transformer.layers.4.post_attention_layernorm.weight
[512, 512]             F32   encoder.encoder_transformer.layers.4.self_attn.k_proj.weight
[512, 512]             F32   encoder.encoder_transformer.layers.4.self_attn.o_proj.weight
[512, 512]             F32   encoder.encoder_transformer.layers.4.self_attn.q_proj.weight
[512, 512]             F32   encoder.encoder_transformer.layers.4.self_attn.v_proj.weight
[512]                  F32   encoder.encoder_transformer.layers.4.self_attn_layer_scale.scale
[512]                  F32   encoder.encoder_transformer.layers.5.input_layernorm.bias
[512]                  F32   encoder.encoder_transformer.layers.5.input_layernorm.weight
[2048, 512]            F32   encoder.encoder_transformer.layers.5.mlp.fc1.weight
[512, 2048]            F32   encoder.encoder_transformer.layers.5.mlp.fc2.weight
[512]                  F32   encoder.encoder_transformer.layers.5.mlp_layer_scale.scale
[512]                  F32   encoder.encoder_transformer.layers.5.post_attention_layernorm.bias
[512]                  F32   encoder.encoder_transformer.layers.5.post_attention_layernorm.weight
[512, 512]             F32   encoder.encoder_transformer.layers.5.self_attn.k_proj.weight
[512, 512]             F32   encoder.encoder_transformer.layers.5.self_attn.o_proj.weight
[512, 512]             F32   encoder.encoder_transformer.layers.5.self_attn.q_proj.weight
[512, 512]             F32   encoder.encoder_transformer.layers.5.self_attn.v_proj.weight
[512]                  F32   encoder.encoder_transformer.layers.5.self_attn_layer_scale.scale
[512]                  F32   encoder.encoder_transformer.layers.6.input_layernorm.bias
[512]                  F32   encoder.encoder_transformer.layers.6.input_layernorm.weight
[2048, 512]            F32   encoder.encoder_transformer.layers.6.mlp.fc1.weight
[512, 2048]            F32   encoder.encoder_transformer.layers.6.mlp.fc2.weight
[512]                  F32   encoder.encoder_transformer.layers.6.mlp_layer_scale.scale
[512]                  F32   encoder.encoder_transformer.layers.6.post_attention_layernorm.bias
[512]                  F32   encoder.encoder_transformer.layers.6.post_attention_layernorm.weight
[512, 512]             F32   encoder.encoder_transformer.layers.6.self_attn.k_proj.weight
[512, 512]             F32   encoder.encoder_transformer.layers.6.self_attn.o_proj.weight
[512, 512]             F32   encoder.encoder_transformer.layers.6.self_attn.q_proj.weight
[512, 512]             F32   encoder.encoder_transformer.layers.6.self_attn.v_proj.weight
[512]                  F32   encoder.encoder_transformer.layers.6.self_attn_layer_scale.scale
[512]                  F32   encoder.encoder_transformer.layers.7.input_layernorm.bias
[512]                  F32   encoder.encoder_transformer.layers.7.input_layernorm.weight
[2048, 512]            F32   encoder.encoder_transformer.layers.7.mlp.fc1.weight
[512, 2048]            F32   encoder.encoder_transformer.layers.7.mlp.fc2.weight
[512]                  F32   encoder.encoder_transformer.layers.7.mlp_layer_scale.scale
[512]                  F32   encoder.encoder_transformer.layers.7.post_attention_layernorm.bias
[512]                  F32   encoder.encoder_transformer.layers.7.post_attention_layernorm.weight
[512, 512]             F32   encoder.encoder_transformer.layers.7.self_attn.k_proj.weight
[512, 512]             F32   encoder.encoder_transformer.layers.7.self_attn.o_proj.weight
[512, 512]             F32   encoder.encoder_transformer.layers.7.self_attn.q_proj.weight
[512, 512]             F32   encoder.encoder_transformer.layers.7.self_attn.v_proj.weight
[512]                  F32   encoder.encoder_transformer.layers.7.self_attn_layer_scale.scale
[256, 512, 1]          F32   encoder.quantizer.acoustic_residual_vector_quantizer.input_proj.weight
[2048]                 F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.0.codebook.cluster_usage
[2048, 256]            F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.0.codebook.embed_sum
[1]                    F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.0.codebook.initialized
[2048]                 F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.1.codebook.cluster_usage
[2048, 256]            F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.1.codebook.embed_sum
[1]                    F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.1.codebook.initialized
[2048]                 F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.10.codebook.cluster_usage
[2048, 256]            F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.10.codebook.embed_sum
[1]                    F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.10.codebook.initialized
[2048]                 F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.11.codebook.cluster_usage
[2048, 256]            F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.11.codebook.embed_sum
[1]                    F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.11.codebook.initialized
[2048]                 F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.12.codebook.cluster_usage
[2048, 256]            F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.12.codebook.embed_sum
[1]                    F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.12.codebook.initialized
[2048]                 F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.13.codebook.cluster_usage
[2048, 256]            F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.13.codebook.embed_sum
[1]                    F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.13.codebook.initialized
[2048]                 F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.14.codebook.cluster_usage
[2048, 256]            F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.14.codebook.embed_sum
[1]                    F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.14.codebook.initialized
[2048]                 F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.15.codebook.cluster_usage
[2048, 256]            F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.15.codebook.embed_sum
[1]                    F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.15.codebook.initialized
[2048]                 F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.16.codebook.cluster_usage
[2048, 256]            F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.16.codebook.embed_sum
[1]                    F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.16.codebook.initialized
[2048]                 F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.17.codebook.cluster_usage
[2048, 256]            F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.17.codebook.embed_sum
[1]                    F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.17.codebook.initialized
[2048]                 F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.18.codebook.cluster_usage
[2048, 256]            F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.18.codebook.embed_sum
[1]                    F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.18.codebook.initialized
[2048]                 F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.19.codebook.cluster_usage
[2048, 256]            F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.19.codebook.embed_sum
[1]                    F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.19.codebook.initialized
[2048]                 F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.2.codebook.cluster_usage
[2048, 256]            F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.2.codebook.embed_sum
[1]                    F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.2.codebook.initialized
[2048]                 F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.20.codebook.cluster_usage
[2048, 256]            F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.20.codebook.embed_sum
[1]                    F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.20.codebook.initialized
[2048]                 F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.21.codebook.cluster_usage
[2048, 256]            F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.21.codebook.embed_sum
[1]                    F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.21.codebook.initialized
[2048]                 F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.22.codebook.cluster_usage
[2048, 256]            F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.22.codebook.embed_sum
[1]                    F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.22.codebook.initialized
[2048]                 F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.23.codebook.cluster_usage
[2048, 256]            F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.23.codebook.embed_sum
[1]                    F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.23.codebook.initialized
[2048]                 F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.24.codebook.cluster_usage
[2048, 256]            F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.24.codebook.embed_sum
[1]                    F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.24.codebook.initialized
[2048]                 F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.25.codebook.cluster_usage
[2048, 256]            F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.25.codebook.embed_sum
[1]                    F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.25.codebook.initialized
[2048]                 F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.26.codebook.cluster_usage
[2048, 256]            F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.26.codebook.embed_sum
[1]                    F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.26.codebook.initialized
[2048]                 F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.27.codebook.cluster_usage
[2048, 256]            F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.27.codebook.embed_sum
[1]                    F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.27.codebook.initialized
[2048]                 F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.28.codebook.cluster_usage
[2048, 256]            F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.28.codebook.embed_sum
[1]                    F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.28.codebook.initialized
[2048]                 F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.29.codebook.cluster_usage
[2048, 256]            F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.29.codebook.embed_sum
[1]                    F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.29.codebook.initialized
[2048]                 F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.3.codebook.cluster_usage
[2048, 256]            F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.3.codebook.embed_sum
[1]                    F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.3.codebook.initialized
[2048]                 F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.30.codebook.cluster_usage
[2048, 256]            F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.30.codebook.embed_sum
[1]                    F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.30.codebook.initialized
[2048]                 F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.4.codebook.cluster_usage
[2048, 256]            F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.4.codebook.embed_sum
[1]                    F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.4.codebook.initialized
[2048]                 F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.5.codebook.cluster_usage
[2048, 256]            F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.5.codebook.embed_sum
[1]                    F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.5.codebook.initialized
[2048]                 F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.6.codebook.cluster_usage
[2048, 256]            F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.6.codebook.embed_sum
[1]                    F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.6.codebook.initialized
[2048]                 F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.7.codebook.cluster_usage
[2048, 256]            F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.7.codebook.embed_sum
[1]                    F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.7.codebook.initialized
[2048]                 F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.8.codebook.cluster_usage
[2048, 256]            F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.8.codebook.embed_sum
[1]                    F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.8.codebook.initialized
[2048]                 F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.9.codebook.cluster_usage
[2048, 256]            F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.9.codebook.embed_sum
[1]                    F32   encoder.quantizer.acoustic_residual_vector_quantizer.layers.9.codebook.initialized
[512, 256, 1]          F32   encoder.quantizer.acoustic_residual_vector_quantizer.output_proj.weight
[256, 512, 1]          F32   encoder.quantizer.semantic_residual_vector_quantizer.input_proj.weight
[2048]                 F32   encoder.quantizer.semantic_residual_vector_quantizer.layers.0.codebook.cluster_usage
[2048, 256]            F32   encoder.quantizer.semantic_residual_vector_quantizer.layers.0.codebook.embed_sum
[1]                    F32   encoder.quantizer.semantic_residual_vector_quantizer.layers.0.codebook.initialized
[512, 256, 1]          F32   encoder.quantizer.semantic_residual_vector_quantizer.output_proj.weight
```
