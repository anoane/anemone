# SPDX-License-Identifier: AGPL-3.0-or-later
# source/models/deepseek_v41_flash/: DeepSeek-V4.1-Flash.
# One module with its part folders. What the layout cannot say: the modules it may include, its prefix, words,
# tensor patterns and distinct facts (the lints), and its kernels' build classes, required traps and
# register-checked instances, which the root Makefile gathers.
models/deepseek_v41_flash_REQUIRES := common engine/backend engine/topology engine/program engine/formats/weights \
    engine/formats/digest engine/formats/pack engine/formats/exl3 engine/library engine/library/block_scaled_linear \
    engine/library/bf16_linear engine/library/exl3 engine/library/moe engine/library/collectives engine/library/rings \
    engine/library/wire_codecs engine/library/tensor_parallel engine/library/sampling engine/storage engine/storage/io \
    engine/storage/load engine/engram engine/input/tokenizer engine/input/image models
MODEL_PREFIX_deepseek_v41_flash := ds41f
MODEL_WORDS_deepseek_v41_flash := deepseek ds41f dspark dsml v41
MODEL_TENSOR_PATTERNS_deepseek_v41_flash := layers\.[0-9]+\.ffn\.experts markov_head confidence_head hc_attn hc_ffn \
    tid2eid mtp\.[0-9] attn\.wq_a attn\.wkv
MODEL_DISTINCT_FACTS_deepseek_v41_flash := 5120 2304 129280 129264 128799 15360 5376 2414 3968
NATIVE_KERNEL_SOURCES += source/models/deepseek_v41_flash/attention/ds41f_attention_core.cu
NATIVE_KERNEL_SOURCES += source/models/deepseek_v41_flash/attention/ds41f_attention_output.cu
REGISTER_CHECKED_INSTANCES += attention_(groups|projection)
WIDE_SLICE_INSTANCES += attention_groups
REQUIRED_TRAPS_ds41f_draft_share := draft_share_wait
NATIVE_KERNEL_SOURCES += source/models/deepseek_v41_flash/indexer/ds41f_index_scan.cu
