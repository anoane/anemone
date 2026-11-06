<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# Credits and reference projects

Anemone benefits from the work shared by the open-source inference community. Thank you to every
maintainer, original author, contributor and reviewer behind the projects, commits and pull requests
listed here, and to the people who published model weights, quantizations, benchmarks and hardware findings.

## Special thanks

- **Curtis [@BBacktesting](https://x.com/BBacktesting)**, for the [DeepSeek EXL3 serving and optimization work](https://github.com/CurtisAccelerate/DeepSeekv41-Flash-1017-EXL3-CMP170HX), component experiments and public benchmark methods.
- **Morrow [@Morrowmake](https://x.com/Morrowmake)**, for [CMP 170HX inference kernels and serving work](https://github.com/Morrowmake/vllm-cmp170hx), [reproducible recipes](https://github.com/Morrowmake/dsv41-flash-cmp170hx-recipe) and community references.
- **GNmac [@GNmac](https://x.com/GNmac)**, for [DeepSeek CMP 170HX serving](https://github.com/GNmac-lab/deepseek-v41-flash-1017-cmp170hx), communication and deployment research.
- **Sean Phan [@seanphan](https://x.com/seanphan)**, and the [PixelML](https://github.com/PixelML) community, for [PixelML/club-170hx](https://github.com/PixelML/club-170hx), shared inference experiments, SM80 work and practical findings.
- **Turboderp** ([@turboderp_](https://x.com/turboderp_)) — [ExLlamaV3](https://github.com/turboderp-org/exllamav3).
- Members of the private **CMP 170HX Owners** group chat on X.
- The [**cmpunlocker Discord channel**](https://discord.gg/CdHSakKSFv).

DeepSeek supplies the model, Engram and DSpark foundations; LibertAI and other quantization authors
supply their respective releases. Turboderp and ExLlamaV3 contributors supply the EXL3 format and
important expert-kernel references. The full project and model credits below include these upstream communities.

## Reference scope

This inventory records **152 distinct public reference origins**, with repeated dated checkouts
combined under each project. Revision links identify the source snapshots present in the reference
library. Repository contributor pages and original commit pages preserve the wider authorship record.
The collection includes source research, comparison baselines, related recipes, model cards and earlier
Anemone projects. The [comparison matrix](results/COMPARISON_MATRIX.md) records feature-specific findings
and measured integration; this inventory records the broader body of work consulted.

Credit extends to contributors whose names are absent from the short lists here, including co-authors,
reviewers, issue reporters and maintainers. Each project retains its own license and attribution terms.

## Specific source and contribution records

- [radixark/miles PR #3135](https://github.com/radixark/miles/pull/3135), with the [recorded PR head](https://github.com/radixark/miles/commit/c0fae0084c30955a84f0c4f5d7a6baf823ce9f35).
- [AlpinDale’s sonar PR #1788](https://github.com/dphnAI/sonar/pull/1788), a reference for native SM120 DeepSeek work.
- [bzantium’s FlashMLA PR #183](https://github.com/deepseek-ai/FlashMLA/pull/183), for SM80 dense MLA decode research.
- [Curtis’s detailed prior-work record](https://github.com/CurtisAccelerate/DeepSeekv41-Flash-EXL3-TC-MoE/blob/main/experiments/ds1017-native-mixed-and-split/PRIOR-WORK.md), which distinguishes his DeepSeek adaptations, six-row transpose, native expert specialization and serving integration from upstream operator foundations.

That prior-work record credits Morrowmake’s [tensor-core MoE and prefill foundations](https://github.com/Morrowmake/vllm-cmp170hx/tree/18305948acba3f4c4d912cdd56f2202d2ebd94e8/csrc/exl3)
and Turboderp’s [cooperative expert kernels](https://github.com/turboderp-org/exllamav3/commit/da68d57de61781e5b74ad68698ac4caf36408d7a).
Its attention lineage also credits Hexiang Wang and vLLM ROCm contributors ([sparse MLA](https://github.com/vllm-project/vllm/commit/7863fff6e591161a03a7df68b0879b9fa61cfc0b)),
Fangzhou Ai and vLLM contributors ([split-K decode](https://github.com/vllm-project/vllm/commit/fcf5115c45b9acfe3a77052ddbb7dfb0f4d5ef18),
[reduction occupancy](https://github.com/vllm-project/vllm/commit/efed8a1e8345de2b4398a48243301d6a5b1a361e)),
Tuukka Sarvi ([split-decode enablement](https://github.com/vllm-project/vllm/commit/b8168e33e000df5c60d63d7ca244ba2ca2c34079)),
and haosdent, thomaslwang and PixelML/sm80vllm contributors ([SM80 and FP8 adaptation](https://github.com/PixelML/sm80vllm/commit/f8ea5bb163c161ef38b401d055cc5fd4a934091a)).

## Project and model references

### Community inference engines and serving recipes

| Project or release | Recorded source revisions | Contributors |
| --- | --- | --- |
| [05yuki/dsv41-flash-dual-epyc](https://github.com/05yuki/dsv41-flash-dual-epyc) | [2d8e455658](https://github.com/05yuki/dsv41-flash-dual-epyc/commit/2d8e4556584baf50e2b660c34d308108ef3d1216) | [Contributors](https://github.com/05yuki/dsv41-flash-dual-epyc/graphs/contributors) |
| [CurtisAccelerate/DeepSeekv41-Flash-1017-EXL3-CMP170HX](https://github.com/CurtisAccelerate/DeepSeekv41-Flash-1017-EXL3-CMP170HX) | [7dcab995fe](https://github.com/CurtisAccelerate/DeepSeekv41-Flash-1017-EXL3-CMP170HX/commit/7dcab995fe31ac27b5e9e5872ae6b10dbba1538d) | [Contributors](https://github.com/CurtisAccelerate/DeepSeekv41-Flash-1017-EXL3-CMP170HX/graphs/contributors) |
| [CurtisAccelerate/DeepSeekv41-Flash-EXL3-TC-MoE](https://github.com/CurtisAccelerate/DeepSeekv41-Flash-EXL3-TC-MoE) | [13f8fe031f](https://github.com/CurtisAccelerate/DeepSeekv41-Flash-EXL3-TC-MoE/commit/13f8fe031f0b8c51d80582bcf037573795a5d3e8), [88eb6e13de](https://github.com/CurtisAccelerate/DeepSeekv41-Flash-EXL3-TC-MoE/commit/88eb6e13de0ea5491579fddb195116a4176f4bc7) | [Contributors](https://github.com/CurtisAccelerate/DeepSeekv41-Flash-EXL3-TC-MoE/graphs/contributors) |
| [Elevatormusic/DeepSeek-V4.1-Flash-1017-EXL3-TensorFold-2x-DGX-Spark](https://github.com/Elevatormusic/DeepSeek-V4.1-Flash-1017-EXL3-TensorFold-2x-DGX-Spark) | [9c6ab02340](https://github.com/Elevatormusic/DeepSeek-V4.1-Flash-1017-EXL3-TensorFold-2x-DGX-Spark/commit/9c6ab02340b770c6c3e1a9a4d5e70a56c5829d32) | [Contributors](https://github.com/Elevatormusic/DeepSeek-V4.1-Flash-1017-EXL3-TensorFold-2x-DGX-Spark/graphs/contributors) |
| [GNmac-lab/deepseek-v41-flash-1017-cmp170hx](https://github.com/GNmac-lab/deepseek-v41-flash-1017-cmp170hx) | [2ad5741ab1](https://github.com/GNmac-lab/deepseek-v41-flash-1017-cmp170hx/commit/2ad5741ab1a278d64f33b842706a41717dfbf181), [45ced96172](https://github.com/GNmac-lab/deepseek-v41-flash-1017-cmp170hx/commit/45ced96172bc53f9979d15af3575cb846d7f189d) | [Contributors](https://github.com/GNmac-lab/deepseek-v41-flash-1017-cmp170hx/graphs/contributors) |
| [MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks) | [64b02dea86](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/commit/64b02dea8681fb1d8e3a659ea432a8e9e9ba3995) | [Contributors](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/graphs/contributors) |
| [MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks) | [6f7d1590ad](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks/commit/6f7d1590ad49a2b8995188e45d7b9db31e677452) | [Contributors](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks/graphs/contributors) |
| [MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks) | [a8cc40ddef](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/commit/a8cc40ddef876f033e0f1b4d8f178b0fee2a5351) | [Contributors](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/graphs/contributors) |
| [Morrowmake/dsv41-flash-cmp170hx-recipe](https://github.com/Morrowmake/dsv41-flash-cmp170hx-recipe) | [7b7de425d5](https://github.com/Morrowmake/dsv41-flash-cmp170hx-recipe/commit/7b7de425d5c69a78b256df13456603b7f4c1a760), [7f9023cd28](https://github.com/Morrowmake/dsv41-flash-cmp170hx-recipe/commit/7f9023cd28ffd33878f1ce6ab68b9acbf3103c0a), [d4d1f525d7](https://github.com/Morrowmake/dsv41-flash-cmp170hx-recipe/commit/d4d1f525d7323be13dd32ffd594c23c5f73923bf) | [Contributors](https://github.com/Morrowmake/dsv41-flash-cmp170hx-recipe/graphs/contributors) |
| [Morrowmake/glm53-flash-cmp170hx-recipe](https://github.com/Morrowmake/glm53-flash-cmp170hx-recipe) | [9d3b0be374](https://github.com/Morrowmake/glm53-flash-cmp170hx-recipe/commit/9d3b0be374802e2f4e66cbc7ec7b1f203a6add78) | [Contributors](https://github.com/Morrowmake/glm53-flash-cmp170hx-recipe/graphs/contributors) |
| [Morrowmake/qwen38-flash-next-cmp170hx-recipe](https://github.com/Morrowmake/qwen38-flash-next-cmp170hx-recipe) | [3506c23587](https://github.com/Morrowmake/qwen38-flash-next-cmp170hx-recipe/commit/3506c235874606f3b5b5abfe00b33b6dccc93b78) | [Contributors](https://github.com/Morrowmake/qwen38-flash-next-cmp170hx-recipe/graphs/contributors) |
| [Morrowmake/vllm-cmp170hx](https://github.com/Morrowmake/vllm-cmp170hx) | [0af5991c00](https://github.com/Morrowmake/vllm-cmp170hx/commit/0af5991c009e0c3ff4181f026a8e7dffe1370b45), [18305948ac](https://github.com/Morrowmake/vllm-cmp170hx/commit/18305948acba3f4c4d912cdd56f2202d2ebd94e8), [3c97075b3d](https://github.com/Morrowmake/vllm-cmp170hx/commit/3c97075b3d1ad0104e6263b1c59b505654371324), [476339d339](https://github.com/Morrowmake/vllm-cmp170hx/commit/476339d33957c3b171bba176dd8b89d825289ec1), [903285fbcc](https://github.com/Morrowmake/vllm-cmp170hx/commit/903285fbcc4e51332d562499bd90ec9a1513c813), [9ad682152a](https://github.com/Morrowmake/vllm-cmp170hx/commit/9ad682152aa9e39e5e40c05dfbf41e7987a064a8), [a0c57b5237](https://github.com/Morrowmake/vllm-cmp170hx/commit/a0c57b52376efd67c57735bc949d4bafa22ae29e), [ad7acffa40](https://github.com/Morrowmake/vllm-cmp170hx/commit/ad7acffa40f726397bf497daeb4842513be0bf1d), [d1934f682c](https://github.com/Morrowmake/vllm-cmp170hx/commit/d1934f682c1d407e1af45972982f1b30e34d5089) | [Contributors](https://github.com/Morrowmake/vllm-cmp170hx/graphs/contributors) |
| [PixelML/DeepSeek-V4-Flash-0731-CMP-170HX](https://github.com/PixelML/DeepSeek-V4-Flash-0731-CMP-170HX) | [5c5b5a4b45](https://github.com/PixelML/DeepSeek-V4-Flash-0731-CMP-170HX/commit/5c5b5a4b45e8def82ec027737df616c55f997963) | [Contributors](https://github.com/PixelML/DeepSeek-V4-Flash-0731-CMP-170HX/graphs/contributors) |
| [PixelML/DeepSeek-V4-Flash-Vision-Exp-CMP-170HX](https://github.com/PixelML/DeepSeek-V4-Flash-Vision-Exp-CMP-170HX) | [7674c0110f](https://github.com/PixelML/DeepSeek-V4-Flash-Vision-Exp-CMP-170HX/commit/7674c0110fb469ac3857937c61eff6119bc60c49) | [Contributors](https://github.com/PixelML/DeepSeek-V4-Flash-Vision-Exp-CMP-170HX/graphs/contributors) |
| [PixelML/GLM-5.3-Flash-CMP-170HX](https://github.com/PixelML/GLM-5.3-Flash-CMP-170HX) | [7c5e32e50f](https://github.com/PixelML/GLM-5.3-Flash-CMP-170HX/commit/7c5e32e50fda2cf66b816f174c85d314665fc08f) | [Contributors](https://github.com/PixelML/GLM-5.3-Flash-CMP-170HX/graphs/contributors) |
| [PixelML/Qwen3.8-27B-CMP-170HX](https://github.com/PixelML/Qwen3.8-27B-CMP-170HX) | [41d2c414fe](https://github.com/PixelML/Qwen3.8-27B-CMP-170HX/commit/41d2c414fe0f293d77087ef18cda5896664754d6) | [Contributors](https://github.com/PixelML/Qwen3.8-27B-CMP-170HX/graphs/contributors) |
| [PixelML/Qwen3.8-Flash-Next-CMP-170HX](https://github.com/PixelML/Qwen3.8-Flash-Next-CMP-170HX) | [3255dd0e14](https://github.com/PixelML/Qwen3.8-Flash-Next-CMP-170HX/commit/3255dd0e148bff826fa3c9081080911edfb223da) | [Contributors](https://github.com/PixelML/Qwen3.8-Flash-Next-CMP-170HX/graphs/contributors) |
| [PixelML/deepspec-qwen38-flash-next](https://github.com/PixelML/deepspec-qwen38-flash-next) | [54b35f57af](https://github.com/PixelML/deepspec-qwen38-flash-next/commit/54b35f57af721ffa8d55f1c0ae79f24952a82fd6) | [Contributors](https://github.com/PixelML/deepspec-qwen38-flash-next/graphs/contributors) |
| [PixelML/luce-megakernel](https://github.com/PixelML/luce-megakernel) | [bab084443f](https://github.com/PixelML/luce-megakernel/commit/bab084443f14e6ab3ba76f8e8e3f721ad2e56144) | [Contributors](https://github.com/PixelML/luce-megakernel/graphs/contributors) |
| [PixelML/sm80vllm](https://github.com/PixelML/sm80vllm) | [27a94d1ce4](https://github.com/PixelML/sm80vllm/commit/27a94d1ce4e3fc100c4732439ccec10f8246a804), [f8ea5bb163](https://github.com/PixelML/sm80vllm/commit/f8ea5bb163c161ef38b401d055cc5fd4a934091a) | [Contributors](https://github.com/PixelML/sm80vllm/graphs/contributors) |
| [andreasgru/ds41-flash-hybrid](https://github.com/andreasgru/ds41-flash-hybrid) | [e74d1eb639](https://github.com/andreasgru/ds41-flash-hybrid/commit/e74d1eb6391bc7b85fec94759a2273d5b8beb648) | [Contributors](https://github.com/andreasgru/ds41-flash-hybrid/graphs/contributors) |
| [bertholomus/TensorFold](https://github.com/bertholomus/TensorFold) | [17c73e189f](https://github.com/bertholomus/TensorFold/commit/17c73e189f5e6a5304cda7ea37f086f9c49b4788) | [Contributors](https://github.com/bertholomus/TensorFold/graphs/contributors) |
| [bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10) | [abe07c7d50](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10/commit/abe07c7d509a0199cc7fec15263be2aa52ac652f) | [Contributors](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10/graphs/contributors) |
| [bertholomus/deepseek-v4.1-tensorfold-tp4-4xgb10](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp4-4xgb10) | [6033ea48a2](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp4-4xgb10/commit/6033ea48a27004b2715db82190e2a61f33411d92) | [Contributors](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp4-4xgb10/graphs/contributors) |
| [cibernox/glm53-flash-cmp170hx-recipe](https://github.com/cibernox/glm53-flash-cmp170hx-recipe) | [cb3311eb98](https://github.com/cibernox/glm53-flash-cmp170hx-recipe/commit/cb3311eb980ac67e890c185f98dddda57e2abd48) | [Contributors](https://github.com/cibernox/glm53-flash-cmp170hx-recipe/graphs/contributors) |
| [cibernox/vllm-cmp170hx](https://github.com/cibernox/vllm-cmp170hx) | [ab60b723ad](https://github.com/cibernox/vllm-cmp170hx/commit/ab60b723ada254a442a4ba5ff27bf837aa27ef83) | [Contributors](https://github.com/cibernox/vllm-cmp170hx/graphs/contributors) |
| [dfmcintosh-hash/cmp170hx-glm-mtp-notes](https://github.com/dfmcintosh-hash/cmp170hx-glm-mtp-notes) | [6e46e11320](https://github.com/dfmcintosh-hash/cmp170hx-glm-mtp-notes/commit/6e46e1132032faaeea9654e6adfedc66475bf628) | [Contributors](https://github.com/dfmcintosh-hash/cmp170hx-glm-mtp-notes/graphs/contributors) |
| [jayleaton/deepseek-v41-tensorfold-spark](https://github.com/jayleaton/deepseek-v41-tensorfold-spark) | [2dcce03034](https://github.com/jayleaton/deepseek-v41-tensorfold-spark/commit/2dcce0303451da2a59f80e6c7413da8b3b0f7a57) | [Contributors](https://github.com/jayleaton/deepseek-v41-tensorfold-spark/graphs/contributors) |
| [k0zakinio/qwen36-vllm-setup](https://github.com/k0zakinio/qwen36-vllm-setup) | [575c3efdd2](https://github.com/k0zakinio/qwen36-vllm-setup/commit/575c3efdd21fd19ea3a26bdfa60269350d655adc) | [Contributors](https://github.com/k0zakinio/qwen36-vllm-setup/graphs/contributors) |
| [kaka86mm/dsv41-flash-pp4-170hx](https://github.com/kaka86mm/dsv41-flash-pp4-170hx) | [950f4c003c](https://github.com/kaka86mm/dsv41-flash-pp4-170hx/commit/950f4c003c074be9434735a37ec67466559909ec) | [Contributors](https://github.com/kaka86mm/dsv41-flash-pp4-170hx/graphs/contributors) |
| [knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4) | [27616beca5](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4/commit/27616beca58690ea3120ad687e721c1c7e735412) | [Contributors](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4/graphs/contributors) |
| [liumorrisclaw/CMP170HX-TensorFold](https://github.com/liumorrisclaw/CMP170HX-TensorFold) | [4371e03d0f](https://github.com/liumorrisclaw/CMP170HX-TensorFold/commit/4371e03d0fb3f541264edd570c8f2def850e15bd) | [Contributors](https://github.com/liumorrisclaw/CMP170HX-TensorFold/graphs/contributors) |
| [liumorrisclaw/cmp170hx-glm53-lab](https://github.com/liumorrisclaw/cmp170hx-glm53-lab) | [e0de545cc8](https://github.com/liumorrisclaw/cmp170hx-glm53-lab/commit/e0de545cc8b168fa16a01fbeb5548b08efff53d1) | [Contributors](https://github.com/liumorrisclaw/cmp170hx-glm53-lab/graphs/contributors) |
| [liumorrisclaw/dsv41-flash-cmp170hx](https://github.com/liumorrisclaw/dsv41-flash-cmp170hx) | [2636c72f2c](https://github.com/liumorrisclaw/dsv41-flash-cmp170hx/commit/2636c72f2cf5a1e4363c4796ae009a9296e1d03e) | [Contributors](https://github.com/liumorrisclaw/dsv41-flash-cmp170hx/graphs/contributors) |
| [sfxnz/DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark](https://github.com/sfxnz/DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark) | [3de914682d](https://github.com/sfxnz/DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark/commit/3de914682d0880c8e478712a60ada930339c22e4) | [Contributors](https://github.com/sfxnz/DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark/graphs/contributors) |
| [tpurtell/ds41rt](https://github.com/tpurtell/ds41rt) | [3067d0684a](https://github.com/tpurtell/ds41rt/commit/3067d0684a0b572d88a17d38ca3f7d10599619b4) | [Contributors](https://github.com/tpurtell/ds41rt/graphs/contributors) |
| [vcruz305/DSV4.1-Flash-1017-EXL3-Spark-recipe](https://github.com/vcruz305/DSV4.1-Flash-1017-EXL3-Spark-recipe) | [63220db0a7](https://github.com/vcruz305/DSV4.1-Flash-1017-EXL3-Spark-recipe/commit/63220db0a7cc62dd8b783f71c9cf28f909c9b09b) | [Contributors](https://github.com/vcruz305/DSV4.1-Flash-1017-EXL3-Spark-recipe/graphs/contributors) |
| [vcruz305/DeepSeek-V4.1-Flash-EXL3-DGX-Spark-recipe](https://github.com/vcruz305/DeepSeek-V4.1-Flash-EXL3-DGX-Spark-recipe) | [e120f52e59](https://github.com/vcruz305/DeepSeek-V4.1-Flash-EXL3-DGX-Spark-recipe/commit/e120f52e599b4244df87f6c3e49ef12f3ecb67ab) | [Contributors](https://github.com/vcruz305/DeepSeek-V4.1-Flash-EXL3-DGX-Spark-recipe/graphs/contributors) |
| [vcruz305/DeepSeek-V4.1-Flash-GGUF-DGX-Spark-recipe](https://github.com/vcruz305/DeepSeek-V4.1-Flash-GGUF-DGX-Spark-recipe) | [c3c5484897](https://github.com/vcruz305/DeepSeek-V4.1-Flash-GGUF-DGX-Spark-recipe/commit/c3c548489771b964bc8b4c507509441276fd2bf9) | [Contributors](https://github.com/vcruz305/DeepSeek-V4.1-Flash-GGUF-DGX-Spark-recipe/graphs/contributors) |
| [vcruz305/exllamav3](https://github.com/vcruz305/exllamav3) | [e43b074096](https://github.com/vcruz305/exllamav3/commit/e43b074096a206a651f18107ee5eb025eb40beb2) | [Contributors](https://github.com/vcruz305/exllamav3/graphs/contributors) |
| [vcruz305/llama.cpp](https://github.com/vcruz305/llama.cpp) | [5210c7c5ed](https://github.com/vcruz305/llama.cpp/commit/5210c7c5ed61dddaee6ed476623abf4b63093d16) | [Contributors](https://github.com/vcruz305/llama.cpp/graphs/contributors) |
| [vcruz305/model-serving-minefield](https://github.com/vcruz305/model-serving-minefield) | [aa94d3dbf0](https://github.com/vcruz305/model-serving-minefield/commit/aa94d3dbf09dc5e80976305a7ead3e386d6b081c) | [Contributors](https://github.com/vcruz305/model-serving-minefield/graphs/contributors) |
| [vcruz305/sixcat-eval](https://github.com/vcruz305/sixcat-eval) | [e74cc8a33e](https://github.com/vcruz305/sixcat-eval/commit/e74cc8a33e5444c945142aca8ad8dc3b69d58f55) | [Contributors](https://github.com/vcruz305/sixcat-eval/graphs/contributors) |
| [vcruz305/tokenspeed](https://github.com/vcruz305/tokenspeed) | [1546438ac8](https://github.com/vcruz305/tokenspeed/commit/1546438ac851391aeb187a4a9dffd12bfb95a3b5) | [Contributors](https://github.com/vcruz305/tokenspeed/graphs/contributors) |
| [vcruz305/vllm-exl3](https://github.com/vcruz305/vllm-exl3) | [053c634db5](https://github.com/vcruz305/vllm-exl3/commit/053c634db54afc981becfb46c0482e7358cc00ba), [223e246ff8](https://github.com/vcruz305/vllm-exl3/commit/223e246ff8da552d2ace976c4bcbed804a8ec04c), [2a740a3773](https://github.com/vcruz305/vllm-exl3/commit/2a740a3773601c994072295fde45738ef8cea55e) | [Contributors](https://github.com/vcruz305/vllm-exl3/graphs/contributors) |

### Inference frameworks, kernels and quantization

| Project or release | Recorded source revisions | Contributors |
| --- | --- | --- |
| [1CatAI/1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM) | [6a143197e6](https://github.com/1CatAI/1Cat-vLLM/commit/6a143197e6e5f6fa77bd483df0c5a09dcea6aa20) | [Contributors](https://github.com/1CatAI/1Cat-vLLM/graphs/contributors) |
| [Cerynitius/freetoken-ox-boost](https://github.com/Cerynitius/freetoken-ox-boost) | [fdb99c6daa](https://github.com/Cerynitius/freetoken-ox-boost/commit/fdb99c6daae162094c8e2a80a5c177136c5e52f6) | [Contributors](https://github.com/Cerynitius/freetoken-ox-boost/graphs/contributors) |
| [FlashML-org/FreeToken](https://github.com/FlashML-org/FreeToken) | [408e6ae8c7](https://github.com/FlashML-org/FreeToken/commit/408e6ae8c7efe4dfabb77544ea98a311b3bfc139) | [Contributors](https://github.com/FlashML-org/FreeToken/graphs/contributors) |
| [HalfVulpes/Deepinfer-SM80](https://github.com/HalfVulpes/Deepinfer-SM80) | [2a828fc212](https://github.com/HalfVulpes/Deepinfer-SM80/commit/2a828fc2122b6fc59805c0c98b6aaf78e7da9b48), [8895510cae](https://github.com/HalfVulpes/Deepinfer-SM80/commit/8895510cae8f8f1733401686e67a4d1e11e7e94b) | [Contributors](https://github.com/HalfVulpes/Deepinfer-SM80/graphs/contributors) |
| [IST-DASLab/GSQ](https://github.com/IST-DASLab/GSQ) | [03fc16484c](https://github.com/IST-DASLab/GSQ/commit/03fc16484c369e3127225615d5e03e8d3a6043e3) | [Contributors](https://github.com/IST-DASLab/GSQ/graphs/contributors) |
| [IST-DASLab/RCO](https://github.com/IST-DASLab/RCO) | [9a1e09c07d](https://github.com/IST-DASLab/RCO/commit/9a1e09c07d468109cbe60a1b87d5036034a79d10) | [Contributors](https://github.com/IST-DASLab/RCO/graphs/contributors) |
| [IST-DASLab/marlin](https://github.com/IST-DASLab/marlin) | [1f25790bdd](https://github.com/IST-DASLab/marlin/commit/1f25790bdd49fba53106164a24666dade68d7c90) | [Contributors](https://github.com/IST-DASLab/marlin/graphs/contributors) |
| [LimeChain/vllm](https://github.com/LimeChain/vllm) | [4fd0bd1bf4](https://github.com/LimeChain/vllm/commit/4fd0bd1bf45733a31355d39bf43209c9d6df2e7f) | [Contributors](https://github.com/LimeChain/vllm/graphs/contributors) |
| [NVIDIA/flare-local-inference](https://github.com/NVIDIA/flare-local-inference) | [ca1f960edb](https://github.com/NVIDIA/flare-local-inference/commit/ca1f960edbf67d624f2d2948d2c111e8a6c8eb3c) | [Contributors](https://github.com/NVIDIA/flare-local-inference/graphs/contributors) |
| [NervanaSystems/maxas](https://github.com/NervanaSystems/maxas) | [54eda7af08](https://github.com/NervanaSystems/maxas/commit/54eda7af086a46c9dae1688b691968235d560164) | [Contributors](https://github.com/NervanaSystems/maxas/graphs/contributors) |
| [Niko1221/Strata](https://github.com/Niko1221/Strata) | [4b9e05a2a8](https://github.com/Niko1221/Strata/commit/4b9e05a2a806a3fe24f7b2560d98626c37dade35), [7ddc17a38f](https://github.com/Niko1221/Strata/commit/7ddc17a38fe7f31637e895ecfba58c2cd43d65ec) | [Contributors](https://github.com/Niko1221/Strata/graphs/contributors) |
| [akol1/TensorFold](https://github.com/akol1/TensorFold) | [9cd52ab4da](https://github.com/akol1/TensorFold/commit/9cd52ab4daba68ddd09be89be8f23ad43175e821) | [Contributors](https://github.com/akol1/TensorFold/graphs/contributors) |
| [antirez/ds4](https://github.com/antirez/ds4) | [ee7e840523](https://github.com/antirez/ds4/commit/ee7e8405232c8971000cc7ba2b945b00bf6d20d3) | [Contributors](https://github.com/antirez/ds4/graphs/contributors) |
| [ashhart/TensorFold](https://github.com/ashhart/TensorFold) | [8c36359d53](https://github.com/ashhart/TensorFold/commit/8c36359d533debdbb8f53a4229b0bb9362c9c783) | [Contributors](https://github.com/ashhart/TensorFold/graphs/contributors) |
| [clee/vllm-sm80](https://github.com/clee/vllm-sm80) | [89439db727](https://github.com/clee/vllm-sm80/commit/89439db7276cf165377c088496e20098c8137727) | [Contributors](https://github.com/clee/vllm-sm80/graphs/contributors) |
| [deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP) | [93eb6eb238](https://github.com/deepseek-ai/DeepEP/commit/93eb6eb238127e96c6d7a4a625a6dad158348509) | [Contributors](https://github.com/deepseek-ai/DeepEP/graphs/contributors) |
| [deepseek-ai/DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) | [1fc9c66975](https://github.com/deepseek-ai/DeepGEMM/commit/1fc9c66975ee322d628b3288534ca0e9ce12a64d), [5081f8513a](https://github.com/deepseek-ai/DeepGEMM/commit/5081f8513af57e61696de4f4f04b0e707ffb503e) | [Contributors](https://github.com/deepseek-ai/DeepGEMM/graphs/contributors) |
| [deepseek-ai/DeepInfer](https://github.com/deepseek-ai/DeepInfer) | [553741557a](https://github.com/deepseek-ai/DeepInfer/commit/553741557a57df44a46fb3ab72b6aafeaf945337), [9bd9cf243f](https://github.com/deepseek-ai/DeepInfer/commit/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9) | [Contributors](https://github.com/deepseek-ai/DeepInfer/graphs/contributors) |
| [deepseek-ai/DeepSelect](https://github.com/deepseek-ai/DeepSelect) | [bfa4507d93](https://github.com/deepseek-ai/DeepSelect/commit/bfa4507d935f17ebfc3d0f00ff7d3c9a4d0e5c18) | [Contributors](https://github.com/deepseek-ai/DeepSelect/graphs/contributors) |
| [deepseek-ai/FlashMLA](https://github.com/deepseek-ai/FlashMLA) | [0f11e178da](https://github.com/deepseek-ai/FlashMLA/commit/0f11e178da1f67becb087a324b400854931ec3e3), [2e5429fc56](https://github.com/deepseek-ai/FlashMLA/commit/2e5429fc5653bab6e081f09477126f731882a6a9) | [Contributors](https://github.com/deepseek-ai/FlashMLA/graphs/contributors) |
| [deepseek-ai/deepseek-recipe](https://github.com/deepseek-ai/deepseek-recipe) | [935817c66f](https://github.com/deepseek-ai/deepseek-recipe/commit/935817c66f5d6e64838f419692545f06787e485f) | [Contributors](https://github.com/deepseek-ai/deepseek-recipe/graphs/contributors) |
| [dphnAI/sonar](https://github.com/dphnAI/sonar) | [11c991b7d7](https://github.com/dphnAI/sonar/commit/11c991b7d7ad1b58bd9dfbfe8f803dd16a29db50) | [Contributors](https://github.com/dphnAI/sonar/graphs/contributors) |
| [eagle-hq/EAGLE-4](https://github.com/eagle-hq/EAGLE-4) | [76deb841a1](https://github.com/eagle-hq/EAGLE-4/commit/76deb841a13f0e3eb0c4b0f6708fcd6029f5c803) | [Contributors](https://github.com/eagle-hq/EAGLE-4/graphs/contributors) |
| [ecohash-co/TensorFold](https://github.com/ecohash-co/TensorFold) | [c7137bb31a](https://github.com/ecohash-co/TensorFold/commit/c7137bb31a7dad924fb541a7068422127b9a94b2) | [Contributors](https://github.com/ecohash-co/TensorFold/graphs/contributors) |
| [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention) | [329b534245](https://github.com/fla-org/flash-linear-attention/commit/329b534245d3d7bb27cc03a99e87bdc51fb28d06) | [Contributors](https://github.com/fla-org/flash-linear-attention/graphs/contributors) |
| [flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer) | [946200de1a](https://github.com/flashinfer-ai/flashinfer/commit/946200de1ae94fc93fdd0926f0a13afd1fa7f0f1), [bbee0f77a0](https://github.com/flashinfer-ai/flashinfer/commit/bbee0f77a0cfe28feb3858078eb2adf72fee935d) | [Contributors](https://github.com/flashinfer-ai/flashinfer/graphs/contributors) |
| [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | [7aca0d2ea8](https://github.com/ggml-org/llama.cpp/commit/7aca0d2ea8d92b56911233b822db01f9fbb1ae50) | [Contributors](https://github.com/ggml-org/llama.cpp/graphs/contributors) |
| [kacper-daftcode/vLLM-Moet](https://github.com/kacper-daftcode/vLLM-Moet) | [69aa6124ee](https://github.com/kacper-daftcode/vLLM-Moet/commit/69aa6124ee7c1fef39188d3232521adb963e09f8), [b0dc737ae4](https://github.com/kacper-daftcode/vLLM-Moet/commit/b0dc737ae431d9a2ab3f62ed3fc30313b00d99c1) | [Contributors](https://github.com/kacper-daftcode/vLLM-Moet/graphs/contributors) |
| [kindlingai/exlr8](https://github.com/kindlingai/exlr8) | [50f5cd1b65](https://github.com/kindlingai/exlr8/commit/50f5cd1b65ee26ab943d5884b65039c8204334cc), [6870c3d91e](https://github.com/kindlingai/exlr8/commit/6870c3d91ee1f1f3d74e85af424df8f3ae9d4733) | [Contributors](https://github.com/kindlingai/exlr8/graphs/contributors) |
| [kindlingai/glm-5.3-flash-exlr8-gx10](https://github.com/kindlingai/glm-5.3-flash-exlr8-gx10) | [ea2c28162e](https://github.com/kindlingai/glm-5.3-flash-exlr8-gx10/commit/ea2c28162e0b0387e885adfc9219239cf70b66e2) | [Contributors](https://github.com/kindlingai/glm-5.3-flash-exlr8-gx10/graphs/contributors) |
| [kindlingai/glm-5.3-flash-gx10](https://github.com/kindlingai/glm-5.3-flash-gx10) | [0d1f4620cc](https://github.com/kindlingai/glm-5.3-flash-gx10/commit/0d1f4620cc9a307fb496caedc70613246791f774) | [Contributors](https://github.com/kindlingai/glm-5.3-flash-gx10/graphs/contributors) |
| [kindlingai/gr8](https://github.com/kindlingai/gr8) | [a5f9aac7a2](https://github.com/kindlingai/gr8/commit/a5f9aac7a2982cbae25c284dd117c2379d835c8c) | [Contributors](https://github.com/kindlingai/gr8/graphs/contributors) |
| [kvcache-ai/ktransformers](https://github.com/kvcache-ai/ktransformers) | [a694e718e8](https://github.com/kvcache-ai/ktransformers/commit/a694e718e8b3ec2e221277d674e369f3be38d2a4) | [Contributors](https://github.com/kvcache-ai/ktransformers/graphs/contributors) |
| [local-inference-lab/b12x](https://github.com/local-inference-lab/b12x) | [2cc7f66aae](https://github.com/local-inference-lab/b12x/commit/2cc7f66aaef76a9c7cbb660a1efb865cf3f26c88) | [Contributors](https://github.com/local-inference-lab/b12x/graphs/contributors) |
| [local-inference-lab/vllm](https://github.com/local-inference-lab/vllm) | [47ccf6c57d](https://github.com/local-inference-lab/vllm/commit/47ccf6c57d92f03630ebcbad3809450545825488) | [Contributors](https://github.com/local-inference-lab/vllm/graphs/contributors) |
| [mw00/project-maya](https://github.com/mw00/project-maya) | [f26d8bab43](https://github.com/mw00/project-maya/commit/f26d8bab43f9b10f0475986b541dc46ada45285e) | [Contributors](https://github.com/mw00/project-maya/graphs/contributors) |
| [radixark/miles](https://github.com/radixark/miles) | [64aba5fc82](https://github.com/radixark/miles/commit/64aba5fc822f2e7761ec130221569f7828bdf96f), [c0fae0084c](https://github.com/radixark/miles/commit/c0fae0084c30955a84f0c4f5d7a6baf823ce9f35) | [Contributors](https://github.com/radixark/miles/graphs/contributors) |
| [sgl-project/sgl-amd-mono](https://github.com/sgl-project/sgl-amd-mono) | [6777a89942](https://github.com/sgl-project/sgl-amd-mono/commit/6777a89942a2a68d236328327815ba97b0ca8a9c) | [Contributors](https://github.com/sgl-project/sgl-amd-mono/graphs/contributors) |
| [sgl-project/sgl-nvidia-mega](https://github.com/sgl-project/sgl-nvidia-mega) | [b61745efcd](https://github.com/sgl-project/sgl-nvidia-mega/commit/b61745efcd99cbe9f5d350a2e07a926dfff0a33b) | [Contributors](https://github.com/sgl-project/sgl-nvidia-mega/graphs/contributors) |
| [sgl-project/sglang](https://github.com/sgl-project/sglang) | [e9c18259c7](https://github.com/sgl-project/sglang/commit/e9c18259c77dad912a13b0df3148a472d19630f4) | [Contributors](https://github.com/sgl-project/sglang/graphs/contributors) |
| [sybil-solutions/exl3xpu](https://github.com/sybil-solutions/exl3xpu) | [15ded2f3ad](https://github.com/sybil-solutions/exl3xpu/commit/15ded2f3add148c4db3c900cba7de878238f53bc) | [Contributors](https://github.com/sybil-solutions/exl3xpu/graphs/contributors) |
| [sybil-solutions/qwen-serve](https://github.com/sybil-solutions/qwen-serve) | [730d8101d0](https://github.com/sybil-solutions/qwen-serve/commit/730d8101d0ff7d53d7d6a3b980c3bae0c3b84181) | [Contributors](https://github.com/sybil-solutions/qwen-serve/graphs/contributors) |
| [syv-ai/HyperQwen](https://github.com/syv-ai/HyperQwen) | [ce9615c4f2](https://github.com/syv-ai/HyperQwen/commit/ce9615c4f2309be727407918a6c979bd67968069) | [Contributors](https://github.com/syv-ai/HyperQwen/graphs/contributors) |
| [tc-mb/llama.cpp-omni](https://github.com/tc-mb/llama.cpp-omni) | [a64932ec61](https://github.com/tc-mb/llama.cpp-omni/commit/a64932ec615e1463158129feaf2f0752d0c07a2e) | [Contributors](https://github.com/tc-mb/llama.cpp-omni/graphs/contributors) |
| [tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark) | [9fa230d3eb](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark/commit/9fa230d3eb119f588e690bf9b87ce41ed902db60) | [Contributors](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark/graphs/contributors) |
| [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) | [38a1252272](https://github.com/turboderp-org/exllamav3/commit/38a12522725ba707b90a674c03111cdeafd6976d), [d3739fd393](https://github.com/turboderp-org/exllamav3/commit/d3739fd393337b1ff4d6c2a342b12f0c87a9592f), [da68d57de6](https://github.com/turboderp-org/exllamav3/commit/da68d57de61781e5b74ad68698ac4caf36408d7a), [f157fc0b87](https://github.com/turboderp-org/exllamav3/commit/f157fc0b87933d63e44361b54ec093575e5a4549) | [Contributors](https://github.com/turboderp-org/exllamav3/graphs/contributors) |
| [vllm-project/DeepGEMM](https://github.com/vllm-project/DeepGEMM) | [727f847082](https://github.com/vllm-project/DeepGEMM/commit/727f847082d2e7946f70dc02b94949db206310d3), [e1f418c2a4](https://github.com/vllm-project/DeepGEMM/commit/e1f418c2a4f20818221f6b0e578b4c2f634d4c3f) | [Contributors](https://github.com/vllm-project/DeepGEMM/graphs/contributors) |
| [vllm-project/DeepSelect](https://github.com/vllm-project/DeepSelect) | [d96d33afe1](https://github.com/vllm-project/DeepSelect/commit/d96d33afe1fab0d6066da49cdc91e64c2bee65ea) | [Contributors](https://github.com/vllm-project/DeepSelect/graphs/contributors) |
| [vllm-project/FlashKDA](https://github.com/vllm-project/FlashKDA) | [17a037d98d](https://github.com/vllm-project/FlashKDA/commit/17a037d98da546deb4591e967cf961a43c034d8b) | [Contributors](https://github.com/vllm-project/FlashKDA/graphs/contributors) |
| [vllm-project/FlashMLA](https://github.com/vllm-project/FlashMLA) | [0eee43b12f](https://github.com/vllm-project/FlashMLA/commit/0eee43b12f034b657133cf2afca6a72ebb6efccf) | [Contributors](https://github.com/vllm-project/FlashMLA/graphs/contributors) |
| [vllm-project/MSA](https://github.com/vllm-project/MSA) | [968061ab16](https://github.com/vllm-project/MSA/commit/968061ab16cb14991a5c7d4c0e8eabf531be0dcf) | [Contributors](https://github.com/vllm-project/MSA/graphs/contributors) |
| [vllm-project/flash-attention](https://github.com/vllm-project/flash-attention) | [b273a02621](https://github.com/vllm-project/flash-attention/commit/b273a02621d3c94bc747df67204f591b67f38fae) | [Contributors](https://github.com/vllm-project/flash-attention/graphs/contributors) |
| [vllm-project/tml-fa4](https://github.com/vllm-project/tml-fa4) | [75765e76a9](https://github.com/vllm-project/tml-fa4/commit/75765e76a9c2c012c1f6ecd64577eb646eb4d303) | [Contributors](https://github.com/vllm-project/tml-fa4/graphs/contributors) |
| [vllm-project/vllm](https://github.com/vllm-project/vllm) | [2b1f75c80f](https://github.com/vllm-project/vllm/commit/2b1f75c80f79496b313e28362f0856632dfbf5ea), [fe77d0906b](https://github.com/vllm-project/vllm/commit/fe77d0906be0c38c5e2c575cd61b545a2fa687e4) | [Contributors](https://github.com/vllm-project/vllm/graphs/contributors) |
| [wtdcode/vllm-backport](https://github.com/wtdcode/vllm-backport) | [29e66dad46](https://github.com/wtdcode/vllm-backport/commit/29e66dad461b11caae6039f510eaa7913f9dfb19) | [Contributors](https://github.com/wtdcode/vllm-backport/graphs/contributors) |
| [z-lab/dflash](https://github.com/z-lab/dflash) | [07ebd93db9](https://github.com/z-lab/dflash/commit/07ebd93db9f472af339b644bb70221ad8428328a) | [Contributors](https://github.com/z-lab/dflash/graphs/contributors) |
| [zankich/vllm](https://github.com/zankich/vllm) | [b9534f2e06](https://github.com/zankich/vllm/commit/b9534f2e06ad2dd520445a3af00d7bdf8b1890d2) | [Contributors](https://github.com/zankich/vllm/graphs/contributors) |
| [zebgop-ops/dsv41reap-pp](https://github.com/zebgop-ops/dsv41reap-pp) | [138e748811](https://github.com/zebgop-ops/dsv41reap-pp/commit/138e74881130a1510b8bcf22cc01456bd6e46219) | [Contributors](https://github.com/zebgop-ops/dsv41reap-pp/graphs/contributors) |

### GPU tools, architecture references and measurement

| Project or release | Recorded source revisions | Contributors |
| --- | --- | --- |
| [GitHub Gist d2978531ba3a1a183e70f62f3de2eb84](https://gist.github.com/d2978531ba3a1a183e70f62f3de2eb84) | [8eb8b88e6b](https://gist.github.com/d2978531ba3a1a183e70f62f3de2eb84/8eb8b88e6bcb76c2e3766aafa178d94882200be5) | Release authors and contributors |
| [GitHub Gist f6f29c3d2992179361c8c0200ed71957](https://gist.github.com/f6f29c3d2992179361c8c0200ed71957) | [db595cca25](https://gist.github.com/f6f29c3d2992179361c8c0200ed71957/db595cca25f20f5917ef9854edaf7106a954147a) | Release authors and contributors |
| [0xD0GF00D/DocumentSASS](https://github.com/0xD0GF00D/DocumentSASS) | [5d9acfaef0](https://github.com/0xD0GF00D/DocumentSASS/commit/5d9acfaef0900fb11b6d84535868d807ac0ac4ca) | [Contributors](https://github.com/0xD0GF00D/DocumentSASS/graphs/contributors) |
| [0xha0/gpu-bench](https://github.com/0xha0/gpu-bench) | [4f2813b6e8](https://github.com/0xha0/gpu-bench/commit/4f2813b6e87502724f7be15fcb177a7ac4a098a9) | [Contributors](https://github.com/0xha0/gpu-bench/graphs/contributors) |
| [NVIDIA/cuda-samples](https://github.com/NVIDIA/cuda-samples) | [5443602d89](https://github.com/NVIDIA/cuda-samples/commit/5443602d89ed99aede2e4b7bf329daddeadb320e) | [Contributors](https://github.com/NVIDIA/cuda-samples/graphs/contributors) |
| [NVlabs/NVBit](https://github.com/NVlabs/NVBit) | [cda43eb372](https://github.com/NVlabs/NVBit/commit/cda43eb3720c69c051385e70791e0f948028a02c) | [Contributors](https://github.com/NVlabs/NVBit/graphs/contributors) |
| [ProjectPhysX/OpenCL-Benchmark](https://github.com/ProjectPhysX/OpenCL-Benchmark) | [4d2f16b5f7](https://github.com/ProjectPhysX/OpenCL-Benchmark/commit/4d2f16b5f7cecd5a29b695353452a23998088854) | [Contributors](https://github.com/ProjectPhysX/OpenCL-Benchmark/graphs/contributors) |
| [SemiAnalysisAI/InferenceX](https://github.com/SemiAnalysisAI/InferenceX) | [4ea8f9a3b5](https://github.com/SemiAnalysisAI/InferenceX/commit/4ea8f9a3b51ad95e538e88d1b6639f435ed496c6) | [Contributors](https://github.com/SemiAnalysisAI/InferenceX/graphs/contributors) |
| [accel-sim/gpu-app-collection](https://github.com/accel-sim/gpu-app-collection) | [dad09cb048](https://github.com/accel-sim/gpu-app-collection/commit/dad09cb0487845edc7524ded814c6cde9f0ef6a1) | [Contributors](https://github.com/accel-sim/gpu-app-collection/graphs/contributors) |
| [cloudcores/CuAssembler](https://github.com/cloudcores/CuAssembler) | [96a9f72baf](https://github.com/cloudcores/CuAssembler/commit/96a9f72baf00f40b9b299653fcef8d3e2b4a3d49) | [Contributors](https://github.com/cloudcores/CuAssembler/graphs/contributors) |
| [daadaada/turingas](https://github.com/daadaada/turingas) | [f7c1a74ebe](https://github.com/daadaada/turingas/commit/f7c1a74ebe6bc2ac51920b0fee29e27767f81724) | [Contributors](https://github.com/daadaada/turingas/graphs/contributors) |
| [edmcman/nvdisasm-re](https://github.com/edmcman/nvdisasm-re) | [21d9c1ee28](https://github.com/edmcman/nvdisasm-re/commit/21d9c1ee28de55f048757cf6cb48f565e1fcd4ed) | [Contributors](https://github.com/edmcman/nvdisasm-re/graphs/contributors) |
| [envytools/envytools](https://github.com/envytools/envytools) | [f102b82381](https://github.com/envytools/envytools/commit/f102b82381f3f11cee113d16374c87091db039d9) | [Contributors](https://github.com/envytools/envytools/graphs/contributors) |
| [google/go-containerregistry](https://github.com/google/go-containerregistry) | [92aac6186b](https://github.com/google/go-containerregistry/commit/92aac6186bc483b20c96c9fae4d98872f2bbace7) | [Contributors](https://github.com/google/go-containerregistry/graphs/contributors) |
| [reoLantern/nvsass-disassembler](https://github.com/reoLantern/nvsass-disassembler) | [28f906486c](https://github.com/reoLantern/nvsass-disassembler/commit/28f906486ca4a23e263e85c2f2f385fe7cb65bf5) | [Contributors](https://github.com/reoLantern/nvsass-disassembler/graphs/contributors) |
| [yushinliu/cuda-ptx-microbenchmark](https://github.com/yushinliu/cuda-ptx-microbenchmark) | [b91e4a607f](https://github.com/yushinliu/cuda-ptx-microbenchmark/commit/b91e4a607fb9da9dd92623ac8d950074e9a41926) | [Contributors](https://github.com/yushinliu/cuda-ptx-microbenchmark/graphs/contributors) |

### Hardware enablement and developer applications

| Project or release | Recorded source revisions | Contributors |
| --- | --- | --- |
| [MiaAI-Lab/sparkDash](https://github.com/MiaAI-Lab/sparkDash) | [3e6d4ef6e0](https://github.com/MiaAI-Lab/sparkDash/commit/3e6d4ef6e01ce0cd8abf65d00f44d69e467fd20b) | [Contributors](https://github.com/MiaAI-Lab/sparkDash/graphs/contributors) |
| [Morrowmake/cmpunlocker](https://github.com/Morrowmake/cmpunlocker) | [1f2c498f57](https://github.com/Morrowmake/cmpunlocker/commit/1f2c498f577cfe846f866e3e8151ae448f64b456) | [Contributors](https://github.com/Morrowmake/cmpunlocker/graphs/contributors) |
| [Morrowmake/local-ai-registry](https://github.com/Morrowmake/local-ai-registry) | [124959d8d4](https://github.com/Morrowmake/local-ai-registry/commit/124959d8d41495dffc2050883877fbb5fe4afbfb) | [Contributors](https://github.com/Morrowmake/local-ai-registry/graphs/contributors) |
| [Morrowmake/local-studio](https://github.com/Morrowmake/local-studio) | [f1cc60c982](https://github.com/Morrowmake/local-studio/commit/f1cc60c982ae5276616934a663d77712ace1ce92) | [Contributors](https://github.com/Morrowmake/local-studio/graphs/contributors) |
| [Morrowmake/oh-my-pi](https://github.com/Morrowmake/oh-my-pi) | [84843d4241](https://github.com/Morrowmake/oh-my-pi/commit/84843d42410c610ced1253681d3ac0f3f6e622ea) | [Contributors](https://github.com/Morrowmake/oh-my-pi/graphs/contributors) |
| [NVIDIA/open-gpu-kernel-modules](https://github.com/NVIDIA/open-gpu-kernel-modules) | [ca2d03e6fa](https://github.com/NVIDIA/open-gpu-kernel-modules/commit/ca2d03e6fafd4d4b56b383497d931c25b283bfcf), [e4a5faa256](https://github.com/NVIDIA/open-gpu-kernel-modules/commit/e4a5faa2567f28c8eabe0ebb6422b6d0abcf37eb) | [Contributors](https://github.com/NVIDIA/open-gpu-kernel-modules/graphs/contributors) |
| [PixelML/club-170hx](https://github.com/PixelML/club-170hx) | [c375fe5201](https://github.com/PixelML/club-170hx/commit/c375fe520147635ec2f81f3535095e0aa6acf24f) | [Contributors](https://github.com/PixelML/club-170hx/graphs/contributors) |
| [PixelML/cmpunlocker](https://github.com/PixelML/cmpunlocker) | [6c442eeb64](https://github.com/PixelML/cmpunlocker/commit/6c442eeb6448b97c803e72b61da344a39e0a26ab) | [Contributors](https://github.com/PixelML/cmpunlocker/graphs/contributors) |
| [aiultrayang/cmp170hx-vbios-flash](https://github.com/aiultrayang/cmp170hx-vbios-flash) | [943dfc50ef](https://github.com/aiultrayang/cmp170hx-vbios-flash/commit/943dfc50efef2667beb5f4166c30bb2cbd7d200b) | [Contributors](https://github.com/aiultrayang/cmp170hx-vbios-flash/graphs/contributors) |
| [amoghmunikote/cmpunlocker](https://github.com/amoghmunikote/cmpunlocker) | [2d10f891a5](https://github.com/amoghmunikote/cmpunlocker/commit/2d10f891a58824f0591c3332c4eccf37ff36b086) | [Contributors](https://github.com/amoghmunikote/cmpunlocker/graphs/contributors) |
| [asm64-hooligan/cmpunlocker](https://github.com/asm64-hooligan/cmpunlocker) | [dea1e84871](https://github.com/asm64-hooligan/cmpunlocker/commit/dea1e8487110918a9135f41863ac33a1f3effacc) | [Contributors](https://github.com/asm64-hooligan/cmpunlocker/graphs/contributors) |
| [cachenetics/170tune](https://github.com/cachenetics/170tune) | [5eb4775063](https://github.com/cachenetics/170tune/commit/5eb4775063b6cc055607ee5affecd210b13f0156) | [Contributors](https://github.com/cachenetics/170tune/graphs/contributors) |
| [cibernox/170tune](https://github.com/cibernox/170tune) | [0f09d26712](https://github.com/cibernox/170tune/commit/0f09d26712c893a521d12c306ea2502e1d8ed62c) | [Contributors](https://github.com/cibernox/170tune/graphs/contributors) |
| [cibernox/cmpunlocker](https://github.com/cibernox/cmpunlocker) | [6c442eeb64](https://github.com/cibernox/cmpunlocker/commit/6c442eeb6448b97c803e72b61da344a39e0a26ab) | [Contributors](https://github.com/cibernox/cmpunlocker/graphs/contributors) |
| [lesj0610/cmpunlocker](https://github.com/lesj0610/cmpunlocker) | [0ac5526edb](https://github.com/lesj0610/cmpunlocker/commit/0ac5526edbe939a793dbdd04b519c6700062642f) | [Contributors](https://github.com/lesj0610/cmpunlocker/graphs/contributors) |
| [liumorrisclaw/cmp170hx-gen2-window-fix](https://github.com/liumorrisclaw/cmp170hx-gen2-window-fix) | [b7825f65d1](https://github.com/liumorrisclaw/cmp170hx-gen2-window-fix/commit/b7825f65d1d5f28e602c17a42f7f6f730e9e5529) | [Contributors](https://github.com/liumorrisclaw/cmp170hx-gen2-window-fix/graphs/contributors) |
| [rjohny55/cmp170hx-unlocker](https://github.com/rjohny55/cmp170hx-unlocker) | [d8ed6abe29](https://github.com/rjohny55/cmp170hx-unlocker/commit/d8ed6abe2905831c348cc46e2ed4c35992b58da8) | [Contributors](https://github.com/rjohny55/cmp170hx-unlocker/graphs/contributors) |

### Model and quantization releases

| Project or release | Recorded source revisions | Contributors |
| --- | --- | --- |
| [0xSero/deepseek-v4-flash-0731-spark](https://huggingface.co/0xSero/deepseek-v4-flash-0731-spark) | [ce5ff0f1ef](https://huggingface.co/0xSero/deepseek-v4-flash-0731-spark/tree/ce5ff0f1efb2e184aafc759d281bfae47d3a359c) | Release authors and contributors |
| [LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw) | [c178a01284](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw/tree/c178a01284fc5c0e859d8668cffaac81ca8f4042) | Release authors and contributors |
| [LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw) | [49cb8dfc61](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw/tree/49cb8dfc610068031239308645f88bb3bc2a0138) | Release authors and contributors |
| [Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw) | [64ba41b6c9](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw/tree/64ba41b6c916a587db06eae2e19b7845f7be6e6b) | Release authors and contributors |
| [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) | [7872f01b1d](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/tree/7872f01b1d1fe23eabc4c98b48bffcef5a386062) | Release authors and contributors |
| [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | [314113f8a7](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/314113f8a7dadb388215396ba5fbee2c610a04d6) | Release authors and contributors |
| [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) | [1bae0e1088](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170) | Release authors and contributors |
| [fraserprice/DeepSeek-4.1-Flash-1017-BRRR](https://huggingface.co/fraserprice/DeepSeek-4.1-Flash-1017-BRRR) | [0bc4dbd0a1](https://huggingface.co/fraserprice/DeepSeek-4.1-Flash-1017-BRRR/tree/0bc4dbd0a15458e0957b899d72c7a5089e3335a5) | Release authors and contributors |
| [jayleaton/DeepSeek-V4.1-Flash-EXL3-2.8bpw](https://huggingface.co/jayleaton/DeepSeek-V4.1-Flash-EXL3-2.8bpw) | [915e1d0de8](https://huggingface.co/jayleaton/DeepSeek-V4.1-Flash-EXL3-2.8bpw/tree/915e1d0de8ee0a99944c5bc88ab802a639c63380) | Release authors and contributors |
| [kindlingai/glm-5.3-exlr8-k2-k3-k4](https://huggingface.co/kindlingai/glm-5.3-exlr8-k2-k3-k4) | [a36ae46808](https://huggingface.co/kindlingai/glm-5.3-exlr8-k2-k3-k4/tree/a36ae4680871ffa6412061d5daf66c4d4ad5802c) | Release authors and contributors |
| [kindlingai/glm-5.3-flash-exlr8-k2-k3-k4](https://huggingface.co/kindlingai/glm-5.3-flash-exlr8-k2-k3-k4) | [867580abd2](https://huggingface.co/kindlingai/glm-5.3-flash-exlr8-k2-k3-k4/tree/867580abd28d142bb4c56218bb81a752b7dea988) | Release authors and contributors |
| [pfeifferj/DeepSeek-V4.1-Flash-GSQ-RCO-GGUF](https://huggingface.co/pfeifferj/DeepSeek-V4.1-Flash-GSQ-RCO-GGUF) | [8270aceb26](https://huggingface.co/pfeifferj/DeepSeek-V4.1-Flash-GSQ-RCO-GGUF/tree/8270aceb26eb5590ea40f347256a1736999f551d) | Release authors and contributors |

### Earlier Anemone projects

| Project or release | Recorded source revisions | Contributors |
| --- | --- | --- |
| [anoane/08-deepseek-v41-flash-exl3-upstream-1m](https://github.com/anoane/08-deepseek-v41-flash-exl3-upstream-1m) | [a0915ca5ca](https://github.com/anoane/08-deepseek-v41-flash-exl3-upstream-1m/commit/a0915ca5cae6a07863ebbaea63dd3e6a6e83519b) | [Contributors](https://github.com/anoane/08-deepseek-v41-flash-exl3-upstream-1m/graphs/contributors) |
| [anoane/09-deepseek-v41-flash-1017-exl3-tp5-1m](https://github.com/anoane/09-deepseek-v41-flash-1017-exl3-tp5-1m) | [45f414a261](https://github.com/anoane/09-deepseek-v41-flash-1017-exl3-tp5-1m/commit/45f414a261b31643d045d128152d41d12381585f) | [Contributors](https://github.com/anoane/09-deepseek-v41-flash-1017-exl3-tp5-1m/graphs/contributors) |
| [anoane/exllamav3-anemone](https://github.com/anoane/exllamav3-anemone) | [309ef6d119](https://github.com/anoane/exllamav3-anemone/commit/309ef6d11920fb2b0df9b64bece4d70646fa23b3) | [Contributors](https://github.com/anoane/exllamav3-anemone/graphs/contributors) |
| [anoane/tabbyAPI](https://github.com/anoane/tabbyAPI) | [ce250ff83b](https://github.com/anoane/tabbyAPI/commit/ce250ff83b72deb51c026e97934645c6fbc8f141) | [Contributors](https://github.com/anoane/tabbyAPI/graphs/contributors) |

## Pull-request authors and reviewers

The following **179 populated pull-request reference records** credit **114 named authors**, alongside
their co-authors, reviewers and project maintainers. Titles and author handles are recorded from the
reference metadata; the original PR pages carry discussion, review and contribution history. Empty
reference placeholders supply no authorship information and are excluded.

### FlashML-org/FreeToken

| Pull request | Recorded author | Title |
| --- | --- | --- |
| [#24](https://github.com/FlashML-org/FreeToken/pull/24) | [@lukasrakauskas](https://github.com/lukasrakauskas) | fix(cuda): support Turing GPUs |
| [#48](https://github.com/FlashML-org/FreeToken/pull/48) | [@ousamabenyounes](https://github.com/ousamabenyounes) | fix: use current GPU for e4m3 capability |
| [#71](https://github.com/FlashML-org/FreeToken/pull/71) | [@calvarado2004](https://github.com/calvarado2004) | \[3/3\] perf(dsv4): add hardware-aware adaptive verification |
| [#89](https://github.com/FlashML-org/FreeToken/pull/89) | [@gdevenyi](https://github.com/gdevenyi) | perf(dsfp4): pick the grouped-prefill tile from route density on sm_89 |
| [#104](https://github.com/FlashML-org/FreeToken/pull/104) | [@RuixiangMa](https://github.com/RuixiangMa) | feat(models): support TP for qwen3_5_moe |
| [#222](https://github.com/FlashML-org/FreeToken/pull/222) | [@Artemowka22](https://github.com/Artemowka22) | fix(server): guarantee AbortMsg delivery on stream cancellation |
| [#253](https://github.com/FlashML-org/FreeToken/pull/253) | [@itongxiaojun](https://github.com/itongxiaojun) | feat(sm70): support Volta / Tesla V100 via torch 2.10 cu128 downgrade |
| [#257](https://github.com/FlashML-org/FreeToken/pull/257) | [@jason-fxz](https://github.com/jason-fxz) | feat(qwen4_exp): support Qwen3.8-Flash-Next |
| [#294](https://github.com/FlashML-org/FreeToken/pull/294) | [@JIAQI13](https://github.com/JIAQI13) | fix(qwen3_5_moe): detect shared-expert quant for modelopt mixed checkpoints |
| [#296](https://github.com/FlashML-org/FreeToken/pull/296) | [@JIAQI13](https://github.com/JIAQI13) | feat(qwen3_5_moe): support compressed-tensors NVFP4 experts with block-fp8 dense side |
| [#329](https://github.com/FlashML-org/FreeToken/pull/329) | [@taking-lying-flat](https://github.com/taking-lying-flat) | fix(kernel): make Triton top-k/top-p sampling exact |
| [#398](https://github.com/FlashML-org/FreeToken/pull/398) | [@AsmanovLev](https://github.com/AsmanovLev) | feat(kvcache): add ISO3/ISO4 KV-cache quantization (--kv-cache-iso) |
| [#468](https://github.com/FlashML-org/FreeToken/pull/468) | [@SanjayS45](https://github.com/SanjayS45) | fix(engine): reject fa attention backend below sm_90 at config time |
| [#475](https://github.com/FlashML-org/FreeToken/pull/475) | [@rafaeldrincon](https://github.com/rafaeldrincon) | fix(tokenizer): respect explicit enable_thinking=false even when tools present |
| [#476](https://github.com/FlashML-org/FreeToken/pull/476) | [@rafaeldrincon](https://github.com/rafaeldrincon) | feat(server): add --default-thinking-mode flag for server-wide thinking default |
| [#477](https://github.com/FlashML-org/FreeToken/pull/477) | [@rafaeldrincon](https://github.com/rafaeldrincon) | feat(server): expose POST /v1/tokenize endpoint for client-side token counting |
| [#507](https://github.com/FlashML-org/FreeToken/pull/507) | [@calvarado2004](https://github.com/calvarado2004) | feat(glm5_next): serve GLM-5.3-Flash with tensor parallelism |
| [#578](https://github.com/FlashML-org/FreeToken/pull/578) | [@YevheniiKotyrlo](https://github.com/YevheniiKotyrlo) | perf(moe): dequant nvfp4 prefill experts by bit placement and load activations contiguously |
| [#631](https://github.com/FlashML-org/FreeToken/pull/631) | [@jason-fxz](https://github.com/jason-fxz) | perf(kernel): avoid layout conversions in nvfp4 marlin decode |
| [#632](https://github.com/FlashML-org/FreeToken/pull/632) | [@jason-fxz](https://github.com/jason-fxz) | perf(kernel): skip masks and out-of-window tiles in triton extend attention |
| [#633](https://github.com/FlashML-org/FreeToken/pull/633) | [@jason-fxz](https://github.com/jason-fxz) | perf(kernel): keep 16-byte kv loads in dsv4 sparse attention and unroll gated pool |
| [#634](https://github.com/FlashML-org/FreeToken/pull/634) | [@jason-fxz](https://github.com/jason-fxz) | perf(kernel): read dsv41 indexer key rows as int32 words |

### Niko1221/Strata

| Pull request | Recorded author | Title |
| --- | --- | --- |
| [#81](https://github.com/Niko1221/Strata/pull/81) | [@mikicvi](https://github.com/mikicvi) | Accept sm_80+ GPUs at runtime, matching the build guard and the README |
| [#87](https://github.com/Niko1221/Strata/pull/87) | [@hireymage](https://github.com/hireymage) | Turing port: run on RTX 20 (sm_75) and newer |
| [#88](https://github.com/Niko1221/Strata/pull/88) | [@hireymage](https://github.com/hireymage) | Pascal port: lower the CUDA floor to sm_61 (GTX 10 series) |
| [#101](https://github.com/Niko1221/Strata/pull/101) | [@eightman999](https://github.com/eightman999) | build: allow sm_70 (Volta) builds — the only sm_80+ dependency is unreferenced |
| [#107](https://github.com/Niko1221/Strata/pull/107) | [@architectds](https://github.com/architectds) | serve: report cached prompt tokens, llama.cpp timings and GET /v1/status |
| [#108](https://github.com/Niko1221/Strata/pull/108) | [@q8atnight](https://github.com/q8atnight) | prefill: bit-identical kernel speed-ups (batched indexer append, parallel GDN conv, column-split GDN recurrence, one-launch embedding gather) |
| [#109](https://github.com/Niko1221/Strata/pull/109) | [@q8atnight](https://github.com/q8atnight) | decode: batch the verify window's per-token kernels (bit-identical, +~10% decode) |
| [#110](https://github.com/Niko1221/Strata/pull/110) | [@q8atnight](https://github.com/q8atnight) | Multi-GPU: a second GPU as an expert tier for decode and prompts (--peer-device), + multi-conversation cache |
| [#120](https://github.com/Niko1221/Strata/pull/120) | [@orangeswim](https://github.com/orangeswim) |  `--kv k8v4`: hybrid KV cache — INT8 K + Hadamard-rotated Q4_0 V (816 B/cell) |
| [#130](https://github.com/Niko1221/Strata/pull/130) | [@DingoOz](https://github.com/DingoOz) | setup: Volta (V100, sm_70) and Turing as an experimental build path |
| [#134](https://github.com/Niko1221/Strata/pull/134) | [@architectds](https://github.com/architectds) | setup: count the 3-bit models' 262K context against RAM, not a fixed 90 GB |
| [#136](https://github.com/Niko1221/Strata/pull/136) | [@architectds](https://github.com/architectds) | Linux prebuilt from CI: CUDA 12.8, sm_80/86/89 + PTX, attached to each release |
| [#139](https://github.com/Niko1221/Strata/pull/139) | [@rluisr](https://github.com/rluisr) | perf(prefill): add shape-selected pre-Ampere BF16 SGEMM with current-main A/B |
| [#176](https://github.com/Niko1221/Strata/pull/176) | [@Efeisot](https://github.com/Efeisot) | HIP backend on gfx1200 (RDNA4, RX 9060 XT): the gfx1100 backend runs unchanged with three deltas (Changes made by GLM5.3-Flash from freebuff) |
| [#186](https://github.com/Niko1221/Strata/pull/186) | [@q8atnight](https://github.com/q8atnight) | decode: hyper-connection read in 2 kernels, stream-split (-1.5 ms per verify window) |
| [#227](https://github.com/Niko1221/Strata/pull/227) | [@demetree](https://github.com/demetree) | AVX1: the Q2_0 expert kernels for CPUs that have AVX but no AVX2 |
| [#234](https://github.com/Niko1221/Strata/pull/234) | [@mad9home](https://github.com/mad9home) | Bench: RTX 3090 / dual RTX 3090 on EPYC 7453 |
| [#245](https://github.com/Niko1221/Strata/pull/245) | [@eddoursul](https://github.com/eddoursul) | Doubled throughput |
| [#258](https://github.com/Niko1221/Strata/pull/258) | [@hireymage](https://github.com/hireymage) | fused_gr: TILE=1280 kernel specialization for sm_75 (Turing) down-projection |
| [#270](https://github.com/Niko1221/Strata/pull/270) | [@kenh0u](https://github.com/kenh0u) | qsa_prompt_attn: run the f16 tensor-core path on Turing (sm_75) |
| [#289](https://github.com/Niko1221/Strata/pull/289) | [@sergqwer](https://github.com/sergqwer) | Start: name the GPU and stop at once when the build has no code for it; STRATA_EMULATE_CC |
| [#315](https://github.com/Niko1221/Strata/pull/315) | [@BlueKingMuch](https://github.com/BlueKingMuch) | hc: the multi-token hyper-connection read split finer |
| [#329](https://github.com/Niko1221/Strata/pull/329) | [@bsorensen110](https://github.com/bsorensen110) | hip: gfx12 (RDNA4) WMMA kernel for the int8-KV QSA prompt attention (1.26-1.35x prompt processing) |
| [#330](https://github.com/Niko1221/Strata/pull/330) | [@taweili](https://github.com/taweili) | V100 (sm_70): the compute-capability floor drops from 7.5 to 7.0 |
| [#395](https://github.com/Niko1221/Strata/pull/395) | [@rafal-prasal](https://github.com/rafal-prasal) | Feature/nvidia p40 |
| [#413](https://github.com/Niko1221/Strata/pull/413) | [@BlueKingMuch](https://github.com/BlueKingMuch) | prefill: DeltaNet recurrence with the three value heads of a key head in one thread (bitwise; 1.4x on a 4080 SUPER, 1.3x on a 3090) |
| [#439](https://github.com/Niko1221/Strata/pull/439) | [@architectds](https://github.com/architectds) | prefill: a group's expert gathers in one launch (bit-identical, long prompts +4%) |
| [#452](https://github.com/Niko1221/Strata/pull/452) | [@architectds](https://github.com/architectds) | qsa_prompt_attn: Q4_0 KV on tensor cores (mode 4) - long prompts with --kv q4_0 ~19% faster |
| [#453](https://github.com/Niko1221/Strata/pull/453) | [@architectds](https://github.com/architectds) | prefill: the draft layer's batched K/V for a ring too (KV streaming) - prefill +4.6% with --kv-resident |
| [#478](https://github.com/Niko1221/Strata/pull/478) | [@yannickloth](https://github.com/yannickloth) | prefill: MMQ prompt kernels for Q6_K experts (STRATA_MMQ_KQUANTS) |
| [#484](https://github.com/Niko1221/Strata/pull/484) | [@xidus90](https://github.com/xidus90) | Monitor: show the conversation cache at work (reuse, hits on switches, slots, cache RAM) |
| [#489](https://github.com/Niko1221/Strata/pull/489) | [@yannickloth](https://github.com/yannickloth) | bench/results: RTX A3000 12GB laptop + i7-12850HX (sm_86): hugepages, PCIe probe, calibration |
| [#563](https://github.com/Niko1221/Strata/pull/563) | [@Mirtraxxx](https://github.com/Mirtraxxx) | Give the expert cache's VRAM back without unloading the model (#533) |
| [#575](https://github.com/Niko1221/Strata/pull/575) | [@orangeswim](https://github.com/orangeswim) | qsa select: split the top-k past the register fit - 9% faster long-prompt prefill |
| [#600](https://github.com/Niko1221/Strata/pull/600) | [@fks](https://github.com/fks) | qsa_prompt_attn: tensor-core kernel for Volta (sm_70, mma.m8n8k4) |
| [#638](https://github.com/Niko1221/Strata/pull/638) | [@JeanP00l](https://github.com/JeanP00l) | hip: AMD Instinct MI50 / MI60 / Radeon VII (gfx906, wave64) as an opt-in build |
| [#646](https://github.com/Niko1221/Strata/pull/646) | [@stuchapin909](https://github.com/stuchapin909) | perf(cuda,decode): zero-doorbell resident verify graph, sub-warp expert packing &amp; shared-mem staging (+39-72% tok/s) |
| [#652](https://github.com/Niko1221/Strata/pull/652) | [@anon761](https://github.com/anon761) | serve: commit only the tokens a verify window hands out |
| [#674](https://github.com/Niko1221/Strata/pull/674) | [@pertain99](https://github.com/pertain99) | bench: community report, RTX 3090 Ti + 2x Xeon E5-2699 v3 (HP Z840), IQ3_S at 262K |
| [#741](https://github.com/Niko1221/Strata/pull/741) | [@konijiwa110](https://github.com/konijiwa110) | prefill: BF16 projections on FP16 tensor cores below sm_80 (+8-17% prefill on a 2080 Ti) |
| [#742](https://github.com/Niko1221/Strata/pull/742) | [@konijiwa110](https://github.com/konijiwa110) | qsa_select: FP32 tiled block scores below sm_80 (+22% 128K / +33% 248K prefill on a 2080 Ti) |
| [#789](https://github.com/Niko1221/Strata/pull/789) | [@InB4DevOps](https://github.com/InB4DevOps) | prefill: overlap shared-expert work with routed expert uploads |
| [#792](https://github.com/Niko1221/Strata/pull/792) | [@0xPreDa](https://github.com/0xPreDa) | batch slots: the zero-doorbell graph (100% VRAM resident) rings no layer |
| [#824](https://github.com/Niko1221/Strata/pull/824) | [@perronemirko](https://github.com/perronemirko) | Qwen3.6 35b a3b |
| [#835](https://github.com/Niko1221/Strata/pull/835) | [@Vop3p](https://github.com/Vop3p) | HIP gfx103x: the prompt path's 16-bit GEMMs in FP16 in and out (prompt reads ~2x on RX 6900 XT) |
| [#878](https://github.com/Niko1221/Strata/pull/878) | [@FearL0rd](https://github.com/FearL0rd) | # Support sm_60, sm_70, and sm_86 with CUDA 12.6 or 12.8 |
| [#910](https://github.com/Niko1221/Strata/pull/910) | [@Hardin22](https://github.com/Hardin22) | Two-GPU decode: bounded attention merge (bitwise, opt-in) and pipelined PLE read-ahead / early chain (stacked on #905) |
| [#930](https://github.com/Niko1221/Strata/pull/930) | [@sanastasiou](https://github.com/sanastasiou) | cpu: IQ3_S / IQ2_S AVX2 decode without the per-index shift/mask/or (1.07-1.6x per row) |
| [#950](https://github.com/Niko1221/Strata/pull/950) | [@DuncanBetts](https://github.com/DuncanBetts) | prefill: opt-in pointer-list bypass for gate/up gather (STRATA_PF_PTR_MMQ) |
| [#960](https://github.com/Niko1221/Strata/pull/960) | [@konijiwa110](https://github.com/konijiwa110) | serve: prefix snapshots on disk - a new chat restores its system prompt instead of reading it again |
| [#988](https://github.com/Niko1221/Strata/pull/988) | [@MirkoMorello](https://github.com/MirkoMorello) | Resident RAM mode with a layer split: only the experts no stage's cache holds (UD-Q4_K_XL on 2x 3090 with 64 GB) |
| [#999](https://github.com/Niko1221/Strata/pull/999) | [@xyzzing](https://github.com/xyzzing) | hip: the fused-int8 MoE kernels on gfx1100 rocWMMA - opt-in STRATA_PF_FUSED arm + parity fixture (engagement: parity, REJECT-at-parity recorded) |
| [#1004](https://github.com/Niko1221/Strata/pull/1004) | [@midagedev](https://github.com/midagedev) | serve: /props total_slots is the engine's batch slots |
| [#1007](https://github.com/Niko1221/Strata/pull/1007) | [@Vop3p](https://github.com/Vop3p) | hip: the QSA block scores on rocBLAS SGEMM, the solution measured on the card (opt-in; a 128K prompt +2.8% on gfx1030) |
| [#1090](https://github.com/Niko1221/Strata/pull/1090) | [@konijiwa110](https://github.com/konijiwa110) | serve: prefix snapshots on disk - a new chat restores its system prompt instead of reading it again |
| [#1091](https://github.com/Niko1221/Strata/pull/1091) | [@konijiwa110](https://github.com/konijiwa110) | qsa_select: FP32 tiled block scores below sm_80 (+22% 128K / +44% 248K prefill on a 2080 Ti) |
| [#1186](https://github.com/Niko1221/Strata/pull/1186) | [@txiki739](https://github.com/txiki739) | setup: --draft-vocab es - the English/code subset plus the tokens of Spanish text |
| [#1199](https://github.com/Niko1221/Strata/pull/1199) | [@lucapug](https://github.com/lucapug) | Community benchmark: RTX 3090 eGPU + 64GB, IQ2_XS vs IQ3_XXS |
| [#1220](https://github.com/Niko1221/Strata/pull/1220) | [@x00r](https://github.com/x00r) | fix #1121: preserve shared draft-head metadata in batch MTP slots |
| [#1223](https://github.com/Niko1221/Strata/pull/1223) | [@chimpera](https://github.com/chimpera) | generate: allow the elastic K/V with --peer-device (VMM ranges keep their memory on the owning GPU) |
| [#1246](https://github.com/Niko1221/Strata/pull/1246) | [@anon761](https://github.com/anon761) | build: vmm.hpp declares size_t via &lt;cstddef&gt; (CUDA 13 headers dropped the transitive include) |
| [#1247](https://github.com/Niko1221/Strata/pull/1247) | [@anon761](https://github.com/anon761) | prefill: --prefill-help and --prefill-help-frac as the official flags for the idle-stage helper |
| [#1255](https://github.com/Niko1221/Strata/pull/1255) | [@midagedev](https://github.com/midagedev) | serve: /props total_slots is the engine's batch slots |
| [#1260](https://github.com/Niko1221/Strata/pull/1260) | [@mcuygjsliy429](https://github.com/mcuygjsliy429) | tests: order MMQ parity transfers on the compute stream |
| [#1264](https://github.com/Niko1221/Strata/pull/1264) | [@merbanan](https://github.com/merbanan) | verify: the batched K/V append writes a streamed K8V4 state's host copy (#1188, #1169) |
| [#1269](https://github.com/Niko1221/Strata/pull/1269) | [@pspranger-throw](https://github.com/pspranger-throw) | \[Engine\] restore a session file without holding its K/V in RAM |
| [#1273](https://github.com/Niko1221/Strata/pull/1273) | [@pspranger-throw](https://github.com/pspranger-throw) | serve: session files work with --peer-device |
| [#1291](https://github.com/Niko1221/Strata/pull/1291) | [@NRK-SH](https://github.com/NRK-SH) | bench: community report - HP Z820, dual Xeon E5-2697 v2 (AVX only) + RTX 3090, Qwen3.8-Flash-Next IQ3_S, 512K context |
| [#1327](https://github.com/Niko1221/Strata/pull/1327) | [@constantindjonkam](https://github.com/constantindjonkam) | serve: opt-in finite video input |
| [#1367](https://github.com/Niko1221/Strata/pull/1367) | [@sergqwer](https://github.com/sergqwer) | prompt attention: STRATA_PROMPT_ATTN_IMMA=1 - the int8-KV kernel on INT8 tensor cores (opt-in, closer to FP32) |
| [#1368](https://github.com/Niko1221/Strata/pull/1368) | [@sergqwer](https://github.com/sergqwer) | prefill: quantize each token's MoE input once and scatter it to its k rows (the same bytes) |
| [#1370](https://github.com/Niko1221/Strata/pull/1370) | [@sergqwer](https://github.com/sergqwer) | prefill: opt-in STRATA_HC_UPMIX=1 on CUDA - the hyper-connection up projection with gr_mix_r as its epilogue (sm_80+) |
| [#1372](https://github.com/Niko1221/Strata/pull/1372) | [@sergqwer](https://github.com/sergqwer) | DeltaNet recurrence: STRATA_GDN_CHUNKED=1 - the prompt's recurrence in 32-token chunks (opt-in, 1.8x faster, other bits) |
| [#1401](https://github.com/Niko1221/Strata/pull/1401) | [@rewin123](https://github.com/rewin123) | V100 (sm_70): prompt experts on FP16 tensor cores (+10% prompt) and three existing decode kernels as defaults (-6.8% GPU per window) |
| [#1416](https://github.com/Niko1221/Strata/pull/1416) | [@architectds](https://github.com/architectds) | prefill: the CPU share's activations quantized on the GPU, copied down beside the routing sync (builds on #1414) |
| [#1418](https://github.com/Niko1221/Strata/pull/1418) | [@imanu86](https://github.com/imanu86) | cuda: opt-in MMVQ activation reuse across output rows |
| [#1424](https://github.com/Niko1221/Strata/pull/1424) | [@ruibeikaa](https://github.com/ruibeikaa) | Pascal GP100: a Q8_0 decode GEMV (STRATA_Q8_SM60=1, opt-in) and IQ3_XXS / IQ3_S tables in shared memory |
| [#1437](https://github.com/Niko1221/Strata/pull/1437) | [@agorevski](https://github.com/agorevski) | perf(cuda): add opt-in shape-tuned sm75 interleaved verify |
| [#1441](https://github.com/Niko1221/Strata/pull/1441) | [@ruibeikaa](https://github.com/ruibeikaa) | Prompt chunks sized for a layer split's pipeline (`--prefill-pipe`, opt-in; b/a from `--calibrate`) |
| [#1461](https://github.com/Niko1221/Strata/pull/1461) | [@q8atnight](https://github.com/q8atnight) | Peer tier with mutual help: concurrency 2 at 155 t/s aggregate on dual RTX 3090 (elastic peer pair, opt-in) |
| [#1504](https://github.com/Niko1221/Strata/pull/1504) | [@sanastasiou](https://github.com/sanastasiou) | serve --batch: a layer split reads a prompt in STRATA_SPLIT_PIECE-token runs, decided at every boundary |
| [#1544](https://github.com/Niko1221/Strata/pull/1544) | [@stuchapin909](https://github.com/stuchapin909) | cpu: IQ2_S in the bit-exact singleton rows, and the singleton path only below the #152 rule (stacked on #1415) |
| [#1546](https://github.com/Niko1221/Strata/pull/1546) | [@butonic](https://github.com/butonic) | Community benchmark: RTX 3090 24 GB + 128 GB RAM |
| [#1548](https://github.com/Niko1221/Strata/pull/1548) | [@stuchapin909](https://github.com/stuchapin909) | decode: STRATA_PCIE_BALANCE=1 picks each layer's PCIe count from measured costs (refile of #1038 by @sergqwer, on 0.1.40.3) |

### antirez/ds4

| Pull request | Recorded author | Title |
| --- | --- | --- |
| [#460](https://github.com/antirez/ds4/pull/460) | [@fmolara](https://github.com/fmolara) | CUDA: batch gate/up/down uploads for selected expert cache misses |
| [#737](https://github.com/antirez/ds4/pull/737) | [@exochard](https://github.com/exochard) | cuda: refuse unsafe host fallback during SSD streaming |
| [#738](https://github.com/antirez/ds4/pull/738) | [@exochard](https://github.com/exochard) | cuda: overlap streamed expert uploads with compute |
| [#739](https://github.com/antirez/ds4/pull/739) | [@exochard](https://github.com/exochard) | cuda: retain routed experts in bounded cache |
| [#740](https://github.com/antirez/ds4/pull/740) | [@exochard](https://github.com/exochard) | cuda: sanitize non-finite router scores |
| [#869](https://github.com/antirez/ds4/pull/869) | [@kmc6042](https://github.com/kmc6042) | cuda: fix Ampere token-tile ldmatrix IMA |
| [#872](https://github.com/antirez/ds4/pull/872) | [@tedin7](https://github.com/tedin7) | CUDA: reject prefill mixed shapes wider than the shared scores window (#803) |
| [#1079](https://github.com/antirez/ds4/pull/1079) | [@fulvian](https://github.com/fulvian) | cuda: give ldmatrix the .shared state space in token-tile attention |
| [#1097](https://github.com/antirez/ds4/pull/1097) | [@triplesparkle](https://github.com/triplesparkle) |  GLM 5.3 Flash on the DGX Spark: measured, and q4 on CUDA (#907) |
| [#1130](https://github.com/antirez/ds4/pull/1130) | [@aamargulies](https://github.com/aamargulies) | CUDA: give the ldmatrix helpers the .shared state space |
| [#1138](https://github.com/antirez/ds4/pull/1138) | [@flaccagora](https://github.com/flaccagora) | cuda: make stream top-512 compaction decision uniform across warps |
| [#1161](https://github.com/antirez/ds4/pull/1161) | [@Laurianti](https://github.com/Laurianti) | server: report Anthropic stop_sequence when a client stop matches |
| [#1162](https://github.com/antirez/ds4/pull/1162) | [@Laurianti](https://github.com/Laurianti) | server: report max_output_tokens as the Responses incomplete reason |

### ashhart/TensorFold

| Pull request | Recorded author | Title |
| --- | --- | --- |
| [#62](https://github.com/ashhart/TensorFold/pull/62) | [@nood-co1](https://github.com/nood-co1) | fix: say why a CUDA start is silent (SIGUSR1 stack dump, extension builds, stale build locks) |
| [#264](https://github.com/ashhart/TensorFold/pull/264) | [@RoscoeTT](https://github.com/RoscoeTT) | Qwen3.8-27B on Tesla V100 (sm_70): Volta kernels |
| [#292](https://github.com/ashhart/TensorFold/pull/292) | [@RoscoeTT](https://github.com/RoscoeTT) | Nemotron 3.5 Lightning on Tesla V100 (sm_70), on #264 |
| [#345](https://github.com/ashhart/TensorFold/pull/345) | [@gilby](https://github.com/gilby) | feat(qwen3.6 cuda): read ModelOpt NVFP4 checkpoints |
| [#355](https://github.com/ashhart/TensorFold/pull/355) | [@gilby](https://github.com/gilby) | feat(qwen3.8 cuda): NVFP4 checkpoints on two ranks (--tp 2) |
| [#356](https://github.com/ashhart/TensorFold/pull/356) | [@gilby](https://github.com/gilby) | perf(cuda): two-rank NVFP4 layers shard as they load; row-parallel sums over a peer mapping |
| [#357](https://github.com/ashhart/TensorFold/pull/357) | [@gilby](https://github.com/gilby) | perf(cuda): two-rank sums through shared page-locked host memory where the GPUs cannot map each other |
| [#358](https://github.com/ashhart/TensorFold/pull/358) | [@gilby](https://github.com/gilby) | perf(qwen3.8 cuda): DFlash2 tree verify, commit and draft step replay CUDA graphs, one stream and two ranks |
| [#359](https://github.com/ashhart/TensorFold/pull/359) | [@gilby](https://github.com/gilby) | perf(qwen3.6 cuda): --parallel rounds of several streams replay CUDA graphs (verify and MTP head) |
| [#363](https://github.com/ashhart/TensorFold/pull/363) | [@gilby](https://github.com/gilby) | feat(cuda): Ampere (sm_80/86) runs the bf16 kernels; FP8 prompts stay at 8.9 and NVFP4 runs W4A16 below it |
| [#364](https://github.com/ashhart/TensorFold/pull/364) | [@gilby](https://github.com/gilby) | perf(cuda): NVFP4 e4m3 scales decode by an exact bit trick below sm_89 |
| [#365](https://github.com/ashhart/TensorFold/pull/365) | [@gilby](https://github.com/gilby) | perf(cuda): lane matmul reads 5-, 6- and 8-bit affine words in groups of 32 |
| [#374](https://github.com/ashhart/TensorFold/pull/374) | [@gilby](https://github.com/gilby) | feat(volta): Qwen3.8-27B NVFP4 checkpoints on sm_70 (W4A16 / W8A16, fp16 activations) |
| [#375](https://github.com/ashhart/TensorFold/pull/375) | [@gilby](https://github.com/gilby) | feat(volta): Qwen3.6 MoE NVFP4 experts on sm_70 |
| [#376](https://github.com/ashhart/TensorFold/pull/376) | [@gilby](https://github.com/gilby) | feat(volta): Qwen3.8-27B NVFP4 checkpoints on two ranks (--tp 2) on sm_70 |
| [#377](https://github.com/ashhart/TensorFold/pull/377) | [@gilby](https://github.com/gilby) | feat(qwen3.8 cuda): --tp 4 for Qwen3.8-27B; rank sums of two to eight ranks |
| [#378](https://github.com/ashhart/TensorFold/pull/378) | [@gilby](https://github.com/gilby) | perf(cuda): rank sums over an engine's communicator; prompt-sized partials by copies through shared host memory |
| [#379](https://github.com/ashhart/TensorFold/pull/379) | [@gilby](https://github.com/gilby) | feat(qwen4_exp cuda): Flash Next on four ranks (--tp 4), each rank's sums over the shared rank-sum transports |
| [#380](https://github.com/ashhart/TensorFold/pull/380) | [@gilby](https://github.com/gilby) | feat(volta): Qwen3.8-Flash-Next NVFP4 checkpoints on sm_70, one GPU or four ranks |
| [#464](https://github.com/ashhart/TensorFold/pull/464) | [@liumorrisclaw](https://github.com/liumorrisclaw) | build: admit Ampere (sm_80) via TENSORFOLD_MIN_CAPABILITY, keep NVFP4/cluster floors |
| [#465](https://github.com/ashhart/TensorFold/pull/465) | [@liumorrisclaw](https://github.com/liumorrisclaw) | GLM CUDA: tensor parallel over 4 ranks, same exactness contract |
| [#466](https://github.com/ashhart/TensorFold/pull/466) | [@liumorrisclaw](https://github.com/liumorrisclaw) | CUDA: optional host-staged all-gather for PCIe-only nodes (TF_SHM_AG=1) |

### deepseek-ai/FlashMLA

| Pull request | Recorded author | Title |
| --- | --- | --- |
| [#183](https://github.com/deepseek-ai/FlashMLA/pull/183) | [@bzantium](https://github.com/bzantium) | Add SM80 (Ampere/A100) dense MLA decode kernel |
| [#189](https://github.com/deepseek-ai/FlashMLA/pull/189) | [@jasonsi1993](https://github.com/jasonsi1993) | Add SM120 (RTX 5090) sparse decode kernel |
| [#226](https://github.com/deepseek-ai/FlashMLA/pull/226) | [@PerryLink](https://github.com/PerryLink) | fix: MSVC portability - uint4 transport for shared-memory 128-bit load/store |

### dphnAI/sonar

| Pull request | Recorded author | Title |
| --- | --- | --- |
| [#1788](https://github.com/dphnAI/sonar/pull/1788) | [@AlpinDale](https://github.com/AlpinDale) | perf(sm120): optimize DeepSeek-V4.1 inference |

### flashinfer-ai/flashinfer

| Pull request | Recorded author | Title |
| --- | --- | --- |
| [#3369](https://github.com/flashinfer-ai/flashinfer/pull/3369) | [@grYe99](https://github.com/grYe99) | Feature: Add support for sm86(3090) FusedMoE in cutlass path |
| [#3739](https://github.com/flashinfer-ai/flashinfer/pull/3739) | [@lesj0610](https://github.com/lesj0610) | Enable Ampere FA2 large-head attention |
| [#5044](https://github.com/flashinfer-ai/flashinfer/pull/5044) | [@jhaotingc](https://github.com/jhaotingc) | feat: enable large-head attention on SM80, for VLLM to run FP8 kv |
| [#5224](https://github.com/flashinfer-ai/flashinfer/pull/5224) | [@kmike](https://github.com/kmike) | Enable XQA for sm80 family |
| [#5889](https://github.com/flashinfer-ai/flashinfer/pull/5889) | [@dhiraj113](https://github.com/dhiraj113) | feat(topk_varlen): exact CuTe-DSL primitives top-k backends with fused page-table output (Ampere and newer) |

### ggml-org/llama.cpp

| Pull request | Recorded author | Title |
| --- | --- | --- |
| [#28204](https://github.com/ggml-org/llama.cpp/pull/28204) | [@sihanyu03](https://github.com/sihanyu03) | cuda : MMQ q4_0 small-batch tile config for Ampere+ |
| [#29152](https://github.com/ggml-org/llama.cpp/pull/29152) | [@JohannesGaessler](https://github.com/JohannesGaessler) | CUDA: tune FA for Gemma 4 on Ampere or newer |

### local-inference-lab/vllm

| Pull request | Recorded author | Title |
| --- | --- | --- |
| [#314](https://github.com/local-inference-lab/vllm/pull/314) | [@malaiwah](https://github.com/malaiwah) | \[EXL3\] allow CUDA-graph decode for dense checkpoints by priming exl3_gemm autotune |
| [#438](https://github.com/local-inference-lab/vllm/pull/438) | [@malaiwah](https://github.com/malaiwah) | scheduler: skip speculative decoding when all scheduled requests need &lt;=1 output token |
| [#491](https://github.com/local-inference-lab/vllm/pull/491) | [@voipmonitor](https://github.com/voipmonitor) | Optimize GLM-5.3 B12X and DFlash2 MXFP8 serving |
| [#673](https://github.com/local-inference-lab/vllm/pull/673) | [@original-el8](https://github.com/original-el8) | dflash_attn: make the split-KV draft kernel available on the SM120 family |
| [#747](https://github.com/local-inference-lab/vllm/pull/747) | [@original-el8](https://github.com/original-el8) | perf(deepseek-v41): configure SWA pages for denser KV allocation |

### turboderp-org/exllamav3

| Pull request | Recorded author | Title |
| --- | --- | --- |
| [#383](https://github.com/turboderp-org/exllamav3/pull/383) | [@Frenchy2k1](https://github.com/Frenchy2k1) | Fix pre-Ampere miscompilation and silent-corruption hazards in the deterministic GEMM paths |

### vllm-project/vllm

| Pull request | Recorded author | Title |
| --- | --- | --- |
| [#41508](https://github.com/vllm-project/vllm/pull/41508) | [@MidasMining](https://github.com/MidasMining) | fix(lightning_attn): make CBLOCK device-aware to fit Ampere SMEM |
| [#43577](https://github.com/vllm-project/vllm/pull/43577) | [@ArkaD171717](https://github.com/ArkaD171717) | fix(turboquant): use SDPA prefill fallback on pre-Ampere GPUs |
| [#47629](https://github.com/vllm-project/vllm/pull/47629) | [@thomaslwang](https://github.com/thomaslwang) | \[Attention\] TRITON_MLA_SPARSE backend for SM80/SM121 sparse MLA (rebase &amp; takeover of #38476) |
| [#49131](https://github.com/vllm-project/vllm/pull/49131) | [@shivasathishs-rp](https://github.com/shivasathishs-rp) | \[Perf\] Shape-adaptive tile config for batch-invariant matmul on SM80 |
| [#50248](https://github.com/vllm-project/vllm/pull/50248) | [@luckymrwang](https://github.com/luckymrwang) | \[Bugfix\] Fix TurboQuant cache dtype propagation and FP8 store on Ampere |
| [#50882](https://github.com/vllm-project/vllm/pull/50882) | [@Hakureirm](https://github.com/Hakureirm) | \[Bugfix\] MiniMax-M3 fp8_e5m2 KV cache on SM80 (A100/A800): admit the format Triton can read, fix the CUDA-graph KV corruption |
| [#50993](https://github.com/vllm-project/vllm/pull/50993) | [@zcxGGmu](https://github.com/zcxGGmu) | \[Doc\] Clarify batch invariance Ampere support |
| [#52534](https://github.com/vllm-project/vllm/pull/52534) | [@Kenny-Hannibal](https://github.com/Kenny-Hannibal) | 魔改vllm: 支持A100 (SM_80) 部署GLM-5.2 |
| [#54138](https://github.com/vllm-project/vllm/pull/54138) | [@jiakangkangfuzhe](https://github.com/jiakangkangfuzhe) | \[Kernel\] Optimize the E=60 MoE Top-K fallback on A100 |
| [#54793](https://github.com/vllm-project/vllm/pull/54793) | [@saichowdary007](https://github.com/saichowdary007) | \[Bugfix\] Ignore empty projected KV-cache groups in PP |
| [#55506](https://github.com/vllm-project/vllm/pull/55506) | [@Karl0007](https://github.com/Karl0007) | \[Bugfix\] Index mamba spec-decode block tables by request slot, not batch row |
| [#56444](https://github.com/vllm-project/vllm/pull/56444) | [@Vaishnavi220506](https://github.com/Vaishnavi220506) | \[Bugfix\]\[Qwen4Exp\] Support N-gram PLE with pipeline parallelism |
| [#57097](https://github.com/vllm-project/vllm/pull/57097) | [@ShuoleiWang](https://github.com/ShuoleiWang) | \[Qwen3.8-Flash-Next\] Fuse main QK-norm/RoPE/gate and KV-cache write into the QSA pre-indexer launch |
| [#57105](https://github.com/vllm-project/vllm/pull/57105) | [@gau-nernst](https://github.com/gau-nernst) | \[Qwen3.8-Flash-Next\] Avoid memory fragmentation in QSA indexer logits workspace |
| [#58114](https://github.com/vllm-project/vllm/pull/58114) | [@ZJY0516](https://github.com/ZJY0516) | \[Perf\]\[Qwen3.8\] Reduce PLE metadata construction overhead |
| [#58449](https://github.com/vllm-project/vllm/pull/58449) | [@bilikaz](https://github.com/bilikaz) | \[Spec Decode\]\[Qwen3.8\] Enable fused draft metadata updates for the QSA side cache |
| [#58762](https://github.com/vllm-project/vllm/pull/58762) | [@lucamotz](https://github.com/lucamotz) | \[Perf\]\[MRV2\] Reuse Mamba/GDN metadata across KV cache groups |
| [#58763](https://github.com/vllm-project/vllm/pull/58763) | [@lucamotz](https://github.com/lucamotz) | \[Perf\]\[GDN\] Slice pure spec-decode rows instead of a host-mask gather |
| [#60491](https://github.com/vllm-project/vllm/pull/60491) | [@nkuhn-vmw](https://github.com/nkuhn-vmw) | \[Model\]\[Qwen4Exp\] fp8_e5m2 KV cache on the QSA path for SM80/SM86 |
| [#60596](https://github.com/vllm-project/vllm/pull/60596) | [@khluu](https://github.com/khluu) | \[CI\] Remove A100 steps that the pipeline generator no longer emits |
| [#60602](https://github.com/vllm-project/vllm/pull/60602) | [@kimseunghyun-kr](https://github.com/kimseunghyun-kr) | \[Model\]\[Qwen4Exp\] FP8 QSA main KV cache on Ampere (sm_80, sm_86) |

## Articles and public benchmark references

Thanks also to the authors of the [CUDA memory-sharing article](https://piyushk52.github.io/jekyll/update/2026/10/22/mem_share.html),
the [NVFP4 decode article](https://humansand.ai/blog/faster-than-light-nvfp4-decode),
and the [HyperQwen performance discussion](https://www.reddit.com/r/LocalLLaMA/comments/1vtup5s/i_pushed_qwen3827b_to_381_tps_for_a_single/).
Benchmark and recipe authors including Curtis, MiaAI-Lab, Fraser Price and the Pi community are
acknowledged through the linked projects and [benchmark methods](results/BENCHMARKS.md).
