# GLM-ASR HW1 Report Checklist（基于 `attention` 分支）

## 0. 使用原则

- [ ] 本 checklist 只围绕我们 **当前分支真实改过的内容** 来写
- [ ] 报告主线定为：**FlashAttention-like attention + layer fusion / tiling + 端到端验证**
- [ ] 不把整条 GLM-ASR pipeline 都描述成“我们全部重写/全部优化”
- [ ] 不把单条测试音频的 benchmark 写成“大规模 ASR accuracy evaluation”
- [ ] 所有最终结论都必须有对应的代码证据、benchmark 结果或 profiling 证据支撑
- [ ] 所有正式实验输出都保存到 `hw1-asr/report_prep/results/`

---

## 1. 先确认我们到底改了什么

### 1.1 `attention.py`
- [ ] 确认基础路径包含：
  - [ ] `attention_scores_kernel`
  - [ ] `softmax_inplace_kernel`
  - [ ] `attention_output_kernel`
- [ ] 确认 fused 路径包含：
  - [ ] `flash_attention_fwd_kernel`
- [ ] 确认支持的功能包含：
  - [ ] causal attention
  - [ ] additive attention mask
  - [ ] GQA（Grouped Query Attention）
  - [ ] `q_len < k_len` 的 decode-like path
- [ ] 确认 `__main__` 中的 reference checks 都能作为正确性证据使用

### 1.2 `layers.py`
- [ ] 确认 `Linear` 有明确 tile 参数：
  - [ ] `TILE_M`
  - [ ] `TILE_N`
  - [ ] `TILE_K`
  - [ ] `NUM_WARPS`
  - [ ] `NUM_STAGES`
- [ ] 确认存在至少一个 fused kernel：
  - [ ] `linear_gelu_kernel`
  - [ ] `swiglu_fused_kernel`
- [ ] 确认 fused path 已接入：
  - [ ] `MLP._forward_fused`
  - [ ] `EncoderMLP._forward_fused`
- [ ] 确认 `Linear.BACKEND` / auto fallback 逻辑能讲清楚

### 1.3 `rope.py`
- [ ] 确认 `compute_freqs_kernel` 已存在
- [ ] 确认 `rope.py` 自测能正常运行
- [ ] 报告里如果提到 Triton track 的三个目标文件，不要漏掉 `rope.py`

---

## 2. 必跑文件与命令

> 下面这些是建议最少要跑的。  
> 建议统一在 `hw1-asr/` 目录下执行。

### 2.0 环境信息
- [ ] `nvidia-smi | tee report_prep/results/01_nvidia_smi.txt`
- [ ] `python -c "import sys, torch, triton; print('python', sys.version); print('torch', torch.__version__); print('cuda', torch.version.cuda); print('cuda_available', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'); print('triton', triton.__version__)" | tee report_prep/results/02_python_env.txt`

### 2.1 基础端到端 benchmark
- [ ] `./benchmark.sh glm_asr_triton_example --warmup 1 --runs 5 | tee report_prep/results/20_benchmark_example.txt`
- [ ] `./benchmark.sh glm_asr_triton_template --warmup 1 --runs 5 | tee report_prep/results/21_benchmark_template_final.txt`

### 2.2 详细 profiling
- [ ] `./benchmark_detailed.sh glm_asr_triton_example --runs 3 | tee report_prep/results/31_detailed_example.txt`
- [ ] `./benchmark_detailed.sh glm_asr_triton_template --runs 3 | tee report_prep/results/30_detailed_template.txt`

### 2.3 多次运行，减少偶然波动
- [ ] 如果 3 次结果波动大，再补：
  - [ ] `./benchmark_detailed.sh glm_asr_triton_template --runs 5`
  - [ ] `./benchmark_detailed.sh glm_asr_triton_example --runs 5`

### 2.4 Nsight Systems
- [ ] `./benchmark_detailed.sh glm_asr_triton_template --nsys --runs 1 | tee report_prep/results/40_nsys_template.txt`
- [ ] `./benchmark_detailed.sh glm_asr_triton_example --nsys --runs 1 | tee report_prep/results/41_nsys_example.txt`

### 2.5 attention 单独正确性测试
- [ ] `python glm_asr_triton_template/attention.py | tee report_prep/results/10_attention_selftest.txt`

### 2.6 layers 单独功能测试
- [ ] `python glm_asr_triton_template/layers.py | tee report_prep/results/11_layers_selftest.txt`

### 2.7 rope 单独功能测试
- [ ] `python glm_asr_triton_template/rope.py | tee report_prep/results/12_rope_selftest.txt`

---

## 3. 正确性检查

### 3.1 端到端 benchmark 正确性
- [ ] 记录 `glm_asr_triton_example` 的输出：
  - [ ] Time
  - [ ] Tokens
  - [ ] ms/token
  - [ ] Transcription
  - [ ] Accuracy
  - [ ] PASS / FAIL
- [ ] 记录 `glm_asr_triton_template` 的同类输出
- [ ] 确保最终报告里能明确说明：
  - [ ] template 至少通过 benchmark 的 correctness check
  - [ ] transcription 不是空输出
  - [ ] 没有明显数值错误或生成崩溃

### 3.2 `attention.py` 的 reference checks
- [ ] basic
- [ ] causal
- [ ] masked
- [ ] masked + causal
- [ ] gqa
- [ ] gqa + causal
- [ ] `q_len < k_len`
- [ ] decode-like causal
- [ ] GQA decode-like causal

### 3.3 `layers.py` 的基础功能检查
- [ ] RMSNorm
- [ ] LayerNorm
- [ ] GELU
- [ ] SiLU
- [ ] Linear
- [ ] Embedding
- [ ] Softmax
- [ ] MLP

### 3.4 正确性写作注意
- [ ] 报告里写“benchmark correctness”或“reference check correctness”
- [ ] 不把这部分夸大成完整 ASR 数据集精度评估

---

## 4. 报告 Section 1 需要准备的材料（Introduction / Overview）

### 4.1 背景
- [ ] 用中文先写清楚：
  - [ ] GLM-ASR 是什么
  - [ ] 任务是 audio-to-text inference
  - [ ] 为什么 GPU kernel optimization 对 latency 重要
  - [ ] 为什么 attention 与 MLP/linear 是重点优化对象

### 4.2 系统总览图
- [ ] 画一张整体流程图：
  - [ ] Audio
  - [ ] Audio Encoder
  - [ ] Multi-modal Projector
  - [ ] Text Decoder
  - [ ] Text
- [ ] 在图中标出：
  - [ ] 我们真正改动的部分
  - [ ] `attention.py` 是核心改动点
  - [ ] `layers.py` 是辅助优化点
- [ ] 不要把未改模块画成“我们的创新”

### 4.3 Introduction 的一句话主线
- [ ] 写成类似：
  - [ ] 我们在 Triton track 下，重点优化了 decoder 中的 attention 路径，并结合 layer-level fusion 与 tile tuning，在完整 GLM-ASR pipeline 中验证 correctness 与 latency 改善

---

## 5. 报告 Section 2 需要准备的材料（Implementation）

### 5.1 attention 实现说明
- [ ] 说明基础路径是怎么分成三步的：
  - [ ] QK^T
  - [ ] softmax
  - [ ] weights @ V
- [ ] 说明 fused 路径为什么更重要：
  - [ ] 减少中间张量读写
  - [ ] 减少 kernel launch overhead
  - [ ] 更符合 FlashAttention-style memory-efficient design
- [ ] 说明 mask 处理方式
- [ ] 说明 causal path 处理方式
- [ ] 说明 GQA 的 head mapping 方式

### 5.2 layer 实现说明
- [ ] 说明 `Linear` 的 tile 化矩阵乘
- [ ] 说明 `linear_gelu_kernel` 的 fusion 逻辑
- [ ] 说明 `swiglu_fused_kernel` 的 fusion 逻辑
- [ ] 说明 `MLP` / `EncoderMLP` 的 fused path 如何接入整体模型

### 5.3 实现参数表
- [ ] 整理一个表，至少包含：
  - [ ] kernel 名称
  - [ ] `BLOCK_M / BLOCK_N / BLOCK_K`
  - [ ] `num_warps`
  - [ ] `num_stages`
  - [ ] 适用模块
  - [ ] 最终是否采用

---

## 6. 必须收集的端到端结果

### 6.1 baseline vs template
- [ ] 记录 `glm_asr_triton_example`
- [ ] 记录 `glm_asr_triton_template`
- [ ] 同时保留：
  - [ ] mean latency
  - [ ] std
  - [ ] tokens
  - [ ] ms/token
  - [ ] transcription
  - [ ] accuracy / PASS

### 6.2 报告里要回答的问题
- [ ] 我们的 template 是否比 example 更快？
- [ ] 如果没有全面更快，是哪些阶段更快、哪些阶段没有收益？
- [ ] correctness 是否保持？
- [ ] 速度变化是否稳定，还是波动很大？
- [ ] benchmark 输出里的 `ATTENTION DISPATCH SUMMARY` 是否显示 flash path 真的被大量命中？
- [ ] benchmark 输出里的 `LAYER DISPATCH SUMMARY` 是否显示 Linear / MLP 的优化路径真的被使用？

---

## 7. 详细 profiling 数据收集

> `benchmark_detailed.py` 已经能给出比较直接的分模块结果。  
> 这部分主要服务于 Section 4 / Section 6。

### 7.1 必收集模块
- [ ] `audio_encoder`
- [ ] `projector`
- [ ] `decoder_prefill`
- [ ] `decode_step`
- [ ] `layers[:5]`（如果能正常输出）

### 7.2 baseline / template 对照
- [ ] 为 example 收集一份完整 detailed profiling
- [ ] 为 template 收集一份完整 detailed profiling
- [ ] 对照每一项：
  - [ ] 哪一项更快
  - [ ] 哪一项更慢
  - [ ] 差异是否明显
  - [ ] 哪一项占总时间比例最高

### 7.3 写作注意
- [ ] 这部分叫“component-level profiling”或“operator-level profiling”
- [ ] 不要把它误写成精确的 kernel-level hardware diagnosis
- [ ] 如果 detailed profiling 和 end-to-end 的结论不一致，用 dispatch summary 帮助解释原因

---

## 8. 优化实验记录（最重要）

> 这一部分是报告里最值钱的内容。  
> 每个优化都必须按 **Hypothesis -> Change -> Result** 记录。

### 8.1 FlashAttention-like attention
- [ ] Hypothesis：fused attention 能减少中间内存访问与 launch overhead
- [ ] Change：启用 / 调整 `flash_attention_fwd_kernel`
- [ ] Result：记录前后 latency 变化
- [ ] Result：记录 correctness 是否保持
- [ ] 备注：如果收益不明显，也要写真实原因

### 8.1b Flash 开关 A/B
- [ ] 用 `ENABLE_FLASH_ATTENTION = True` 跑一组正式 benchmark / detailed profiling
- [ ] 用 `ENABLE_FLASH_ATTENTION = False` 跑一组正式 benchmark / detailed profiling
- [ ] Flash on / off 的输出文件分开命名，避免覆盖
- [ ] 对比：
  - [ ] end-to-end latency
  - [ ] component-level profiling
  - [ ] `ATTENTION DISPATCH SUMMARY`
- [ ] 报告里明确写：这个对比衡量的是 attention fast path on/off，而不是纯隔离单个 kernel 的理论贡献

### 8.2 attention 配置调优
- [ ] 尝试至少 2–3 组配置
- [ ] 记录：
  - [ ] `BLOCK_M`
  - [ ] `BLOCK_N`
  - [ ] `num_warps`
  - [ ] `num_stages`
- [ ] 标出最终采用的配置
- [ ] 写明为什么选它

### 8.3 `Linear` tile tuning
- [ ] 记录尝试过的：
  - [ ] `TILE_M`
  - [ ] `TILE_N`
  - [ ] `TILE_K`
- [ ] 记录最终配置
- [ ] 判断：
  - [ ] 是不是所有 linear 都受益
  - [ ] 还是只在特定 shape 下受益

### 8.4 `linear_gelu_kernel`
- [ ] 验证 fused `Linear + GELU` 是否确实接入
- [ ] 收集有无收益
- [ ] 如果收益有限，记录原因

### 8.5 `swiglu_fused_kernel`
- [ ] 验证 fused SwiGLU 是否确实接入
- [ ] 收集有无收益
- [ ] 如果收益有限，记录原因

### 8.6 每个优化都要保留失败记录
- [ ] 哪个配置更差
- [ ] 更差多少
- [ ] 可能原因是什么
- [ ] 为什么最终没采用

---

## 9. Nsight Systems / 瓶颈分析

### 9.1 必做
- [ ] 采集 template 的 `.nsys-rep`
- [ ] 采集 example 的 `.nsys-rep`

### 9.2 观察重点
- [ ] attention 相关 kernel 是否成为热点
- [ ] prefill 和 decode 哪个更重
- [ ] 是否存在大量小 kernel launch
- [ ] 是否存在明显空洞 / synchronization gap
- [ ] 是否存在 memory-bound 特征迹象
- [ ] 是否存在 compute 没有吃满的现象
- [ ] 是否和 `ATTENTION DISPATCH SUMMARY` 中的 flash 命中率一致
- [ ] 是否和 `LAYER DISPATCH SUMMARY` 中的 backend / fused 使用率一致

### 9.3 写作注意
- [ ] 只在有 Nsight 证据时再谈更底层的瓶颈
- [ ] 没有 Nsight 证据时，只能保守地说“可能原因”

---

## 10. 报告 Section 4 / 5 / 6 的证据整理

### 10.1 Section 4（Profiling / Bottleneck）
- [ ] 用 `benchmark_detailed.sh` 的结果说明高层瓶颈
- [ ] 用 Nsight 补充更细的热点观察
- [ ] 用 `ATTENTION DISPATCH SUMMARY` 解释 attention 优化为什么收益大/小
- [ ] 用 `LAYER DISPATCH SUMMARY` 解释 layer 优化为什么收益大/小
- [ ] 明确指出：
  - [ ] 最大耗时阶段
  - [ ] 我们优化是否命中这个阶段
  - [ ] 为什么有的优化对端到端收益有限

### 10.2 Section 5（Optimization）
- [ ] 至少选 2–3 个最有代表性的优化实验写进去
- [ ] 每个实验都要有 before / after
- [ ] 每个实验都要有 hypothesis
- [ ] 每个实验都要有结果解释
- [ ] 至少有一个 fused kernel 的实验被完整展示

### 10.3 Section 6（Comparison / Discussion）
- [ ] baseline vs template 的最终表格
- [ ] 哪些模块变快
- [ ] 哪些模块没变快
- [ ] correctness 是否保持
- [ ] root cause 分析是否有足够证据支撑

---

## 11. 最终表格与图片清单

### 11.1 表格
- [ ] Table 1：baseline vs template 端到端结果
- [ ] Table 2：detailed profiling 分模块对比
- [ ] Table 3：attention / layer 配置调优结果
- [ ] Table 4：最终采用的 kernel 参数表

### 11.2 图片
- [ ] Figure 1：GLM-ASR 系统总览图
- [ ] Figure 2：我们修改的 attention / layer 结构概览
- [ ] Figure 3：profiling 或 Nsight 热点图
- [ ] Figure 4：优化前后对比图（可选）

---

## 12. 写作时的红线

- [ ] 不写“我们优化了整个 GLM-ASR 的所有模块”
- [ ] 不写“我们完成了完整 ASR accuracy evaluation”
- [ ] 不把 benchmark 的一个测试音频说成大规模实验
- [ ] 不把 component-level profiling 说成硬件级最终定论
- [ ] 没有数据支撑的地方不用强行下结论
- [ ] 如果某个优化没有收益，也照实写

---

## 13. 交稿前最后检查

- [ ] 所有表格里的数字都能追溯到具体运行结果
- [ ] 所有结论都能对应到代码实现或 profiling 证据
- [ ] 所有命令都真正跑通过
- [ ] 最终版本只保留真实采用的配置
- [ ] 报告主线始终围绕：
  - [ ] FlashAttention-like attention
  - [ ] layer fusion / tile tuning
  - [ ] end-to-end GLM-ASR validation
