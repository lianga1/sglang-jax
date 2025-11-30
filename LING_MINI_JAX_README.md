# Ling-mini-2.0 JAX Implementation

本项目演示如何使用 JAX 框架加载和运行 Ling-mini-2.0 模型的推理。

## 模型信息

**Ling-mini-2.0** 是一个基于 BailingMoeV2 架构的大规模语言模型，采用 Mixture of Experts (MoE) 设计：

- **架构**: BailingMoeV2ForCausalLM
- **层数**: 20 层 Transformer 解码器
- **隐藏维度**: 2048
- **注意力头**: 16 个 (4 个 KV 头，GQA 架构)
- **专家数量**: 256 个专家
- **每 Token 激活专家**: 8 个
- **共享专家**: 1 个
- **词汇表大小**: 157,184
- **位置编码**: RoPE (theta=600,000)
- **数据类型**: bfloat16
- **特殊特性**:
  - QK-Norm 注意力稳定化
  - Group-limited Top-K 专家选择
  - 专家偏置 (Expert Bias)
  - 路由缩放因子 (Routed Scaling Factor)

## 文件说明

### 1. `ling_mini_jax_simple.py` (推荐)
简化的 JAX 实现，展示核心流程：
- 加载 config.json 配置
- 初始化 JAX 模型
- 加载 safetensors 权重
- 运行前向推理

### 2. `ling_mini_jax_demo.py`
完整的 JAX 实现，包含：
- 模型加载器类 (LingMiniJAXInference)
- 文本生成功能
- 详细的日志记录
- 多种使用示例

## 环境要求

```bash
# 安装 JAX
pip install jax[cuda] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# 安装依赖
pip install -e .
```

## 使用方法

### 简单示例

```python
from ling_mini_jax_simple import load_ling_mini_model, run_simple_inference
import jax.numpy as jnp

# 加载模型
model_config, model = load_ling_mini_model(
    model_path="/path/to/Ling-mini-2.0",
    dtype=jnp.bfloat16
)

# 创建输入令牌
input_tokens = jax.random.randint(
    jax.random.PRNGKey(0),
    (1, 10),  # batch_size, seq_len
    0, 1000,
    dtype=jnp.int32
)

# 运行推理
logits = run_simple_inference(model_config, model, input_tokens)
print(f"输出形状: {logits.shape}")  # (1, 10, 157184)
```

### 详细示例

```python
from ling_mini_jax_demo import LingMiniJAXInference
import jax.numpy as jnp

# 初始化推理器
inference = LingMiniJAXInference(
    model_path="/path/to/Ling-mini-2.0",
    dtype=jnp.bfloat16,
    tensor_parallel_size=1
)

# 加载模型
inference.load_model()

# 前向推理
input_ids = jax.random.randint(...)
outputs = inference.forward(input_ids)
print(f"Logits: {outputs['logits'].shape}")

# 文本生成
generated = inference.generate(
    prompt,
    max_new_tokens=100,
    temperature=0.7,
    top_p=0.9
)
```

## 运行脚本

```bash
# 简单示例
python ling_mini_jax_simple.py

# 完整演示
python ling_mini_jax_demo.py
```

## 模型权重

模型权重存储在以下格式中：
- **格式**: Safetensors
- **文件**: model-00001-of-00004.safetensors, model-00002-of-00004.safetensors, ...
- **索引**: model.safetensors.index.json

权重加载由 `WeightLoader` 类处理，它会将 PyTorch 权重映射到 JAX 模型。

## 核心组件

### BailingMoEModel
主要的模型类，包含：
- **嵌入层** (Embeddings)
- **解码器层** (Decoder Layers)
- **RMSNorm** (层归一化)
- **RoPE** (旋转位置编码)

### BailingMoEDecoderLayer
单个解码器层，包含：
- **BailingMoEAttention** (注意力机制)
- **MoE 模块** (256 个专家)
- **RMSNorm** 层

### EPMoE
专家并行 Mixture of Experts 模块：
- 专家路由 (Gate)
- Top-K 选择
- 专家并行计算

## 性能优化

### 1. 张量并行
```python
# 在多个 GPU 上并行
inference = LingMiniJAXInference(
    model_path="...",
    tensor_parallel_size=4  # 使用 4 个 GPU
)
```

### 2. 混合精度
```python
# 使用 bfloat16 加速
dtype = jnp.bfloat16

# 或 float16
dtype = jnp.float16
```

### 3. KV 缓存
对于生成任务，使用 KV 缓存可以显著加速：
```python
# 在实际实现中需要正确配置 KV 缓存
token_to_kv_pool = KVCache(...)
```

## 注意事项

1. **内存要求**: 模型较大 (约 33GB)，需要充足 GPU 内存
2. **权重映射**: PyTorch 和 JAX 的权重布局可能需要转置
3. **数据类型**: 建议使用 bfloat16 平衡精度和性能
4. **并行配置**: 根据 GPU 数量调整 tensor_parallel_size
5. **专家负载均衡**: MoE 模型需要确保专家使用的均衡性

## 故障排除

### 常见错误

1. **权重加载失败**
   ```
   检查模型路径是否正确
   确保所有 safetensors 文件都存在
   ```

2. **内存不足**
   ```
   减少 batch_size
   使用梯度检查点
   启用张量并行
   ```

3. **数值不稳定**
   ```
   使用 bfloat16 而不是 float16
   检查输入数据的范围
   ```

### 调试提示

```python
# 启用详细日志
logging.basicConfig(level=logging.DEBUG)

# 检查模型配置
print(f"Model config: {model_config.hf_config}")

# 检查权重形状
for name, param in model.params.items():
    print(f"{name}: {param.shape}")
```

## 参考资料

- [JAX 文档](https://jax.readthedocs.io/)
- [Flax 文档](https://flax.readthedocs.io/)
- [sglang-jax 项目](https://github.com/sgl-project/sglang-jax)
- [Ling-mini-2.0 模型](https://huggingface.co/inclusionAI/Ling-mini-2.0)
- [RoPE 位置编码](https://arxiv.org/abs/2104.09864)
- [Mixture of Experts](https://arxiv.org/abs/1701.06538)

## 许可证

本示例代码遵循 MIT 许可证。Ling-mini-2.0 模型本身的许可证请参考其官方文档。
