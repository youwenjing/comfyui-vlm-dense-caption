# ComfyUI VLM Dense Caption

使用视觉语言模型（VLM）对图片进行稠密描述打标，并基于描述生成产品编辑指令。

---

## 节点总览

| 节点 | 功能 | 输入 | 输出 |
|------|------|------|------|
| **VLM Dense Caption** | 单张图片稠密打标 | 图片 + API 配置 | 一条 caption 文本 |
| **VLM Dense Caption (Batch)** | 多张图片逐张打标 | 图片 + API 配置 | 多条 caption（分隔符拼接） |
| **LLM Edit Instruction** | 单条编辑指令生成 | 源描述 + 编辑类型 | instruction + target_caption |
| **LLM Edit Instruction (Batch)** | 批量编辑指令生成 | 源描述 + 数量 | 多条 instruction + target_caption |

---

## 节点 1: VLM Dense Caption

对**单张**图片调用 VLM，输出一段稠密描述文本。

### 参数说明

| 参数 | 类型 | 默认值 | 范围 | 说明 |
|------|------|--------|------|------|
| `image` | IMAGE | — | — | 输入图片。连接 Load Image 等节点的 IMAGE 输出。 |
| `base_url` | STRING | (空) | — | OpenAI 兼容 API 的基础地址。会自动补全 `/v1` 后缀。 |
| `api_key` | STRING | (空) | — | API 密钥。若后端不需要鉴权可留空。也支持通过环境变量 `LLM_GATEWAY_API_KEY` 设置。 |
| `vl_model` | STRING | `Qwen2.5-VL-32B-Instruct-AWQ` | — | 视觉语言模型名称。需与后端支持的模型名完全一致。 |
| `prompt` | STRING | 鞋类稠密描述 prompt | — | 打标指令文本。**多行输入**，可自定义描述维度和输出格式。 |
| `temperature` | FLOAT | `0.3` | 0.0 ~ 2.0，步长 0.05 | 采样温度。越低输出越确定，越高越有创造性。打标场景建议 0.1~0.3。 |
| `max_tokens` | INT | `2000` | 1 ~ 32768，步长 1 | 最大输出 token 数。稠密描述建议 1500~3000。 |
| `clean_markdown` | BOOLEAN | `true` | — | 是否清理 Markdown 格式。开启后移除 `#` 标题、`**` 粗体、`---` 分隔线等。 |
| `verify_ssl` | BOOLEAN | `true` | — | 是否验证 SSL 证书。内网自签名证书时设为 `false`。HTTP 地址不需要关闭。 |

### 输出

| 输出名 | 类型 | 说明 |
|--------|------|------|
| `caption` | STRING | VLM 生成的稠密描述文本 |

### 工作流示例

```
Load Image ──▶ VLM Dense Caption ──▶ Show Text / Save Text / CLIP Text Encode
```

---

## 节点 2: VLM Dense Caption (Batch)

对**多张**图片逐张调用 VLM，用分隔符拼接所有结果后输出。

### 参数说明

除以下额外参数外，其余参数含义与 **VLM Dense Caption** 完全相同：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `images` | IMAGE | — | 输入图片（batch）。可以是多张图的 batch 维度。 |
| `separator` | STRING | `\n\n---\n\n` | 分隔符。每条 caption 之间的分隔字符串。 |

> 共享参数：`base_url`、`api_key`、`vl_model`、`prompt`、`temperature`、`max_tokens`、`clean_markdown`、`verify_ssl` 同上。

### 输出

| 输出名 | 类型 | 说明 |
|--------|------|------|
| `captions` | STRING | 所有图片的 caption 用分隔符拼接后的文本 |

### 工作流示例

```
Load Images (Batch) ──▶ VLM Dense Caption (Batch) ──▶ Save Text
```

---

## 节点 3: LLM Edit Instruction

根据源图片的稠密描述和指定的编辑类型，调用 LLM 生成一条**编辑指令**和**修改后的目标描述**。

### 参数说明

| 参数 | 类型 | 默认值 | 范围 | 说明 |
|------|------|--------|------|------|
| `source_caption` | STRING | (空) | — | 源图片的稠密描述。多行输入，一般来自 VLM Dense Caption 的输出。 |
| `edit_type` | 下拉选择 | `material_swap` | 12 种类型 | 编辑类型（详见下方对照表）。 |
| `base_url` | STRING | (空) | — | OpenAI 兼容 API 地址。 |
| `api_key` | STRING | (空) | — | API 密钥。也支持环境变量 `LLM_GATEWAY_API_KEY`。 |
| `llm_model` | STRING | `Qwen2.5-32B-Instruct` | — | 文本 LLM 模型名。用于生成编辑指令（纯文本模型即可）。 |
| `temperature` | FLOAT | `0.7` | 0.0 ~ 2.0，步长 0.05 | 采样温度。编辑指令需要一定创造性，建议 0.6~0.9。 |
| `max_tokens` | INT | `1500` | 1 ~ 32768，步长 1 | 最大输出 token 数。 |
| `max_retries` | INT | `2` | 0 ~ 5，步长 1 | 最大重试次数。JSON 解析失败时自动重试。0 表示不重试。 |
| `verify_ssl` | BOOLEAN | `true` | — | 是否验证 SSL 证书。内网自签名证书时设为 `false`。 |
| `style_target` | STRING | (空) | — | 可选。仅当 `edit_type=style_mod` 时生效。填入目标品类名，如 `loafers (乐福鞋)`。留空由模型自由选择。 |

### 编辑类型 (`edit_type`) 对照表

| 类型 | 含义 | 示例 |
|------|------|------|
| `material_swap` | 更换鞋面材质 | 网布 → 全粒面牛皮 |
| `color_change` | 改变颜色（整体或局部） | 蓝色 → 酒红色 |
| `component_mod` | 修改组件（鞋带/鞋舌/鞋口等） | 扁鞋带 → 弹性快速系带 |
| `accessory_add` | 添加/修改配饰 | 添加珍珠脚踝绑带 |
| `texture_upgrade` | 材质纹理升级 | 平滑皮革 → 鳄鱼压纹 |
| `style_shift` | 风格微调 | 运动 → 休闲/街头 |
| `sole_change` | 鞋底改造 | 橡胶底 → EVA 厚底 |
| `pattern_add` | 添加图案/花纹 | 豹纹/迷彩/条纹 |
| `toe_mod` | 鞋头类型修改 | 圆头 → 尖头 |
| `heel_mod` | 鞋跟类型修改 | 平底 → 小猫跟 |
| `craft_upgrade` | 工艺升级 | 添加刺绣/绗缝/压纹 |
| `style_mod` | 款式/品类变换 | 运动鞋 → 乐福鞋（保留配色风格） |

### 输出

| 输出名 | 类型 | 说明 |
|--------|------|------|
| `instruction` | STRING | 编辑指令（英文），描述具体如何修改 |
| `target_caption` | STRING | 修改后的完整产品描述 |

### 工作流示例

```
VLM Dense Caption ──▶ LLM Edit Instruction ──▶ Show Text / 下游节点
```

---

## 节点 4: LLM Edit Instruction (Batch)

对同一张源图描述，**批量生成多条不同类型**的编辑指令。

### 参数说明

除以下额外参数外，其余参数含义与 **LLM Edit Instruction** 完全相同：

| 参数 | 类型 | 默认值 | 范围 | 说明 |
|------|------|--------|------|------|
| `num_edits` | INT | `3` | 1 ~ 12，步长 1 | 生成编辑指令的数量。 |
| `fixed_edit_types` | STRING | `style_mod` | — | 固定编辑类型（逗号分隔）。这些类型一定会出现在结果中。留空则全部随机。如 `style_mod,color_change`。 |
| `enable_style_mod` | BOOLEAN | `true` | — | 是否在随机抽取时包含 `style_mod`。如果 `fixed_edit_types` 中已指定 `style_mod`，该开关不影响固定列表。 |

> 共享参数：`source_caption`、`base_url`、`api_key`、`llm_model`、`temperature`、`max_tokens`、`max_retries`、`verify_ssl` 同上。

### 输出

| 输出名 | 类型 | 说明 |
|--------|------|------|
| `all_instructions` | STRING | 所有编辑指令，每条前缀 `[edit_type]`，用双换行分隔 |
| `all_target_captions` | STRING | 所有目标描述，每条前缀 `[edit_type]`，用双换行分隔 |

### 工作流示例

```
VLM Dense Caption ──▶ LLM Edit Instruction (Batch) ──▶ Show Text / Save Text
```

---

## 完整工作流（端到端）

```
Load Image ──▶ VLM Dense Caption ──▶ LLM Edit Instruction (Batch) ──▶ Save Text
                                 │
                                 └──▶ CLIP Text Encode ──▶ KSampler ──▶ VAE Decode ──▶ Save Image
```

---

## 各后端配置示例

| 后端 | `base_url` | 适用模型 |
|------|-----------|----------|
| 阿里 DashScope | `https://dashscope-intl.aliyuncs.com/compatible-mode` | qwen-vl-max / qwen-vl-plus |
| 本地 vLLM | `http://localhost:8000` | 任意部署的 VLM / LLM |
| Ollama (OpenAI 兼容) | `http://localhost:11434` | llava / minicpm-v 等 |
| OpenAI 官方 | `https://api.openai.com` | gpt-4o / gpt-4-vision-preview |

---

## 默认 prompt（鞋类稠密描述）

节点内置的默认打标 prompt 覆盖以下维度：

- 鞋型/轮廓（运动鞋、休闲鞋、跑鞋、篮球鞋等）
- 整体形状（鞋头造型、鞋身线条、鞋口高度）
- 鞋面材质（皮革、网布、帆布、针织、麂皮等，含光泽和纹理）
- 部件材质（鞋舌、鞋跟、鞋头、侧面板等）
- 颜色分布（主色、辅色、强调色及其区域）
- 鞋底特征（中底材质/颜色、外底纹路、厚度）
- 闭合系统（鞋带类型、魔术贴、弹性鞋口等）
- 配饰/细节（Logo 位置和风格、缝线颜色、反光条、装饰）
- 光照与背景（光线方向、背景颜色、是否影棚/纯色背景）

如需自定义，直接在节点 `prompt` 参数中修改即可。

---

## 依赖

| 包 | 最低版本 | 说明 |
|----|----------|------|
| `httpx` | — | HTTP 客户端（pip 安装） |
| `torch` | ≥ 2.0.0 | ComfyUI 自带 |
| `numpy` | ≥ 1.24.0 | ComfyUI 自带 |
| `Pillow` | ≥ 9.0.0 | ComfyUI 自带 |

---

## 安装

### 方法 1：ComfyUI Manager（推荐）

在 ComfyUI Manager 中搜索 `VLM Dense Caption` 安装。

### 方法 2：手动安装

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/yourname/comfyui-vlm-dense-caption.git
cd comfyui-vlm-dense-caption
pip install -r requirements.txt
```

重启 ComfyUI。

---

## License

MIT