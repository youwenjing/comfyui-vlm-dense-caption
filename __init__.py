"""
ComfyUI VLM Dense Caption — 使用视觉语言模型对图片进行稠密描述打标。

GitHub: https://github.com/youwenjing/comfyui-vlm-dense-caption

安装方式:
    1. 克隆到 ComfyUI/custom_nodes/
       git clone https://github.com/yourname/comfyui-vlm-dense-caption.git
    2. 安装依赖
       cd comfyui-vlm-dense-caption && pip install -r requirements.txt
    3. 重启 ComfyUI
"""

import base64
import io
import json
import random
import re

import numpy as np
import torch
from openai import OpenAI
from PIL import Image

# ============================================================
# 默认 VLM 稠密打标 prompt（鞋类）
# ============================================================

DEFAULT_VLM_PROMPT = (
    "Please provide an extremely detailed, deconstructed description of the core "
    "subject (footwear/shoe) in the image.\n\n"
    "## Dimensions that MUST be covered\n"
    "- Shoe type/silhouette (sneaker, casual, runner, basketball, etc.)\n"
    "- Overall shape (toe shape, body lines, collar height)\n"
    "- Upper material (leather, mesh, canvas, knit, suede, etc. — note gloss, texture)\n"
    "- Component materials (tongue, heel, toe cap, side panels, etc.)\n"
    "- Color distribution (primary, secondary, accent colors and their regions)\n"
    "- Sole features (midsole material/color, outsole tread, thickness)\n"
    "- Closure system (lace type, velcro, elastic collar, etc.)\n"
    "- Accessories/details (logo position & style, stitching color, reflective strips, decorations)\n"
    "- Lighting & background (light direction, background color, whether studio/plain)\n\n"
    "## Output format\n"
    "Write a single natural, fluent English paragraph (3-5 sentences).\n"
    "Do NOT use markdown formatting, lists, or bullet points.\n"
    "Output only the description paragraph, no prefix or explanation."
)

# ============================================================
# 工具函数
# ============================================================

def _tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    """[B, H, W, C] tensor → PIL Image (取第一张，转 RGB)"""
    arr = image_tensor[0].cpu().numpy()
    arr = (arr * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")


def _pil_to_base64(img: Image.Image, fmt: str = "JPEG") -> str:
    """PIL Image → base64 string"""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _clean_markdown(text: str) -> str:
    """清理 markdown 格式"""
    lines = text.strip().split("\n")
    cleaned = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("#") or (line.startswith("**") and ":**" in line):
            continue
        if line in ("---", "***"):
            continue
        line = line.replace("**", "").replace("*", "")
        cleaned.append(line)
    result = " ".join(cleaned)
    if result and result[-1] not in ".。":
        result += "."
    return result


# ============================================================
# 节点 1: VLMDenseCaption — 单图打标
# ============================================================

class VLMDenseCaption:
    """对单张图片进行 VLM 稠密描述打标"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "base_url": (
                    "STRING",
                    {
                        "default": "your_url",
                        "multiline": False,
                    },
                ),
                "api_key": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                    },
                ),
                "vl_model": (
                    "STRING",
                    {
                        "default": "Qwen2.5-VL-32B-Instruct-AWQ",
                        "multiline": False,
                    },
                ),
                "prompt": (
                    "STRING",
                    {
                        "default": DEFAULT_VLM_PROMPT,
                        "multiline": True,
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.3,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                    },
                ),
                "max_tokens": (
                    "INT",
                    {
                        "default": 2000,
                        "min": 1,
                        "max": 32768,
                        "step": 1,
                    },
                ),
                "clean_markdown": (
                    "BOOLEAN",
                    {"default": True},
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("caption",)
    FUNCTION = "caption"
    CATEGORY = "text"

    def caption(
        self,
        image,
        base_url,
        api_key,
        vl_model,
        prompt,
        temperature,
        max_tokens,
        clean_markdown,
    ):
        if not base_url.strip():
            raise ValueError("base_url 不能为空")
        if not vl_model.strip():
            raise ValueError("vl_model 不能为空")

        # 规范化 base_url
        url = base_url.strip().rstrip("/")
        if not url.endswith("/v1"):
            url += "/v1"

        # 图片转 base64
        pil_img = _tensor_to_pil(image)
        img_b64 = _pil_to_base64(pil_img, fmt="JPEG")
        img_data_url = f"data:image/jpeg;base64,{img_b64}"

        # 创建客户端
        client_kwargs = {"base_url": url}
        if api_key.strip():
            client_kwargs["api_key"] = api_key.strip()
        client = OpenAI(**client_kwargs)

        # 调用 VLM
        resp = client.chat.completions.create(
            model=vl_model.strip(),
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": img_data_url}},
                ],
            }],
            max_tokens=max_tokens,
            temperature=temperature,
        )

        caption = resp.choices[0].message.content or ""
        caption = caption.strip()

        if not caption:
            return ("(VLM returned an empty response)",)

        if clean_markdown:
            caption = _clean_markdown(caption)

        return (caption,)


# ============================================================
# 节点 2: VLMDenseCaptionBatch — 批量打标
# ============================================================

class VLMDenseCaptionBatch:
    """对批量图片逐张进行 VLM 稠密描述，用分隔符拼接输出"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "base_url": (
                    "STRING",
                    {
                        "default": "your_url",
                        "multiline": False,
                    },
                ),
                "api_key": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                    },
                ),
                "vl_model": (
                    "STRING",
                    {
                        "default": "Qwen2.5-VL-32B-Instruct-AWQ",
                        "multiline": False,
                    },
                ),
                "prompt": (
                    "STRING",
                    {
                        "default": DEFAULT_VLM_PROMPT,
                        "multiline": True,
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.3,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                    },
                ),
                "max_tokens": (
                    "INT",
                    {
                        "default": 2000,
                        "min": 1,
                        "max": 32768,
                        "step": 1,
                    },
                ),
                "clean_markdown": (
                    "BOOLEAN",
                    {"default": True},
                ),
                "separator": (
                    "STRING",
                    {
                        "default": "\n\n---\n\n",
                        "multiline": False,
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("captions",)
    FUNCTION = "caption_batch"
    CATEGORY = "text"

    def caption_batch(
        self,
        images,
        base_url,
        api_key,
        vl_model,
        prompt,
        temperature,
        max_tokens,
        clean_markdown,
        separator,
    ):
        if not base_url.strip():
            raise ValueError("base_url 不能为空")
        if not vl_model.strip():
            raise ValueError("vl_model 不能为空")

        url = base_url.strip().rstrip("/")
        if not url.endswith("/v1"):
            url += "/v1"

        client_kwargs = {"base_url": url}
        if api_key.strip():
            client_kwargs["api_key"] = api_key.strip()
        client = OpenAI(**client_kwargs)

        batch_size = images.shape[0]
        captions = []

        for i in range(batch_size):
            single = images[i : i + 1]
            pil_img = _tensor_to_pil(single)
            img_b64 = _pil_to_base64(pil_img, fmt="JPEG")
            img_data_url = f"data:image/jpeg;base64,{img_b64}"

            resp = client.chat.completions.create(
                model=vl_model.strip(),
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": img_data_url}},
                    ],
                }],
                max_tokens=max_tokens,
                temperature=temperature,
            )

            caption = resp.choices[0].message.content or ""
            caption = caption.strip()

            if not caption:
                caption = f"(Image {i + 1}: empty response)"

            if clean_markdown:
                caption = _clean_markdown(caption)

            captions.append(caption)

        return (separator.join(captions),)


# ============================================================
# Step 2: LLM 编辑指令生成 — 配置
# ============================================================

LLM_EDIT_SYSTEM_PROMPT = """你是一个顶尖的工业产品设计专家，专门负责鞋类产品的变体设计。
我会给你一段鞋类产品的密集文本描述（Source Caption）和指定的编辑类型。

请严格按照指定的编辑类型，执行以下任务：
1. 识别出产品的一个局部组件或材质属性，作为修改目标。
2. 提出一个符合美学和物理规律的修改建议（Instruction）。
3. 将修改点融合进原描述中，输出修改后的完整产品描述（Target Caption）。

编辑类型说明及示例：

- material_swap: 更换鞋面材质
  可选项：蛇皮革(snakeskin)、漆皮(patent leather)、皮毛一体(all in one skin)、反绒皮(suede leather)、油蜡皮(oil wax leather)、摔纹皮(welt leather)、粒面皮(grain leather)、擦色皮(brush skin)、网布(mesh cloth)、塑料(plastic)、草编(straw weave)、亚麻(hessian)、丝绒(velvet fabric)、牛仔布(denim)、弹力布(stretch cloth)、格利特(gleit)、蕾丝(lace mesh)、纱线织物(mesh)、帆布(canvas)、丝绸(silk fabric)、猪皮革(pigskin leather)、织物(fabric)、羊皮革(sheep hide)、人造革(artificial leather)、牛皮革(cow leather)、皮草(furs)
  示例: "Turn the breathable mesh upper into premium full-grain leather, keeping the same color scheme"

- color_change: 改变颜色
  可改变整体颜色或局部区域颜色（鞋面/鞋底/鞋带/鞋舌/中底/logo等）
  示例: "Change the vibrant blue and teal accents to rich burgundy and deep maroon tones"
  示例: "Transform the all-white upper into a soft blush pink, keeping the white midsole and outsole unchanged"

- component_mod: 修改组件
  修改鞋带系统（扁鞋带→圆鞋带、鞋带→魔术贴、弹性鞋口等）、鞋舌样式、鞋口造型、侧边支撑等
  示例: "Replace the traditional flat laces with an elastic speed-lacing system with a toggle closure"

- accessory_add: 添加/修改配饰
  可选项：蝴蝶结(bow)、钻饰(rhinestones)、五金饰扣(buckle)、珍珠(pearl)、流苏(tassel)、铆钉(rivet)、链条(chain)、丝带(banderole)、亮片(sequin)、星星(star)、花朵(flower)、金属(metal)
  示例: "Add a delicate pearl-embellished ankle strap with a small gold buckle accent"

- texture_upgrade: 材质纹理升级
  普通材质→特殊纹理（荔枝纹/鳄鱼压纹/编织纹/碳纤维纹理/拉丝金属等）
  示例: "Upgrade the smooth leather surface to an embossed crocodile skin texture while maintaining the same base color"

- style_shift: 风格微调
  运动→休闲/复古→未来感/正式→街头/极简→繁复/经典→潮流
  示例: "Shift the classic formal oxford style towards a modern streetwear aesthetic with slightly chunkier proportions"

- sole_change: 鞋底改造
  可选项：橡胶底(rubber sole)、发泡底(foam sole)、PVC、EVA、气垫底(air cushion sole)，也可改变鞋底厚度/纹路/颜色
  示例: "Replace the standard rubber outsole with a chunky EVA foam platform sole in contrasting white"

- pattern_add: 添加图案/花纹
  可选项：鳄鱼纹(crocodile print)、豹纹(leopard print)、斑马纹(zebra-stripe)、蛇纹(serpentine)、迷彩、扎染、渐变色、条纹、圆点
  示例: "Apply a subtle leopard print pattern across the entire upper in tonal brown shades"

- toe_mod: 鞋头类型修改
  可选项：尖头(pointed toe)、圆头(round toe)、方头(square toe)、杏仁头(almond toe)、露趾(open-toe)、鱼嘴(peep toe)
  示例: "Reshape the round toe into an elegant pointed toe silhouette while keeping all other design elements identical"

- heel_mod: 鞋跟类型修改
  可选项：低平跟(low flat heel)、中跟(mid-heel)、高跟(high heel)、坡跟(slipsole/wedge)、防水台(platform)、厚底(thick-soled)、酒杯跟(glass heel)、小猫跟(kitten heel)、锥形跟(tapered heel)、方跟(square heel)、粗高跟(chunky heel)、细高跟(stiletto)
  示例: "Transform the low flat heel into a refined kitten heel, approximately 5cm in height"

- craft_upgrade: 产品工艺升级
  可选项：压纹(embossing)、编织(knit)、刺绣(embroidery)、绗缝(quilted)、粗饰线(heavy stitching)、抽褶(shirring)、填海绵(padded sponge)
  示例: "Add intricate embroidery detailing along the heel counter and collar with tonal thread"

- style_mod: 款式修改（品类变换）
  将鞋子变换为另一种品类，但必须保留原鞋的设计风格和配色方案。修改后生成的是与原鞋不同品类的单只新鞋。
  可选项：拖鞋(slippers)、单鞋(shoes)、凉鞋(sandals)、靴子(boots)、运动鞋(sneakers)、浅口鞋(pumps)、前包后空鞋(slingback)、高帮鞋(high-top)、短靴(short boots)、中筒靴(mid-calf boot)、及膝长靴(knee boots)、弹力靴(stretch boots)、德比鞋(derby)、牛津鞋(oxford)、穆勒鞋(mules)、乐福鞋(loafers)、玛丽珍鞋(mary jane)、勃肯鞋(birkenstock)、老爹鞋(chunky sneakers)、小白鞋(little white shoes)、马丁靴(martens boots)、切尔西靴(chelsea boots)、骑士靴(riding boots)、帆布鞋(canvas shoes)、松糕鞋(platform shoes)、渔夫鞋(fisherman shoes/espadrilles)、沙滩鞋(beach shoes)、罗马鞋(calceus/gladiator sandals)、袜靴(sock boots)、工装靴(work boots)、布洛克鞋(brogue)、豆豆鞋(gommino/driving shoes)、毛毛鞋(fluffy shoes)、雪地靴(snow boots)、阿甘鞋(cortez shoes)、板鞋(skateboard shoes)
  示例: "Transform this lace-up sneaker into an elegant slip-on loafer, preserving the original cream leather color palette and minimalist design language"
  示例: "Convert this low-top canvas shoe into a mid-calf Chelsea boot with elastic side panels, keeping the same beige canvas material and clean aesthetic"

你必须以严格的 JSON 格式输出，不要包含任何其他文字：
```json
{
  "edit_type": "material_swap",
  "instruction": "Turn the breathable mesh upper into premium full-grain leather, keeping the same color scheme",
  "target_caption": "A running shoe with a white premium full-grain leather upper..."
}
```

注意：
- instruction 必须用英文，使用 "Turn...into..." / "Add..." / "Change..." / "Upgrade...to..." / "Shift...to..." / "Apply..." / "Reshape...into..." / "Transform...into..." / "Convert...to..." / "Replace...with..." 等主动句式
- target_caption 必须是完整的图片描述（融合修改后的），而不是只描述修改部分
- 每次只做一个修改，不要叠加多个修改
- 修改必须合理，保持在鞋类设计范畴内
- 对于 style_mod 类型：确保变换后的品类与原鞋品类明显不同，但保留原鞋的核心配色方案、材质风格和整体设计语言"""

EDIT_TYPES = [
    "material_swap",
    "color_change",
    "component_mod",
    "accessory_add",
    "texture_upgrade",
    "style_shift",
    "sole_change",
    "pattern_add",
    "toe_mod",
    "heel_mod",
    "craft_upgrade",
    "style_mod",
]

STYLE_MOD_TARGETS = [
    "slippers (拖鞋)",
    "pumps (浅口单鞋)",
    "sandals (凉鞋)",
    "slingback shoes (前包后空鞋)",
    "high-top sneakers (高帮运动鞋)",
    "skateboard shoes (板鞋)",
    "chunky sneakers / dad shoes (老爹鞋)",
    "canvas shoes (帆布鞋)",
    "cortez shoes (阿甘鞋)",
    "little white shoes (小白鞋)",
    "oxford shoes (牛津鞋)",
    "derby shoes (德比鞋)",
    "loafers (乐福鞋)",
    "mules (穆勒鞋)",
    "mary jane shoes (玛丽珍鞋)",
    "brogue shoes (布洛克鞋)",
    "gommino / driving shoes (豆豆鞋)",
    "birkenstock (勃肯鞋)",
    "espadrilles / fisherman shoes (渔夫鞋)",
    "beach shoes (沙滩鞋)",
    "gladiator sandals / calceus (罗马鞋)",
    "fluffy shoes (毛毛鞋)",
    "wedding shoes (婚鞋)",
    "ankle boots / short boots (短靴)",
    "mid-calf boots (中筒靴)",
    "martens boots (马丁靴)",
    "chelsea boots (切尔西靴)",
    "sock boots (袜靴)",
    "work boots (工装靴)",
    "snow boots (雪地靴)",
    "riding boots (骑士靴)",
    "stretch boots (弹力靴)",
    "platform shoes (松糕鞋)",
]


# ============================================================
# Step 2: JSON 解析与客户端工具
# ============================================================

def _get_client(base_url: str, api_key: str) -> OpenAI:
    """创建 OpenAI 兼容客户端"""
    url = base_url.strip().rstrip("/")
    if not url.endswith("/v1"):
        url += "/v1"
    kwargs = {"base_url": url}
    if api_key.strip():
        kwargs["api_key"] = api_key.strip()
    return OpenAI(**kwargs)


def _parse_json_response(raw: str) -> dict | None:
    """多策略从 LLM 输出中提取 JSON"""
    strategies = [
        lambda s: s.split("```json")[1].split("```")[0].strip() if "```json" in s else None,
        lambda s: s.split("```")[1].split("```")[0].strip() if s.count("```") >= 2 else None,
        lambda s: s,
    ]
    for strategy in strategies:
        try:
            text = strategy(raw)
            if text is None:
                continue
            text = text.strip()
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or start >= end:
                continue
            text = text[start : end + 1]
            result = json.loads(text)
            if "instruction" in result and "target_caption" in result:
                return {
                    "edit_type": result.get("edit_type", "unknown"),
                    "instruction": result.get("instruction", ""),
                    "target_caption": result.get("target_caption", ""),
                }
        except (json.JSONDecodeError, IndexError, KeyError):
            continue
    return None


def _fallback_edit(source_caption: str) -> dict:
    """兜底编辑"""
    return {
        "edit_type": "unknown",
        "instruction": "Make subtle design refinements to the shoe",
        "target_caption": source_caption,
    }


# ============================================================
# 节点 3: LLMEditInstruction — 单条编辑指令生成
# ============================================================

class LLMEditInstruction:
    """根据源图描述 + 编辑类型，调用 LLM 生成编辑指令和目标描述"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_caption": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "粘贴 VLM Dense Caption 输出的稠密描述...",
                    },
                ),
                "edit_type": (EDIT_TYPES, {"default": "material_swap"}),
                "base_url": (
                    "STRING",
                    {
                        "default": "your_url",
                        "multiline": False,
                    },
                ),
                "api_key": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                    },
                ),
                "llm_model": (
                    "STRING",
                    {
                        "default": "Qwen3-235B-A22B-Instruct-2507-AWQ",
                        "multiline": False,
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.7,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                    },
                ),
                "max_tokens": (
                    "INT",
                    {
                        "default": 1500,
                        "min": 1,
                        "max": 32768,
                        "step": 1,
                    },
                ),
                "max_retries": (
                    "INT",
                    {
                        "default": 2,
                        "min": 0,
                        "max": 5,
                        "step": 1,
                        "tooltip": "JSON 解析失败时的最大重试次数",
                    },
                ),
            },
            "optional": {
                "style_target": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "仅 style_mod 时生效，如 loafers (乐福鞋)",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("instruction", "target_caption")
    FUNCTION = "generate"
    CATEGORY = "text"

    def generate(
        self,
        source_caption,
        edit_type,
        base_url,
        api_key,
        llm_model,
        temperature,
        max_tokens,
        max_retries,
        style_target="",
    ):
        if not source_caption.strip():
            raise ValueError("source_caption 不能为空")
        if not base_url.strip():
            raise ValueError("base_url 不能为空")

        client = _get_client(base_url, api_key)

        # 构建用户 prompt
        style_hint = ""
        if edit_type == "style_mod" and style_target.strip():
            style_hint = (
                f"\n\n【重要约束】本次必须将鞋子品类变换为: {style_target.strip()}。"
                "instruction 和 target_caption 中必须体现该品类特征。"
            )
        user_prompt = f"请基于以下描述，进行 {edit_type} 类型的编辑：\n\n{source_caption}{style_hint}"

        for attempt in range(max_retries + 1):
            try:
                resp = client.chat.completions.create(
                    model=llm_model.strip(),
                    messages=[
                        {"role": "system", "content": LLM_EDIT_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                raw = resp.choices[0].message.content or ""
                raw = raw.strip()

                if not raw:
                    if attempt < max_retries:
                        continue
                    result = _fallback_edit(source_caption)
                    return (result["instruction"], result["target_caption"])

                result = _parse_json_response(raw)
                if result:
                    return (result["instruction"], result["target_caption"])

                if attempt < max_retries:
                    continue

            except Exception as e:
                if attempt < max_retries:
                    continue
                return (f"API Error: {str(e)}", source_caption)

        result = _fallback_edit(source_caption)
        return (result["instruction"], result["target_caption"])


# ============================================================
# 节点 4: LLMEditInstructionBatch — 批量生成多条编辑
# ============================================================

class LLMEditInstructionBatch:
    """对同一源图描述，批量生成多条不同类型的编辑指令"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_caption": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "粘贴 VLM Dense Caption 输出的稠密描述...",
                    },
                ),
                "base_url": (
                    "STRING",
                    {
                        "default": "your_url",
                        "multiline": False,
                    },
                ),
                "api_key": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                    },
                ),
                "llm_model": (
                    "STRING",
                    {
                        "default": "Qwen3-235B-A22B-Instruct-2507-AWQ",
                        "multiline": False,
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.7,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                    },
                ),
                "max_tokens": (
                    "INT",
                    {
                        "default": 1500,
                        "min": 1,
                        "max": 32768,
                        "step": 1,
                    },
                ),
                "max_retries": (
                    "INT",
                    {
                        "default": 2,
                        "min": 0,
                        "max": 5,
                        "step": 1,
                    },
                ),
                "num_edits": (
                    "INT",
                    {
                        "default": 3,
                        "min": 1,
                        "max": 12,
                        "step": 1,
                        "tooltip": "生成几条不同的编辑",
                    },
                ),
                "fixed_edit_types": (
                    "STRING",
                    {
                        "default": "style_mod",
                        "multiline": False,
                        "placeholder": "逗号分隔，如 style_mod,color_change。空 = 随机",
                        "tooltip": "必定包含的编辑类型（逗号分隔），其余随机抽取。留空则全部随机",
                    },
                ),
                "enable_style_mod": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "是否在随机抽取时包含 style_mod（品类变换）",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("all_instructions", "all_target_captions")
    FUNCTION = "generate_batch"
    CATEGORY = "text"

    def generate_batch(
        self,
        source_caption,
        base_url,
        api_key,
        llm_model,
        temperature,
        max_tokens,
        max_retries,
        num_edits,
        fixed_edit_types,
        enable_style_mod,
    ):
        if not source_caption.strip():
            raise ValueError("source_caption 不能为空")

        client = _get_client(base_url, api_key)

        # 解析固定类型
        fixed_list = []
        if fixed_edit_types.strip():
            fixed_list = [t.strip() for t in fixed_edit_types.split(",") if t.strip() in EDIT_TYPES]

        # 过滤可用类型池
        pool = [t for t in EDIT_TYPES if t not in fixed_list]
        if not enable_style_mod:
            pool = [t for t in pool if t != "style_mod"]
            fixed_list = [t for t in fixed_list if t != "style_mod"]

        if len(fixed_list) > num_edits:
            raise ValueError(f"固定编辑类型数量 ({len(fixed_list)}) 超过 num_edits ({num_edits})")

        # 构建最终类型列表
        chosen = list(fixed_list)
        remaining = num_edits - len(chosen)
        if remaining > 0 and pool:
            chosen.extend(random.sample(pool, min(remaining, len(pool))))
        random.shuffle(chosen)

        instructions = []
        target_captions = []

        for edit_type in chosen:
            style_hint = ""
            if edit_type == "style_mod":
                target = random.choice(STYLE_MOD_TARGETS)
                style_hint = (
                    f"\n\n【重要约束】本次必须将鞋子品类变换为: {target}。"
                    "instruction 和 target_caption 中必须体现该品类特征。"
                )

            user_prompt = f"请基于以下描述，进行 {edit_type} 类型的编辑：\n\n{source_caption}{style_hint}"

            result = None
            for attempt in range(max_retries + 1):
                try:
                    resp = client.chat.completions.create(
                        model=llm_model.strip(),
                        messages=[
                            {"role": "system", "content": LLM_EDIT_SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                        ],
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                    raw = resp.choices[0].message.content or ""
                    raw = raw.strip()

                    if raw:
                        result = _parse_json_response(raw)
                        if result:
                            break

                except Exception:
                    if attempt >= max_retries:
                        break

            if result is None:
                result = _fallback_edit(source_caption)

            instructions.append(f"[{edit_type}] {result['instruction']}")
            target_captions.append(f"[{edit_type}] {result['target_caption']}")

        return (
            "\n\n".join(instructions),
            "\n\n".join(target_captions),
        )


# ============================================================
# ComfyUI 注册
# ============================================================

NODE_CLASS_MAPPINGS = {
    "VLMDenseCaption": VLMDenseCaption,
    "VLMDenseCaptionBatch": VLMDenseCaptionBatch,
    "LLMEditInstruction": LLMEditInstruction,
    "LLMEditInstructionBatch": LLMEditInstructionBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VLMDenseCaption": "VLM Dense Caption",
    "VLMDenseCaptionBatch": "VLM Dense Caption (Batch)",
    "LLMEditInstruction": "LLM Edit Instruction",
    "LLMEditInstructionBatch": "LLM Edit Instruction (Batch)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
