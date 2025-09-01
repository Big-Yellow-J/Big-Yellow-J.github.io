---
layout: mypost
title: 多模态模型——QwenVL2.5的微调以及强化学习代码操作
categories: 多模态
address: 武汉🏯
extMath: true
show_footer_image: true
tags:
- 多模态
- QwenVL
description: 从代码角度解析QwenVL2.5模型处理流程，包括模板化输入（通过processor.apply_chat_template处理对话messages，data_loader预处理）、编码（smart_resize动态调整图像分辨率，Qwen2VLImageProcessor完成归一化及patch切割）、模型处理（VisionTransformer处理pixel_values，经Conv3d与window-attention计算），并介绍其SFT和强化学习微调方法。
---

从代码角度去理解QwenVL2.5是如何处理，以及结合实际操作理解如何去对一个QwenVL2.5-3B进行SFT和强化学习处理，首先直接通过官方提供例子了解QwenVL是怎么使用的：
```python
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="auto"
)

min_pixels = 256*28*28
max_pixels = 1280*28*28
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to(model.device)

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
```

逐一了解一下QwenVL2.5模型的整个处理过程，模型整体过程大致为：1、首先是通过模板化处理我的模型的输入（image+text）；2、将输入转化为编码形式（比如文本tokenizer处理等）；3、出入模型处理输入然后模型输出；4、解码输出内容。整体主要是上述4个过程，因此下面逐一了解一下模型到底在做什么。
## QwenVL的基本使用
### 1、模板化模型输入
```python
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
```
所谓模板化模型的输入，很容易理解（通过`processor.apply_chat_template`把**对话 messages 转成模型能理解的 prompt**，不过值得注意的是不同模型可能处理的方式不同），就是将我的内容“填充”到模板中模拟对话内容，比如说上面处理得到的一个简单结果就是：
```python
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>
<|im_start|>assistant
```
一般在**data_loader里面就会提前将我们的模型需要的输入处理好**，比如说我们定义如下的模板
```python
def format_data(self, image, text, prompt):
    # self.SYSTEM_MESSAGE = """You are a helpful assistant."""
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": self.SYSTEM_MESSAGE}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": prompt
                },
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
        },
    ]
""" 
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>This is a prompt<|im_end|>
<|im_start|>assistant
This is a text<|im_end|>
<|im_start|>assistant
"""

```
对于上面内容输出理解，首先 `<|im_start|>....<|im_end|>`一般是一组“发言”的开始和结束标记，而后里面内容就是我们的文本/图像内容，`user`/ `assistant`/ `system` 则是分别代表：用户、模型、角色（告诉模型今天扮什么角色）。`<|vision_start|>...<|vision_end|>`：表示图像输入的占位符，告诉模型这里有一段视觉信息。`<|image_pad|>`：图像实际的 embedding 会在这里替换（填充），不是文字，而是图像编码后的向量。值得注意的是 `assistant`后面的内容就是 **模型需要输出的文本内容**。上面过程很容易理解，只不过需要注意如下问题，因为QwenVL2.5对于分辨率是存在处理（一般直接通过`smart_resize`处理，后续有介绍），因此如果涉及到目标识别，可能需要提前将坐标进行转换避免分辨率不同导致bbox对应不上的问题

### 2、编码模板输入
```python
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
```
编码模板输入就比较简单，因为我的输入都是文本/图片，此过程就是需要将这些内容转化为编码形式（比如tokenizer处理等），处理方式如下：
* 1、[process_vision_info](https://github.com/QwenLM/Qwen2.5-VL/blob/c15045f8829fee29d4b3996e068775fe6a5855db/qwen-vl-utils/src/qwen_vl_utils/vision_process.py#L352):返回我的图像/视频输出（都存储在list中）

首先是过[extract_vision_info](https://github.com/QwenLM/Qwen2.5-VL/blob/c15045f8829fee29d4b3996e068775fe6a5855db/qwen-vl-utils/src/qwen_vl_utils/vision_process.py#L334)从我上面的内容中提取出图片/视频（`[{'type': 'image', 'image': 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg'}]`）提取完毕之后就是交给处理图片/视频的函数进行处理
**图片处理过程**（[`fetch_image`](https://github.com/QwenLM/Qwen2.5-VL/blob/c15045f8829fee29d4b3996e068775fe6a5855db/qwen-vl-utils/src/qwen_vl_utils/vision_process.py#L97)）此过程也会比较简单，首先去判断类型（是`Image.Image`对象/图片链接等）然后打开图片，而后就是**确定图片分辨率尺寸**，有两种`smart_resize`处理方式，第一种是直接通过：`resized_height` 和 `resized_width`来确定改变，另外一种直接通过 `min_pixels` 和 `max_pixels` 来处理图像尺寸。对于`smart_rezie`函数处理过程为：
```python
def smart_resize(
    height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS)
    # IMAGE_FACTOR= 28
    if max(height, width) / min(height, width) > MAX_RATIO:
        ...
    h_bar = max(factor, round_by_factor(height, factor)) # round(number / factor) * factor
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor) # 按比例缩小并向下取整  math.floor(number / factor) * factor
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor) # 按比例放大并向上取整 math.ceil(number / factor) * factor
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar
```
上面3个小的子函数表示：计算factor倍数、向上取整计算倍数、向下取整计算倍数，对于smart_resize（去实现动态分辨率）函数：**通过四舍五入的方式，重新设置图片的 h 和 w 值，确保它们可以被28整除**，这样一来就得到了图像的需要修改的尺寸了，比如说：
输入: 一张 1000x500 的图像
计算基础尺寸：round(1000/28)=36, round(500/28)=18 → 1008x504
检查像素数：1008*504 = 508,032 > MAX_PIXELS(200,704)
计算缩放系数：beta = sqrt(1000*500/200704) ≈ 1.58
最终尺寸：floor(1000/1.58)=632, floor(500/1.58)=316 → 616x308（28的倍数）
**视频处理过程**（[fetch_video](https://github.com/QwenLM/Qwen2.5-VL/blob/c15045f8829fee29d4b3996e068775fe6a5855db/qwen-vl-utils/src/qwen_vl_utils/vision_process.py#L277)）对于视频处理和图像处理相类似打开-->改变尺寸。只不过在打开过程中QwenLV2.5处理过程为：
```python
def fetch_video(ele: dict, image_factor: int = IMAGE_FACTOR, return_video_sample_fps: bool = False):
    if isinstance(ele["video"], str):
        video_reader_backend = get_video_reader_backend()
        try:
            video, sample_fps = VIDEO_READER_BACKENDS[video_reader_backend](ele)
        except Exception as e:
            logger.warning(f"video_reader_backend {video_reader_backend} error, use torchvision as default, msg: {e}")
            video, sample_fps = VIDEO_READER_BACKENDS["torchvision"](ele)
    ...
```
对于`VIDEO_READER_BACKENDS`设计了3中不同范式：1、[_read_video_decord](https://github.com/QwenLM/Qwen2.5-VL/blob/c15045f8829fee29d4b3996e068775fe6a5855db/qwen-vl-utils/src/qwen_vl_utils/vision_process.py#L226)；2、[_read_video_torchvision](https://github.com/QwenLM/Qwen2.5-VL/blob/c15045f8829fee29d4b3996e068775fe6a5855db/qwen-vl-utils/src/qwen_vl_utils/vision_process.py#L183)；3、_read_video_torchcodec。
* `_read_video_decord`

```python
def _read_video_decord(
    ele: dict,
) -> (torch.Tensor, float):
    """read video using decord.VideoReader

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).
    """
    import decord
    video_path = ele["video"]
    st = time.time()
    vr = decord.VideoReader(video_path)
    total_frames, video_fps = len(vr), vr.get_avg_fps()
    start_frame, end_frame, total_frames = calculate_video_frame_range(
        ele,
        total_frames,
        video_fps,
    ) # 得到视频的开始 结束 总结多少帧
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    idx = torch.linspace(start_frame, end_frame, nframes).round().long().tolist()
    video = vr.get_batch(idx).asnumpy()
    video = torch.tensor(video).permute(0, 3, 1, 2)  # Convert to TCHW format
    ...
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps
    return video, sample_fps
```
对于其中的 `calculate_video_frame_range`函数处理过程也很简单（直接去计算视频开始、结束、总共多少帧），而后类似动态分辨率（smart_resize中成立相类似的）对于视频会通过智能视频帧数计算算法（smart_nframes），用于**确定从视频中提取多少帧作为模型输入**，处理过程为：第一种直接通过`round_by_factor(ele["nframes"], FRAME_FACTOR)`来得到帧数；第二种处理方式为（FPS_MIN_FRAMES = 4、FRAME_FACTOR = 2、FPS_MAX_FRAMES = 768、FPS = 2.0）：
```python
fps = ele.get("fps", FPS)
min_frames = ceil_by_factor(ele.get("min_frames", FPS_MIN_FRAMES), FRAME_FACTOR)
max_frames = floor_by_factor(ele.get("max_frames", min(FPS_MAX_FRAMES, total_frames)), FRAME_FACTOR)
nframes = total_frames / video_fps * fps
nframes = min(min(max(nframes, min_frames), max_frames), total_frames)
nframes = floor_by_factor(nframes, FRAME_FACTOR)

"""
config = {"nframes": 24}
result = smart_nframes(config, total_frames=100, video_fps=30)
# 输出：24（直接使用配置值）

config = {"fps": 10, "min_frames": 16, "max_frames": 32}
result = smart_nframes(config, total_frames=100, video_fps=30)
# 计算：100/30*10 ≈ 33.33 → 约束到32 → 对齐到32（FRAME_FACTOR=8的倍数）
"""
```
* 2、[processor](https://github.com/huggingface/transformers/blob/41925e42135257361b7f02aa20e3bbdab3f7b923/src/transformers/models/qwen2_5_vl/processing_qwen2_5_vl.py#L48)：去将图片/文本进行编码

其中对于文本编码直接通过 `self.tokenizer` 来处理，而对于图像直接通过 `self.image_processor`来处理。首先在 [代码](https://github.com/huggingface/transformers/blob/41925e42135257361b7f02aa20e3bbdab3f7b923/src/transformers/models/qwen2_5_vl/processing_qwen2_5_vl.py#L48)中很容易看到使用的图像/文本处理方式`image_processor_class = "AutoImageProcessor"` 对于文本处理方式 `tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")`。
对于**图片处理方式**的 `Qwen2VLImageProcessor`（[代码](https://github.com/huggingface/transformers/blob/41925e42135257361b7f02aa20e3bbdab3f7b923/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py#L87)）的处理思路：
```python
class Qwen2VLImageProcessor(BaseImageProcessor):
    def __init(...):
        ...
    def _preprocess(self, images, ...):
        ...
        height, width = get_image_size(images[0], channel_dim=input_data_format)
        resized_height, resized_width = height, width
        processed_images = []
        # Step-1
        for image in images:
            if do_resize:
                resized_height, resized_width = smart_resize(
                    height,
                    width,
                    factor=self.patch_size * self.merge_size,
                    min_pixels=self.min_pixels,
                    max_pixels=self.max_pixels,
                )
                image = resize(
                    image, size=(resized_height, resized_width), resample=resample, input_data_format=input_data_format
                )
            if do_rescale:
                image = self.rescale(image,...)
            if do_normalize:
                image = self.normalize(image,...)
        # Step-2
        patches = np.array(processed_images)
        if data_format == ChannelDimension.LAST:
            patches = patches.transpose(0, 3, 1, 2)
        if patches.shape[0] % self.temporal_patch_size != 0:
            # 视频补帧处理
            repeats = np.repeat(patches[-1][np.newaxis], self.temporal_patch_size - 1, axis=0)
            patches = np.concatenate([patches, repeats], axis=0)
        # 计算不同 patch 网格大小
        channel = patches.shape[1]
        grid_t = patches.shape[0] // self.temporal_patch_size
        grid_h, grid_w = resized_height // self.patch_size, resized_width // self.patch_size

        patches = patches.reshape(
            grid_t,
            self.temporal_patch_size,
            channel,
            grid_h // self.merge_size,
            self.merge_size,
            self.patch_size,
            grid_w // self.merge_size,
            self.merge_size,
            self.patch_size,
        )
        patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
        flatten_patches = patches.reshape(
            grid_t * grid_h * grid_w, channel * self.temporal_patch_size * self.patch_size * self.patch_size
        )

        return flatten_patches, (grid_t, grid_h, grid_w)
```
对于上面处理过程中，**首先**对于 `_preprocess`主要是对图像进行一些预处理：1、do_resize：改变图片大小（直接通过`smrt_resize`进行处理）2、do_rescale：像素缩减到0-1之间；3、do_normalize：对图片进行归一化处理（通道维度）；**而后**直接对于预处理后的图像直接进行切割处理为不同的patch输入到Vit中，比如假设我们的 `patches=(1,3,1024,1024)` 那么首先计算不同 patch网格大小（temporal_patch_size=2，patch_size=16，merge_size=2，resized_height=1024 值得注意的是temporal_patch_size=2会将图片处理为 2 3 1024 1024）那么计算得到网络patch大小为（grid_h= grid_w= 64）：1x64x64，而后分别将不同维度信息（t h w）进行划分，也就是将 （2，3，1024，1024）-->（1，2，3，32（64//2），2，16，32（64//2），2，16）最后再去交换维度并且进行合并即可。
**回顾一下QwenVL2.5的图片处理过程**：首先是去对图片进行改变尺寸（保证图片最后可以整除patch_size）/缩放/归一化。而后就是直接将图片处理为vit能够处理的“序列输入”得到的维度为：`[grid_t * grid_h * grid_w, channel * temporal_patch_size(2) * patch_size(14) * patch_size(14)]`。
> **补充一**：图片输入具体例子说明
> 假设默认参数为：patch_size= 14, temporal_patch_size= 2, merge_size= 2
> 图像输入为（通过process_vision_info提前处理之后的维度）：(1092, 1568) 
> 首先计算 `resized_height, resized_width = smart_resize`得到 812 1176
> 首先计算：grid_t=1，grit_h=812//14=58，grid_w=1176//14=84那么计算得到为 4872另外一项为 1176也就是最后图像处理得到的输出为：`(1*58*84, 14*14*2*3)=(4872,1176)`
> **补充二**：对于 smart_resize快速估算最后大小：
> 先 round 到 factor 的倍数
> 如果超出 max_pixels → 除以 sqrt(HW/max_pixels)，floor → factor 倍数
> 如果小于 min_pixels → 乘以 sqrt(min_pixels/HW)，ceil → factor 倍数
> 其实也就是：**首先将图像处理到为factor倍数的分辨率，而后去判断和max_pixels和min_pixels之间大小，大于前者就缩小，小于前者就放大**

最后通过一系列编码之后得到输出：
```python
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
"""
input_ids: torch.Size([1, 1243])
attention_mask: torch.Size([1, 1243])
pixel_values: torch.Size([4872, 1176])
image_grid_thw: torch.Size([1, 3])
"""
```

### 3、模型输入处理
```python
generated_ids = model.generate(**inputs, max_new_tokens=128)
```
整体[模型](https://github.com/huggingface/transformers/blob/41925e42135257361b7f02aa20e3bbdab3f7b923/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L1724)输入处理，输入模型也就是上面编码模板输入几个部分，只不过主要就是如下几个处理：首先是模型处理输入 `input_ids` 以及我的图像 `pixel_values`（`inputs_embeds = self.model.embed_tokens(input_ids)` [代码](https://github.com/huggingface/transformers/blob/41925e42135257361b7f02aa20e3bbdab3f7b923/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L1790)），而后将输入进行位置编码处理（[代码](https://github.com/huggingface/transformers/blob/41925e42135257361b7f02aa20e3bbdab3f7b923/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L1838)），最后输出模型结果（[代码](https://github.com/huggingface/transformers/blob/41925e42135257361b7f02aa20e3bbdab3f7b923/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L1861)），对于QwenVL2.5完整模型结构：
```python
Qwen2_5_VLForConditionalGeneration(
  (model): Qwen2_5_VLModel(
    (visual): Qwen2_5_VisionTransformerPretrainedModel(
      (patch_embed): Qwen2_5_VisionPatchEmbed(
        (proj): Conv3d(3, 1280, kernel_size=(2, 14, 14), stride=(2, 14, 14), bias=False)
      )
      (rotary_pos_emb): Qwen2_5_VisionRotaryEmbedding()
      (blocks): ModuleList(
        (0-31): 32 x Qwen2_5_VLVisionBlock(
          (norm1): Qwen2RMSNorm((1280,), eps=1e-06)
          (norm2): Qwen2RMSNorm((1280,), eps=1e-06)
          (attn): Qwen2_5_VLVisionAttention(
            (qkv): Linear(in_features=1280, out_features=3840, bias=True)
            (proj): Linear(in_features=1280, out_features=1280, bias=True)
          )
          (mlp): Qwen2_5_VLMLP(
            (gate_proj): Linear(in_features=1280, out_features=3420, bias=True)
            (up_proj): Linear(in_features=1280, out_features=3420, bias=True)
            (down_proj): Linear(in_features=3420, out_features=1280, bias=True)
            (act_fn): SiLU()
          )
        )
      )
      (merger): Qwen2_5_VLPatchMerger(
        (ln_q): Qwen2RMSNorm((1280,), eps=1e-06)
        (mlp): Sequential(
          (0): Linear(in_features=5120, out_features=5120, bias=True)
          (1): GELU(approximate='none')
          (2): Linear(in_features=5120, out_features=2048, bias=True)
        )
      )
    )
    (language_model): Qwen2_5_VLTextModel(
      (embed_tokens): Embedding(151936, 2048)
      (layers): ModuleList(
        (0-35): 36 x Qwen2_5_VLDecoderLayer(
          (self_attn): Qwen2_5_VLAttention(
            (q_proj): Linear(in_features=2048, out_features=2048, bias=True)
            (k_proj): Linear(in_features=2048, out_features=256, bias=True)
            (v_proj): Linear(in_features=2048, out_features=256, bias=True)
            (o_proj): Linear(in_features=2048, out_features=2048, bias=False)
            (rotary_emb): Qwen2_5_VLRotaryEmbedding()
          )
          (mlp): Qwen2MLP(
            (gate_proj): Linear(in_features=2048, out_features=11008, bias=False)
            (up_proj): Linear(in_features=2048, out_features=11008, bias=False)
            (down_proj): Linear(in_features=11008, out_features=2048, bias=False)
            (act_fn): SiLU()
          )
          (input_layernorm): Qwen2RMSNorm((2048,), eps=1e-06)
          (post_attention_layernorm): Qwen2RMSNorm((2048,), eps=1e-06)
        )
      )
      (norm): Qwen2RMSNorm((2048,), eps=1e-06)
      (rotary_emb): Qwen2_5_VLRotaryEmbedding()
    )
  )
  (lm_head): Linear(in_features=2048, out_features=151936, bias=False)
)
```
* **首先**：对于视觉部分处理（`Qwen2_5_VisionTransformerPretrainedModel`）

> 对于视觉模型主要需要处理的就是 `pixel_values`，假设输入的 `pixel_values`信息为：`[4872, 1176]`，image_grid_thw为： [1, 84, 58]（就是对应grid_t、grid_h、grid_w这三个数值）

主要包括如下几个模块：
1、[Qwen2_5_VisionPatchEmbed](https://github.com/huggingface/transformers/blob/41925e42135257361b7f02aa20e3bbdab3f7b923/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L88)：主要进行处理通过一个 `Conv3d`处理，处理[过程](https://github.com/huggingface/transformers/blob/41925e42135257361b7f02aa20e3bbdab3f7b923/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L105C4-L111C29)也就是说首先将输入的维度进行修改得到：`view(-1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size)` --> (4872,1176)-->(4872,3,2,14,14)而后再去通过卷积处理得到 (4872,1280,1,1,1)最后得到：**(4872,1280)**，也就对应着：`(grid_t*grid_h*grid_w, hiddend_size)`；
2、Qwen2_5_VisionRotaryEmbedding；
3、[Qwen2_5_VLVisionAttention](https://github.com/huggingface/transformers/blob/41925e42135257361b7f02aa20e3bbdab3f7b923/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L233)：首先去划分[window_size](https://github.com/huggingface/transformers/blob/41925e42135257361b7f02aa20e3bbdab3f7b923/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L465)这一步直接根据计算得到的：`[grid_t, grid_h, grid_w]`去划分windows，比如说在上述例子中，得到的cu_seqlens = [0,64,128,...,4872]，而后再去通过如下处理：
```python
lengths = cu_seqlens[1:] - cu_seqlens[:-1]
splits = [
    torch.split(tensor, lengths.tolist(), dim=2) for tensor in (query_states, key_states, value_states)
]
```
去划分q、k、v（形状都为：[1, 16, 4872, 80]）然后计算注意力，而后通过[Qwen2_5_VLPatchMerger](https://github.com/huggingface/transformers/blob/41925e42135257361b7f02aa20e3bbdab3f7b923/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L146)将结果合并起来。
**具体计算过程**，首先是如何得到cu_seqlens，因为我们得到的gird_thw=(1, 84, 58)也就是说总共有84*58=4872个token去计算全局注意力，那么这就会导致计算注意力的消耗过大，因此可以先去切分成小的window然后小块内部注意力计算。因此首先计算“块”的大小：`vit_merger_window_size = self.window_size // self.spatial_merge_size // self.patch_size`得到结果为: 4（112/2/14）也就是说每块大小为：4x4=16，但是不一定我的grid_h和grid_w可能整除4，因此就需要去计算填充数量 `vit_merger_window_size - llm_grid_h % vit_merger_window_size` 分别得到 4和2因此填充后的h和w为：88,60这样一来计算得到window数量为：88//4 * 60//4=330每个窗口的tokens数量：16

### 4、图像处理过程总结
**总结上述图像处理过程**：对于任意输入图像首先通过smart_resize（首先将图像改变到 factor的倍数，然后去判断和min_pixels和max_pixels之间大小，然后进行扩大，缩小）进行处理保证都可以整除patch_size（14）然后丢到 `processor`中进行处理主要是对图像归一化、正则化、改变维度（还会通过smart_resize在处理一次），处理之后再去确定他的 `grid_t, grid_h, grid_w`（对于这3个参数确定：直接通过 第二次smart_resize处理之后的结果除 patch_size即可）也就是tokens数量，而后将图像内容通过 conv3d处理得到：`(grid_t* grid_h* grid_w, hidden_size)`，最后就是计算window_attention（首先确定widow_size索引，通过索引进行切分，最后计算注意力）
> 补充：对于window-attention可以用卷积的思路去理解，比如说我得到“图像”：`(grid_t, grid_h, grid_w)` 我提前计算我的“卷积核”大小（`vit_merger_window_size = self.window_size // self.spatial_merge_size // self.patch_size`）为了保证我的 “图像”可以被卷积核处理就需要做一部分填充，而后用这个“卷积核”去划分成不同“小块”在到这个小块里面计算注意力。

### 5、位置编码

## QwenVL的微调过程
### SFT 处理
https://www.f22labs.com/blogs/complete-guide-to-fine-tuning-qwen2-5-vl-model/

### DL 处理