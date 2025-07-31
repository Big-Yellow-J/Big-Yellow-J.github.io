# DF模型数据集构建
## image-to-text数据集构建
通过输入图片生成文本描述：`./image2text.py`。目前基础版本直接通过的是 blip去生成文本描述，并且通过多进程的方式，多个进程共享模型，不过文本描述上很短。

## instance_background数据集构建
通过YoLo+SAM算法识别instance而后进行分割，最后将实体贴到背景中
- [ ] 分割背景将图像贴到背景中