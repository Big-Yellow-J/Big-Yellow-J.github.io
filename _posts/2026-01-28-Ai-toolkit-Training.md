---
layout: mypost
title: 深入浅出了解生成模型-9：扩散模型微调框架ai-toolkit介绍
categories: 生成模型
extMath: true
images: true
address: changsha
show_footer_image: true
tags:
- 生成模型
- diffusion model
- 工具介绍
description: 本文介绍使用Ai-toolkit框架对扩散模型进行微调的方法，涵盖环境准备、数据集配置、训练参数设置及代码自定义参数。环境需autodl服务器（VGPU-32G，CUDA
  13.0），配置hf token后，数据集要求图像为.jpg/.jpeg/.png格式、文本为.txt格式且图文匹配，存放于指定路径。训练时模型路径建议默认，模型下载报错可重启任务。微调流程简单，修改数据后点击开始训练即可。代码支持自定义参数，包括model模块的assistant_lora_path，train模块的xformers加速、attention_backend计算方式，decorator模块的num_tokens（支持flux模型），以及adapter模块的t2i图像到图像条件适配功能。
---

本文主要介绍Ai-toolkit框架去对扩散模型进行微调操作
## Ai-toolkit
### Ai-toolkit安装介绍
#### 环境准备
在autodl上的服务器进行的操作（GPU：VGPU-32G，CUDA Version: 13.0 ）
```python
# 首先安装基本环境
source /etc/network_turbo  # autodl 上执行该命令进行代理
conda create -n ai-toolkit python=3.12
conda activate ai-toolkit
pip install --no-cache-dir torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu126

git clone https://github.com/ostris/ai-toolkit.git
cd ai-toolkit
pip3 install --no-cache-dir torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu126
pip3 install -r requirements.txt

# 安装 npm
curl -fsSL https://deb.nodesource.com/setup_lts.x | bash -
apt update
apt install -y nodejs

# 安装完毕之后直接测试，如果显示版本那么表示安装成功
node -v # v24.13.0
npm -v # 11.6.2
npm config set registry https://registry.npmmirror.com # 换npm源

# 由于启动了代理可以先使用下面代码之后再去执行 run 程序
npm config set strict-ssl false
export NODE_TLS_REJECT_UNAUTHORIZED=0

export HF_ENDPOINT=https://hf-mirror.com
# 建议取修改hf模型下载路径
vim ~/.bashrc
export HF_HOME="/path/to/you/dir"  # 替换为你想更改的目标路径
source ~/.bashrc

cd ai-toolkit/ui
export HF_ENDPOINT=https://hf-mirror.com
npm run build_and_start
```
上面运行代码运行之后出现：
![](https://s2.loli.net/2026/02/03/SYvOuAUjJtMZV4x.webp)
可以直接访问上面地址进入 ai-toolkit
#### 界面简单介绍
首先取配置自己hf token
![](https://s2.loli.net/2026/01/28/es4i58XBa3K7Oxr.webp)
配置完毕之后可以直接上传数据集/直接在本地数据集，不过数据集需要在路径：`xxx/ai-toolkit/datasets`（对应上面图像中的路径） 中除此之外还需要注意数据集的格式问题，以文生图任务为例，我的数据集必须满足：1、图像必须是：.jpg, .jpeg,  .png；2、文本：txt。除此之外图片文本之间必须匹配：1.png 1.txt.....
![](https://s2.loli.net/2026/02/03/marREeQHCq3bnyi.webp)
可以直接将上面文件夹上传到`xxx/ai-toolkit/datasets`中
![](https://s2.loli.net/2026/01/28/rxjsKY8hdpgV1BE.webp)
![](https://s2.loli.net/2026/01/28/VLBpKZX3JSti2IG.webp)
对于**训练界面参数**介绍：
![](https://s2.loli.net/2026/02/03/os42J5FXbDHVA1a.webp)
![](https://s2.loli.net/2026/02/03/weFAziYN1xkv3mp.webp)
1、模型路径尽量不要去修改就用默认的，如果要去修改可以参考：reddit上的方法
2、如果报错是和模型下载相关（如CAS报错、hf_transfer报错），可以直接重启任务就行（去Training Quene找到任务然后重新启动即可）
**模型训练**处理
![](https://s2.loli.net/2026/02/03/PnH8iG7k4tmhNBu.webp)
### Ai-toolkit模型微调
对于文生图/图生图微调训练很简单只需要将上面的数据进行修改即可，而后点击开始训练即可
![](https://s2.loli.net/2026/01/28/ydSs2CKgG8tD5T7.webp)
### Ai-toolkit代码分析
在Ai-toolkit中模型微调整个流程如下：[Googledrive-Drawio](https://drive.google.com/file/d/1X87iDyYk2ebtdrG5-_Q4qUvu67wwvEOs/view?usp=sharing)。值得注意的是除去ai-toolkit中前端默认参数还可以直接自定义参数（较多参数都在`ai-toolkit/toolkit/config_modules.py`文件中**给了默认参数**）：
* `model` 模块参数（直接去`BaseSDTrainProcess.py`看参数）

`assistant_lora_path:str`： lora模型路径（建议直接使用 hf地址）

* `train` 模块参数（直接去`BaseSDTrainProcess.py`看参数）

|参数名称 | 参数描述 | 注意事项 |
|:---:|:---:|:---:|
|`xformers:bool`                          | 是否启动xformer，直接去vae以及unet中启动xformer加速计算| 注意模型是不是支持xformer，代码位置 |
|`attention_backendattention_backend:str` | 后端attention计算方式，也是对vae/unet进行，比如说`flash`等| |
| `decorator_config`                      | 这部分参数配置和 `train` 中配置参数写的方式是一样的| |

* `decorator` 模块参数（暂时只支持 flux 模型）（默认没有，参数方法和 train 中使用方法相同）

`num_tokens:int` 文本嵌入修饰器/适配器，专门用于在扩散模型的文本条件输入上额外拼接几个可学习的 token

* `adapter` t2i模块参数（默认没有，参数方法和 train 中使用方法相同）

对于这个模块参数可以[参考配置](https://huggingface.co/spaces/rahul7star/ai-toolkit/blob/main/config/examples/train_flex_redux.yaml)。主要作用是实现图像到图像（image-to-image）条件适配器，主要功能是，让模型“看见”参考图像，然后根据文本提示 + 参考图像 来生成变体（variation）、风格迁移、细节保持等。
```yaml
adapter:
  train: false                # adapter是否参与训练  
  type: "ip_adapter"          # 支持："t2i", "control_net", "clip", "ip" 也支持直接自定义  
  name_or_path: "h94/IP-Adapter"      # 上面4种参数直接会从 hf 上进行加载，对于自定义的值就回去加载custom
  weight_name: "ip-adapter_sd15.safetensors"
  scale: 0.8                  # 强度缩放（可选，默认 1.0）
  test_img_path:
    - "path/to/your/image.png"
    - "path/to/your/image2.png"
```