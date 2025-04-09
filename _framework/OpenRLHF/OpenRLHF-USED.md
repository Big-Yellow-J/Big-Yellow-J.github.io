# 开源项目：OpenRLHF使用
> From：https://github.com/OpenRLHF/OpenRLHF#

一、优化函数处理
1、[actor](openrlhf/models/actor.py): 加载需要的LLM模型
支持lora、flash_atten等操作

## 二、数据处理
### 1.划分训练集测试集：[blending_datasets](openrlhf/utils/utils.py)
### 2.数据集操作：
[ProcessRewardDataset](openrlhf/datasets/process_reward_dataset.py)
[RewardDataset](openrlhf/datasets/reward_dataset.py)
[PromptDataset](openrlhf/datasets/prompts_dataset.py)
[SFTDataset](openrlhf/datasets/sft_dataset.py)
上面几种数据集处理类中，都会使用如下几个参数：
1、dataset；2、tokenizer；3、strategy
前面两个参数好理解就是数据集以及tokenizer，第3个参数一般指的是**策略模板**，通过策略模板来格式化输出，默认的就是直接使用[Hugging Face](https://huggingface.co/docs/transformers/main/en/chat_templating?template=Mistral)也可以直接自己指定



## DPO主要训练过程

启动脚本：[examples/scripts/train_dpo_llama.sh](examples/scripts/train_dpo_llama.sh)
训练器：[openrlhf/cli/train_dpo.py](openrlhf/cli/train_dpo.py)
文件：[openrlhf/cli/train_dpo.py](openrlhf/cli/train_dpo.py)过程分析
