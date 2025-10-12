---
layout: mypost
title: 模型的量化与部署————GPTQ和AWQ量化、ONNX和TensorRT部署
categories: 量化部署
address: 武汉🏯
extMath: true
show_footer_image: true
tags:
- 模型量化
description: 
---

## 模型量化技术
简单了解几个概念：
**量化**：是一种模型压缩的常见方法，将模型权重从高精度（如FP16或FP32）量化为低比特位（如INT8、INT4）。常见的量化策略可以分为PTQ和QAT两大类。
**量化感知训练**（Quantization-Aware Training）：在模型训练过程中进行量化，一般效果会更好一些，但需要额外训练数据和大量计算资源。
**后量化**（Post-Training Quantization, PTQ）：在模型训练完成后，对模型进行量化，无需重新训练。
### GPTQ量化技术
GPTQ[^1]是一种用于大型语言模型（LLM）的后训练量化技术。它通过将模型权重从高精度（如FP16或FP32）压缩到低比特（如3-4位整数）来减少模型大小和内存占用，同时保持较高的推理准确性。一般而言对于量化过程为：对于给定的权重矩阵$W\in R^{n\times m}$，**量化过程**就是需要找到一个低比特的矩阵$\hat{W}$使得：

$$
\min_{\hat{w}}\Vert WX-\hat{W}X\Vert^2_F
$$

其中$X$为输入向量，$\Vert. \Vert_F$为Frobenius范数。按照论文里面的描述GPTQ整个过程为：
![](https://s2.loli.net/2025/10/12/Qs5KqtHgBATcJdj.png)
对于具体数学原理的描述参考文章[^2][^3]，简单总结一下上面过程就是：1、每行独立计算二阶海森矩阵。2、每行按顺序进行逐个参数量化，从而可以并行计算。3、按block维度进行更新，对剩余参数进行延迟更新弥补。4、对逆海森矩阵使用cholesky分解，等价消除迭代中的矩阵更新计算。**它的核心流程其实就是量化-补偿-量化-补偿的迭代**（具体过程见流程图中内部循环：首先量化$W_{:,j}$，而后去计算误差并且补充到 $W_{:,j:(i+B)}$），具体的代码实现过程（[官方GPTQ-Github](https://github.com/IST-DASLab/gptq)）主要是对其中LlamaAttention和LlamaMLP层中的Linear层[权重进行量化](https://github.com/IST-DASLab/gptq/blob/2d65066eeb06a5c9ff5184d8cebdf33662c67faf/llama.py#L75C1-L84C1)。代码处理过程[^4]：
**首先**、计算Hessian矩阵（因为后续计算损失和补偿权重需要，因此提前计算矩阵）。实现方式是在每一层Layer上注册hook，通过hook的方式在layer forward后使用calibration data的input来生成Hessian矩阵，这种计算方式常见于量化流程中校准数据的处理
```python
def add_batch(name):
    def tmp(_, inp, out):
        gptq[name].add_batch(inp[0].data, out.data)
    return tmp
handles = []
for name in subset:
    handles.append(subset[name].register_forward_hook(add_batch(name)))
for j in range(args.nsamples):
    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
for h in handles:
    h.remove()
```


https://github.com/IST-DASLab/gptq
https://qwen.readthedocs.io/zh-cn/latest/quantization/gptq.html
### AWQ量化技术

## 模型部署技术
### ONNX模型部署
### TensorRT模型部署

## 参考
[^1]: [https://github.com/IST-DASLab/gptq](https://github.com/IST-DASLab/gptq)
[^2]: [https://zhuanlan.zhihu.com/p/646210009](https://zhuanlan.zhihu.com/p/646210009)
[^3]: [https://zhuanlan.zhihu.com/p/629517722](https://zhuanlan.zhihu.com/p/629517722)
[^4]:[https://zhuanlan.zhihu.com/p/697860995](https://zhuanlan.zhihu.com/p/697860995)