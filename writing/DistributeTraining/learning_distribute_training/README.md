# TODO
- [ ] （⭐⭐⭐）修改基于torch ddp等让其支持 ray 分布式寻找超参
- [ ] 修改 [torchDDP_training.py](./torchDDP_training.py) 让其支持 vllm/或者直接看trl中如何支持 vllm 引擎启动（最好是再去新建一个新的文件）
- [ ] 修改 [torchDDP_training.py](./torchDDP_training.py)，可以加入支持模型评估策略（或者说支持模型直接load训练好的权重然后让模型进行加载评估）