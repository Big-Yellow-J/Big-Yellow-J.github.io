import json
import torch
from torch.utils.data import IterableDataset, DataLoader

class IterableDatasetJsonl(IterableDataset):
  def __init__(self, file_path, file_type: str='jsonl', shard_rank=0, num_shards=1):
    self.file_path = file_path
    self.file_type = file_type
    self.shard_rank = shard_rank
    self.num_shards = num_shards

  def __iter__(self):
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
      shard_rank = self.shard_rank * worker_info.num_workers + worker_info.id
      num_shards = self.num_shards * worker_info.num_workers
    else:
      shard_rank, num_shards = self.shard_rank, self.num_shards

    with open(self.file_path, 'r', encoding='utf-8') as f:
      for i, line in enumerate(f):
        if i % num_shards == shard_rank:
          sample = json.loads(line)
          '''
          继续后处理
          '''
          yield sample

def collate_fn(batch):
  # 亦或者直接通过 zip(*batch) 进行解包（如果数据返回多个对象）
  images = [_['image'] for _ in batch]
  return images

if __name__ == '__main__':
  jsonl_dataset = IterableDatasetJsonl('./test_json.jsonl')
  jsonl_dataloader = DataLoader(jsonl_dataset, batch_size=3,
                                num_workers=8, pin_memory=True,
                                prefetch_factor=4, persistent_workers=True,
                                collate_fn= collate_fn)
  for i, sample in enumerate(jsonl_dataloader):
    if i == 0:
      print(sample)
    else:
      break