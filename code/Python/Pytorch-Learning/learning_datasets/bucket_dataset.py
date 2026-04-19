import torch
import random
from collections import defaultdict
from torch.utils.data import DataLoader, Sampler, Dataset

class BucketBatchSampler(Sampler):
  def __init__(self, dataset, batch_size, drop_last=True, shuffle=True):
    self.dataset = dataset
    self.batch_size = batch_size
    self.drop_last = drop_last
    self.shuffle = shuffle

    self.buckets = defaultdict(list)
    for idx in range(len(dataset)):
      h, w = dataset.get_image_size(idx)
      self.buckets[(h, w)].append(idx)
    self.bucket_keys = list(self.buckets.keys())

  def __iter__(self):
    batches = []
    for key in self.bucket_keys:
      indices = self.buckets[key][:]
      if self.shuffle:
        random.shuffle(indices)

      for i in range(0, len(indices), self.batch_size):
        batch = indices[i:i + self.batch_size]
        if self.drop_last and len(batch) < self.batch_size:
          continue
        if batch:
          batches.append(batch)

    if self.shuffle:
      random.shuffle(batches)
    print(f"Sampler Get batches: {batches}")
    for batch in batches:
      yield batch

  def __len__(self):
    total = 0
    for indices in self.buckets.values():
      total += len(indices) // self.batch_size
      if not self.drop_last and len(indices) % self.batch_size != 0:
        total += 1
    return total

class DummyDataset(Dataset):
    def __init__(self, num_samples=100):
      self.num_samples = num_samples
      self.resolutions = [(512, 512), (512, 768), (768, 512)]
      self.data = []
      for i in range(num_samples):
        res = random.choice(self.resolutions)
        self.data.append({
          "image": torch.randn(3, res[0], res[1]),
          "res": res,
          "id": i
        })

    def __len__(self):
      return self.num_samples

    def __getitem__(self, idx):
      return self.data[idx]

    def get_image_size(self, idx):
      return self.data[idx]["res"]

def test_sampler():
  batch_size = 4
  dataset = DummyDataset(num_samples=50)
  sampler = BucketBatchSampler(dataset, batch_size=batch_size, drop_last=True)
  dataloader = DataLoader(dataset, batch_sampler=sampler)

  print(f"Dataset 总数: {len(dataset)}")
  print(f"预期的 Batch 数量 (drop_last=True): {len(sampler)}")
  print("-" * 30)

  for i, batch in enumerate(dataloader):
    print(batch.keys())
    resolutions_in_batch = [tuple(img.shape[1:]) for img in batch["image"]]
    res_set = set(resolutions_in_batch)

    print(f"Batch {i}: 包含索引 {batch['id'].tolist()}")
    print(f"  -> 分辨率: {res_set} | Batch 大小: {len(batch['id'])}")

if __name__ == "__main__":
  test_sampler()