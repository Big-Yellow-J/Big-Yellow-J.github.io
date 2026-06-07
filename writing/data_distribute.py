import argparse
import os
import random
import shutil
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
try:
    from accelerate import Accelerator
except Exception:
    Accelerator = None
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms


class IndexedNoisyDataset(Dataset):
    def __init__(self, base_dataset: Dataset, indices: List[int], noisy_targets: List[int]):
        self.base_dataset = base_dataset
        self.indices = indices
        self.noisy_targets = noisy_targets

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        real_idx = self.indices[idx]
        image, _ = self.base_dataset[real_idx]
        label = int(self.noisy_targets[real_idx])
        return image, label


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor = None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss = alpha_t * loss

        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()


class PriorAdjustedSparseCELoss(nn.Module):
    """等价于 Keras sparse_categorical_crossentropy_with_prior。"""

    def __init__(self, class_priors: torch.Tensor, tau: float = 1.0):
        super().__init__()
        self.tau = tau
        self.register_buffer("log_prior", torch.log(class_priors.clamp_min(1e-8)))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        adjusted_logits = logits + self.tau * self.log_prior
        return F.cross_entropy(adjusted_logits, targets)


class PriorAdjustedCategoricalCELoss(nn.Module):
    """等价于 Keras categorical_crossentropy_with_prior (y_true one-hot)。"""

    def __init__(self, class_priors: torch.Tensor, tau: float = 1.0):
        super().__init__()
        self.tau = tau
        self.register_buffer("log_prior", torch.log(class_priors.clamp_min(1e-8)))

    def forward(self, logits: torch.Tensor, targets_one_hot: torch.Tensor) -> torch.Tensor:
        adjusted_logits = logits + self.tau * self.log_prior
        log_prob = F.log_softmax(adjusted_logits, dim=1)
        loss = -(targets_one_hot * log_prob).sum(dim=1)
        return loss.mean()


@dataclass
class EvalResult:
    overall: Dict[str, float]
    per_class: Dict[str, Dict[str, float]]
    confusion_matrix: List[List[int]]


def _ascii_table(headers: List[str], rows: List[List[str]]) -> str:
    headers = [str(h) for h in headers]
    norm_rows = [[str(cell) for cell in row] for row in rows]
    widths = [len(h) for h in headers]
    for row in norm_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def _line(sep: str = "-") -> str:
        return "+" + "+".join(sep * (w + 2) for w in widths) + "+"

    def _row(cells: List[str]) -> str:
        return "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells)) + " |"

    parts = [_line("-"), _row(headers), _line("=")]
    parts.extend(_row(r) for r in norm_rows)
    parts.append(_line("-"))
    return "\n".join(parts)


def _kv_table(title: str, mapping: Dict[str, object]) -> str:
    rows = [[k, str(v)] for k, v in mapping.items()]
    return f"[{title}]\n" + _ascii_table(["Field", "Value"], rows)


class DistributeTrainer:
    def __init__(
        self,
        data_name: str = "cifar10",
        model_name: str = "resnet50",
        random_seed: int = 20260607,
        error_factor: float = 0.0,
        imbalance_factor: float = 0.01,
        sampling_strategy: str = "none",
        loss_name: str = "ce",
        focal_gamma: float = 2.0,
        prior_tau: float = 1.0,
        image_size: Optional[int] = None,
        use_accelerate: bool = True,
        mixed_precision: str = "no",
        eval_logit_adjustment: bool = False,
        eval_prior_tau: Optional[float] = None,
        best_metric: str = "macro_f1",
        early_stop_patience: int = 0,
        early_stop_min_delta: float = 0.0,
        batch_size: int = 128,
        lr: float = 1e-4,
        num_workers: int = 4,
        data_root: str = "/home/huangjie/.cache",
        best_ckpt_path: str = "./best_model.pth",
        result_txt_path: str = "./train_result.txt",
    ):
        self.data_name = data_name.lower()
        self.model_name = model_name.lower()
        self.random_seed = random_seed
        self.error_factor = error_factor
        self.imbalance_factor = imbalance_factor
        self.sampling_strategy = sampling_strategy.lower()
        self.loss_name = loss_name.lower()
        self.focal_gamma = focal_gamma
        self.prior_tau = prior_tau
        self.image_size = image_size if image_size is not None else self._default_image_size_for_model(self.model_name)
        self.use_accelerate = use_accelerate
        self.mixed_precision = mixed_precision
        self.eval_logit_adjustment = eval_logit_adjustment
        self.eval_prior_tau = prior_tau if eval_prior_tau is None else eval_prior_tau
        self.best_metric = best_metric
        self.early_stop_patience = early_stop_patience
        self.early_stop_min_delta = early_stop_min_delta
        self.batch_size = batch_size

        self.lr = lr
        self.num_workers = num_workers
        self.data_root = data_root
        self.best_ckpt_path = best_ckpt_path
        self.result_txt_path = result_txt_path

        self.num_classes = 10
        self.class_names = [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]

        self.label_error_pairs = {
            3: 5,
            5: 3,
            1: 9,
            9: 1,
            4: 7,
            7: 4,
            2: 0,
            0: 2,
        }

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.accelerator = None
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.eval_log_prior = None

        self._set_seed(self.random_seed)

    def _setup_accelerator(self) -> None:
        if not self.use_accelerate:
            return
        if self.accelerator is not None:
            return
        if Accelerator is None:
            raise RuntimeError("未安装 accelerate，请先 `pip install accelerate`")
        self.accelerator = Accelerator(mixed_precision=self.mixed_precision)
        self.device = self.accelerator.device

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _default_image_size_for_model(model_name: str) -> int:
        name = model_name.lower()
        if name in {"vit-b/16", "vit_b_16", "vitb16", "vit-b16"}:
            return 224
        return 224

    def _build_imbalanced_indices(self, targets: List[int], mu: float) -> Tuple[List[int], List[int]]:
        if not (0 < mu <= 1.0):
            raise ValueError("imbalance_factor (mu) 必须在 (0, 1] 区间内")

        class_to_indices = {i: [] for i in range(self.num_classes)}
        for idx, label in enumerate(targets):
            class_to_indices[int(label)].append(idx)

        for i in range(self.num_classes):
            random.shuffle(class_to_indices[i])

        c_minus_1 = self.num_classes - 1
        n_max = max(len(v) for v in class_to_indices.values())

        num_per_class = []
        selected_indices = []
        for i in range(self.num_classes):
            n_i = int(n_max * (mu ** (i / c_minus_1)))
            n_i = max(1, min(n_i, len(class_to_indices[i])))
            num_per_class.append(n_i)
            selected_indices.extend(class_to_indices[i][:n_i])

        random.shuffle(selected_indices)
        return selected_indices, num_per_class

    def _inject_label_error(self, clean_targets: List[int], selected_indices: List[int]) -> List[int]:
        if not (0.0 <= self.error_factor <= 1.0):
            raise ValueError("error_factor 必须在 [0, 1] 区间内")

        noisy_targets = list(clean_targets)
        if self.error_factor == 0:
            return noisy_targets

        for idx in selected_indices:
            y = int(noisy_targets[idx])
            if y in self.label_error_pairs and random.random() < self.error_factor:
                noisy_targets[idx] = self.label_error_pairs[y]
        return noisy_targets

    def _resample_indices(self, indices: List[int], noisy_targets: List[int], strategy: str) -> List[int]:
        strategy = strategy.lower()
        if strategy == "none":
            return indices

        class_to_indices = {i: [] for i in range(self.num_classes)}
        for idx in indices:
            class_to_indices[int(noisy_targets[idx])].append(idx)

        for i in range(self.num_classes):
            if len(class_to_indices[i]) == 0:
                raise ValueError(f"类别 {i} 在当前分布下为空，无法执行采样策略 {strategy}")

        new_indices: List[int] = []
        if strategy == "undersample":
            target_n = min(len(v) for v in class_to_indices.values())
            for i in range(self.num_classes):
                random.shuffle(class_to_indices[i])
                new_indices.extend(class_to_indices[i][:target_n])
        elif strategy in {"oversample", "resample"}:
            target_n = max(len(v) for v in class_to_indices.values())
            for i in range(self.num_classes):
                source = class_to_indices[i]
                new_indices.extend(source)
                if len(source) < target_n:
                    extra = [random.choice(source) for _ in range(target_n - len(source))]
                    new_indices.extend(extra)
        else:
            raise ValueError("sampling_strategy 仅支持: none, undersample, oversample, compare")

        random.shuffle(new_indices)
        return new_indices

    def _count_labels(self, labels: List[int]) -> List[int]:
        counts = [0] * self.num_classes
        for y in labels:
            counts[int(y)] += 1
        return counts

    @staticmethod
    def _to_ratio(counts: List[int]) -> List[float]:
        total = float(sum(counts))
        if total <= 0:
            return [0.0 for _ in counts]
        return [float(c) / total for c in counts]

    def _build_class_weights(self, class_counts: List[int]) -> torch.Tensor:
        count_tensor = torch.tensor(class_counts, dtype=torch.float32)
        inv = 1.0 / count_tensor.clamp_min(1.0)
        weights = inv / inv.mean().clamp_min(1e-12)
        return weights.to(self.device)

    def load_model(self, pretrained=False) -> nn.Module:
        if self.model_name == "resnet50":
            weights = (models.ResNet50_Weights.DEFAULT if pretrained else None)
            model = models.resnet50(weights=weights)

            model.fc = nn.Linear(model.fc.in_features,self.num_classes)
        elif self.model_name in {
            "vit-b/16", "vit_b_16", "vitb16", "vit-b16"
        }:
            weights = (models.ViT_B_16_Weights.DEFAULT if pretrained else None)
            model = models.vit_b_16(weights=weights)

            model.heads.head = nn.Linear(model.heads.head.in_features,self.num_classes)

        self.model = model.to(self.device)
        return self.model

    def load_data(self, strategy: str = None) -> Tuple[DataLoader, DataLoader, Dict[str, List[int]]]:
        if self.data_name not in {"cifar10", "cifra10"}:
            raise ValueError("data_name 仅支持: cifar10")

        strategy = (strategy or self.sampling_strategy).lower()

        train_transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        )

        train_set = datasets.CIFAR10(root=self.data_root, train=True, transform=train_transform, download=True)
        test_set = datasets.CIFAR10(root=self.data_root, train=False, transform=test_transform, download=True)

        clean_targets = list(train_set.targets)
        selected_indices, imbalance_counts = self._build_imbalanced_indices(clean_targets, self.imbalance_factor)
        noisy_targets = self._inject_label_error(clean_targets, selected_indices)
        selected_indices = self._resample_indices(selected_indices, noisy_targets, strategy)

        raw_train_counts = self._count_labels(clean_targets)
        train_labels = [noisy_targets[i] for i in selected_indices]
        train_counts = self._count_labels(train_labels)

        train_dataset = IndexedNoisyDataset(train_set, selected_indices, noisy_targets)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        stats = {
            "raw_train_counts": raw_train_counts,
            "imbalance_counts": imbalance_counts,
            "train_counts_after_sampling_and_noise": train_counts,
        }
        return train_loader, test_loader, stats

    def loss_function(self, class_counts: List[int]) -> nn.Module:
        priors = torch.tensor(class_counts, dtype=torch.float32)
        priors = (priors / priors.sum().clamp_min(1.0)).to(self.device)
        class_weights = self._build_class_weights(class_counts)

        if self.loss_name == "ce":
            criterion = nn.CrossEntropyLoss()
        elif self.loss_name in {"weighted_ce", "wce"}:
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        elif self.loss_name == "focal":
            criterion = FocalLoss(gamma=self.focal_gamma, alpha=class_weights)
        elif self.loss_name in {"prior_ce", "prior_sparse_ce", "logit_adjusted_ce", "balanced_softmax_ce"}:
            criterion = PriorAdjustedSparseCELoss(class_priors=priors, tau=self.prior_tau)
        elif self.loss_name in {"prior_categorical_ce", "categorical_ce_with_prior"}:
            criterion = PriorAdjustedCategoricalCELoss(class_priors=priors, tau=self.prior_tau)
        else:
            raise ValueError(
                "loss_name 仅支持: ce, weighted_ce, focal, prior_sparse_ce(prior_ce), prior_categorical_ce"
            )

        self.criterion = criterion.to(self.device)
        return self.criterion

    def run_batch(self, batch):
        images, labels = batch
        if self.accelerator is None:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

        logits = self.model(images)
        if self.loss_name in {"prior_categorical_ce", "categorical_ce_with_prior"}:
            one_hot = F.one_hot(labels, num_classes=self.num_classes).to(dtype=logits.dtype)
            loss = self.criterion(logits, one_hot)
        else:
            loss = self.criterion(logits, labels)
        return logits, labels, loss

    @staticmethod
    def _metrics_from_confusion_matrix(cm: torch.Tensor, class_names: List[str]) -> EvalResult:
        cm = cm.to(torch.float64)
        eps = 1e-12

        tp = torch.diag(cm)
        support = cm.sum(dim=1)
        pred_count = cm.sum(dim=0)

        precision = tp / (pred_count + eps)
        recall = tp / (support + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)

        total = cm.sum().item()
        correct = tp.sum().item()

        acc = correct / max(1.0, total)
        macro_recall = recall.mean().item()
        macro_f1 = f1.mean().item()

        support_sum = support.sum().item()
        weighted_recall = (recall * support).sum().item() / max(1.0, support_sum)
        weighted_f1 = (f1 * support).sum().item() / max(1.0, support_sum)

        overall = {
            "acc": float(acc),
            "macro_recall": float(macro_recall),
            "macro_f1": float(macro_f1),
            "weighted_recall": float(weighted_recall),
            "weighted_f1": float(weighted_f1),
        }

        per_class = {}
        for i, name in enumerate(class_names):
            per_class[name] = {
                "precision": float(precision[i].item()),
                "recall": float(recall[i].item()),
                "f1": float(f1[i].item()),
                "support": int(support[i].item()),
            }

        return EvalResult(overall=overall, per_class=per_class, confusion_matrix=cm.to(torch.int64).tolist())

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> EvalResult:
        previous_mode = self.model.training
        self.model.eval()

        cm = torch.zeros((self.num_classes, self.num_classes), dtype=torch.int64, device=self.device)

        for images, labels in data_loader:
            if self.accelerator is None:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

            logits = self.model(images)
            if self.eval_logit_adjustment and self.eval_log_prior is not None:
                logits = logits + self.eval_prior_tau * self.eval_log_prior
            preds = torch.argmax(logits, dim=1)

            flat_index = labels * self.num_classes + preds
            batch_cm = torch.bincount(flat_index, minlength=self.num_classes * self.num_classes)
            cm += batch_cm.reshape(self.num_classes, self.num_classes)

        if self.accelerator is not None:
            cm = self.accelerator.reduce(cm, reduction="sum")

        if previous_mode:
            self.model.train()

        return self._metrics_from_confusion_matrix(cm.detach().cpu(), self.class_names)

    @staticmethod
    def _strategy_ckpt_path(base_ckpt_path: str, strategy: str) -> str:
        root, ext = os.path.splitext(base_ckpt_path)
        if not ext:
            ext = ".pth"
        return f"{root}.{strategy}{ext}"

    def _train_once(self, epochs: int, strategy: str) -> Dict[str, object]:
        self._set_seed(self.random_seed)
        self._setup_accelerator()

        self.load_model()
        train_loader, val_loader, stats = self.load_data(strategy=strategy)
        self.loss_function(class_counts=stats["train_counts_after_sampling_and_noise"])
        priors = torch.tensor(stats["train_counts_after_sampling_and_noise"], dtype=torch.float32)
        priors = (priors / priors.sum().clamp_min(1.0)).to(self.device)
        self.eval_log_prior = torch.log(priors.clamp_min(1e-8))
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

        if self.accelerator is not None:
            self.model, self.optimizer, train_loader, val_loader = self.accelerator.prepare(
                self.model,
                self.optimizer,
                train_loader,
                val_loader,
            )

        strategy_ckpt_path = self._strategy_ckpt_path(self.best_ckpt_path, strategy)
        if self.accelerator is None or self.accelerator.is_main_process:
            os.makedirs(os.path.dirname(os.path.abspath(strategy_ckpt_path)), exist_ok=True)
            if os.path.exists(strategy_ckpt_path):
                os.remove(strategy_ckpt_path)
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()

        best_score = float("-inf")
        best_epoch = -1
        stop_epoch = epochs
        epochs_no_improve = 0

        self.model.train()
        pbar = tqdm(
            total=len(train_loader) * epochs,
            desc=f"Total Training Steps- {self.model_name}",
            unit="step",
            dynamic_ncols=True,
            disable=(self.accelerator is not None and not self.accelerator.is_main_process),
        )
        with pbar:
            for epoch in range(epochs):
                for batch in train_loader:
                    _, _, loss = self.run_batch(batch)
                    self.optimizer.zero_grad()
                    if self.accelerator is not None:
                        self.accelerator.backward(loss)
                    else:
                        loss.backward()
                    self.optimizer.step()
                    pbar.update(1)
                    if self.accelerator is None or self.accelerator.is_main_process:
                        pbar.set_postfix({"loss": f"{loss.item():.4f}", "score": f"{best_score:.4f}"})
                val_result = self.evaluate(val_loader)
                score = val_result.overall.get(self.best_metric, val_result.overall["macro_f1"])

                # if score > (best_score + self.early_stop_min_delta) and epoch %2== 0:
                if epoch %5==0:
                    best_score = score
                    best_epoch = epoch + 1
                    epochs_no_improve = 0
                    if self.accelerator is None or self.accelerator.is_main_process:
                        state_dict = (
                            self.accelerator.get_state_dict(self.model)
                            if self.accelerator is not None
                            else self.model.state_dict()
                        )
                        torch.save(
                            {
                                "epoch": best_epoch,
                                "strategy": strategy,
                                "model_name": self.model_name,
                                "loss_name": self.loss_name,
                                "state_dict": state_dict,
                            },
                            strategy_ckpt_path,
                        )
                # else:
                #     epochs_no_improve += 1
                #     if self.early_stop_patience > 0 and epochs_no_improve >= self.early_stop_patience:
                #         stop_epoch = epoch + 1
                #         break
                if self.accelerator is not None:
                    self.accelerator.wait_for_everyone()

        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()
        checkpoint = torch.load(strategy_ckpt_path, map_location=self.device)
        load_model = self.accelerator.unwrap_model(self.model) if self.accelerator is not None else self.model
        load_model.load_state_dict(checkpoint["state_dict"])
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()
        final_result = self.evaluate(val_loader)

        return {
            "strategy": strategy,
            "best_checkpoint": strategy_ckpt_path,
            "best_epoch": best_epoch,
            "stop_epoch": stop_epoch,
            "best_score": best_score,
            "stats": stats,
            "overall": final_result.overall,
            "per_class": final_result.per_class,
            "confusion_matrix": final_result.confusion_matrix,
        }

    def _write_result_text(self, final_result: Dict[str, object], all_results: List[Dict[str, object]]) -> None:
        if self.accelerator is not None and not self.accelerator.is_main_process:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.result_txt_path)), exist_ok=True)

        lines: List[str] = []
        lines.append("=== Experiment Result ===")
        lines.append(
            _kv_table(
                "Config",
                {
                    "data_name": self.data_name,
                    "model_name": self.model_name,
                    "loss_name": self.loss_name,
                    "image_size": self.image_size,
                    "use_accelerate": self.use_accelerate,
                    "mixed_precision": self.mixed_precision,
                    "world_size": (self.accelerator.num_processes if self.accelerator is not None else 1),
                    "imbalance_factor": self.imbalance_factor,
                    "error_factor": self.error_factor,
                    "sampling_strategy": self.sampling_strategy,
                    "eval_logit_adjustment": self.eval_logit_adjustment,
                    "eval_prior_tau": self.eval_prior_tau,
                    "early_stop_patience": self.early_stop_patience,
                    "early_stop_min_delta": self.early_stop_min_delta,
                    "best_metric": self.best_metric,
                },
            )
        )
        lines.append("")

        if len(all_results) > 1:
            lines.append("[Strategy Comparison]")
            strategy_rows = []
            for r in all_results:
                o = r["overall"]
                strategy_rows.append(
                    [
                        r["strategy"],
                        f"{o['acc']:.6f}",
                        f"{o['macro_recall']:.6f}",
                        f"{o['macro_f1']:.6f}",
                        f"{o['weighted_f1']:.6f}",
                    ]
                )
            lines.append(
                _ascii_table(
                    ["strategy", "acc", "macro_recall", "macro_f1", "weighted_f1"],
                    strategy_rows,
                )
            )
            lines.append("")

        lines.append(
            _kv_table(
                "Best Result",
                {
                    "best_strategy": final_result["strategy"],
                    "best_checkpoint": self.best_ckpt_path,
                    "best_epoch": final_result["best_epoch"],
                    "stop_epoch": final_result.get("stop_epoch", 0),
                    f"selected_metric({self.best_metric})": f"{final_result['overall'].get(self.best_metric, 0.0):.6f}",
                },
            )
        )
        lines.append("")

        stats = final_result["stats"]
        raw_counts = stats.get("raw_train_counts", [])
        target_counts = stats.get("imbalance_counts", [])
        processed_counts = stats.get("train_counts_after_sampling_and_noise", [])

        raw_ratio = self._to_ratio(raw_counts)
        processed_ratio = self._to_ratio(processed_counts)
        raw_head_tail = (max(raw_counts) / max(1, min(raw_counts))) if raw_counts else 0.0
        processed_head_tail = (max(processed_counts) / max(1, min(processed_counts))) if processed_counts else 0.0

        lines.append("[Distribution]")
        dist_rows: List[List[str]] = []
        for i, cls in enumerate(self.class_names):
            dist_rows.append(
                [
                    str(i),
                    cls,
                    str(raw_counts[i] if i < len(raw_counts) else 0),
                    f"{raw_ratio[i] if i < len(raw_ratio) else 0.0:.6f}",
                    str(target_counts[i] if i < len(target_counts) else 0),
                    str(processed_counts[i] if i < len(processed_counts) else 0),
                    f"{processed_ratio[i] if i < len(processed_ratio) else 0.0:.6f}",
                ]
            )
        lines.append(
            _ascii_table(
                ["id", "class", "raw_count", "raw_ratio", "target_count", "processed_count", "processed_ratio"],
                dist_rows,
            )
        )
        lines.append(
            _ascii_table(
                ["tail_metric", "value"],
                [
                    ["raw_head_tail_ratio(max/min)", f"{raw_head_tail:.6f}"],
                    ["processed_head_tail_ratio(max/min)", f"{processed_head_tail:.6f}"],
                ],
            )
        )
        lines.append("")

        lines.append("[Overall]")
        lines.append(
            _ascii_table(
                ["metric", "value"],
                [[k, f"{v:.6f}"] for k, v in final_result["overall"].items()],
            )
        )
        lines.append("")

        lines.append("[Per Class]")
        per_class_rows = []
        for class_name, metrics in final_result["per_class"].items():
            per_class_rows.append(
                [
                    class_name,
                    f"{metrics['precision']:.6f}",
                    f"{metrics['recall']:.6f}",
                    f"{metrics['f1']:.6f}",
                    str(metrics["support"]),
                ]
            )
        lines.append(_ascii_table(["class", "precision", "recall", "f1", "support"], per_class_rows))
        lines.append("")

        lines.append("[Confusion Matrix]")
        cm_headers = ["true\\pred"] + [f"{i}:{n}" for i, n in enumerate(self.class_names)]
        cm_rows = []
        for i, row in enumerate(final_result["confusion_matrix"]):
            cm_rows.append([f"{i}:{self.class_names[i]}"] + [str(x) for x in row])
        lines.append(_ascii_table(cm_headers, cm_rows))

        with open(self.result_txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def trainer(self, epochs: int = 1) -> Dict[str, object]:
        strategies = [self.sampling_strategy]
        if self.sampling_strategy == "compare":
            strategies = ["none", "undersample", "oversample"]

        all_results = [self._train_once(epochs=epochs, strategy=s) for s in strategies]

        best_result = max(
            all_results,
            key=lambda r: r["overall"].get(self.best_metric, r["overall"]["macro_f1"]),
        )

        if self.accelerator is None or self.accelerator.is_main_process:
            shutil.copyfile(best_result["best_checkpoint"], self.best_ckpt_path)

            for r in all_results:
                ckpt = r["best_checkpoint"]
                if ckpt != self.best_ckpt_path and os.path.exists(ckpt):
                    os.remove(ckpt)
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()

        final_result = {
            "strategy": best_result["strategy"],
            "best_checkpoint": self.best_ckpt_path,
            "best_epoch": best_result["best_epoch"],
            "stop_epoch": best_result.get("stop_epoch", 0),
            "stats": best_result["stats"],
            "overall": best_result["overall"],
            "per_class": best_result["per_class"],
            "confusion_matrix": best_result["confusion_matrix"],
        }

        self._write_result_text(final_result=final_result, all_results=all_results)
        return final_result


def _normalize_name(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_").lower()


def _build_eval_la_jobs(args) -> List[Tuple[str, bool, Optional[float]]]:
    if args.eval_prior_tau_list:
        jobs: List[Tuple[str, bool, Optional[float]]] = []
        for item in args.eval_prior_tau_list:
            token = str(item).strip().lower()
            if token in {"off", "none", "false", "no"}:
                jobs.append(("off", False, None))
            else:
                tau = float(token)
                jobs.append((f"tau_{tau:g}", True, tau))
        return jobs

    if args.eval_logit_adjustment:
        return [("on", True, args.eval_prior_tau)]
    return [("off", False, None)]


def _build_loss_jobs(args) -> List[str]:
    if args.loss_names:
        return [str(x).strip().lower() for x in args.loss_names]
    return [str(args.loss_name).strip().lower()]


def _is_global_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def run_experiment_matrix(args) -> str:
    os.makedirs(args.output_dir, exist_ok=True)

    summary_lines: List[str] = []
    summary_lines.append("=== Batch Experiment Summary ===")
    summary_lines.append(
        _kv_table(
            "Config",
            {
                "data_name": args.data_name,
                "imbalance_factor": args.imbalance_factor,
                "error_factor": args.error_factor,
                "include_normal_condition": args.include_normal_condition,
                "loss_name(default)": args.loss_name,
                "loss_names(grid)": " ".join(args.loss_names) if args.loss_names else "None",
                "use_accelerate": (not args.disable_accelerate),
                "mixed_precision": args.mixed_precision,
                "eval_prior_tau_list": " ".join(args.eval_prior_tau_list) if args.eval_prior_tau_list else "None",
                "eval_logit_adjustment(default)": args.eval_logit_adjustment,
                "eval_prior_tau(default)": args.eval_prior_tau,
                "early_stop_patience": args.early_stop_patience,
                "early_stop_min_delta": args.early_stop_min_delta,
                "epochs": args.epochs,
            },
        )
    )
    summary_lines.append("")

    all_rows: List[Tuple[str, str, str, str, str, Dict[str, float], str]] = []
    eval_la_jobs = _build_eval_la_jobs(args)
    loss_jobs = _build_loss_jobs(args)

    condition_jobs = [
        ("imbalanced_noisy", args.imbalance_factor, args.error_factor, list(args.sampling_strategies)),
    ]
    if args.include_normal_condition:
        condition_jobs.append(("normal", 1.0, 0.0, ["none"]))

    for condition_name, imbalance_factor, error_factor, strategy_list in condition_jobs:
        for model_name in args.models:
            for strategy in strategy_list:
                for loss_name in loss_jobs:
                    for eval_label, eval_enabled, eval_tau in eval_la_jobs:
                        tag = (
                            f"{_normalize_name(condition_name)}__"
                            f"{_normalize_name(model_name)}__{_normalize_name(strategy)}__"
                            f"loss_{_normalize_name(loss_name)}__"
                            f"evalla_{_normalize_name(eval_label)}"
                        )
                        ckpt_path = os.path.join(args.output_dir, f"best_model.{tag}.pth")
                        result_path = os.path.join(args.output_dir, f"result.{tag}.txt")

                        trainer = DistributeTrainer(
                            data_name=args.data_name,
                            model_name=model_name,
                            random_seed=args.random_seed,
                            error_factor=error_factor,
                            imbalance_factor=imbalance_factor,
                            sampling_strategy=strategy,
                            loss_name=loss_name,
                            focal_gamma=args.focal_gamma,
                            prior_tau=args.prior_tau,
                            image_size=args.image_size,
                            use_accelerate=(not args.disable_accelerate),
                            mixed_precision=args.mixed_precision,
                            eval_logit_adjustment=eval_enabled,
                            eval_prior_tau=eval_tau,
                            best_metric=args.best_metric,
                            early_stop_patience=args.early_stop_patience,
                            early_stop_min_delta=args.early_stop_min_delta,
                            batch_size=args.batch_size,
                            lr=args.lr,
                            num_workers=args.num_workers,
                            data_root=args.data_root,
                            best_ckpt_path=ckpt_path,
                            result_txt_path=result_path,
                        )
                        result = trainer.trainer(epochs=args.epochs)
                        all_rows.append(
                            (
                                condition_name,
                                model_name,
                                strategy,
                                loss_name,
                                eval_label,
                                result["overall"],
                                result["best_checkpoint"],
                            )
                        )

    all_rows.sort(
        key=lambda x: x[5].get(args.best_metric, x[5]["macro_f1"]),
        reverse=True,
    )

    summary_lines.append("[Leaderboard]")
    leaderboard_rows = []
    for condition_name, model_name, strategy, loss_name, eval_label, overall, ckpt in all_rows:
        leaderboard_rows.append(
            [
                condition_name,
                model_name,
                strategy,
                loss_name,
                eval_label,
                f"{overall['acc']:.6f}",
                f"{overall['macro_recall']:.6f}",
                f"{overall['macro_f1']:.6f}",
                f"{overall['weighted_f1']:.6f}",
                ckpt,
            ]
        )
    summary_lines.append(
        _ascii_table(
            ["condition", "model", "strategy", "loss", "eval_la", "acc", "macro_recall", "macro_f1", "weighted_f1", "best_ckpt"],
            leaderboard_rows,
        )
    )

    summary_path = os.path.join(args.output_dir, "summary.txt")
    if _is_global_main_process():
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(summary_lines) + "\n")
    return summary_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CIFAR-10 imbalance/noise training and evaluation")
    parser.add_argument("--data-name", type=str, default="cifar10")
    parser.add_argument("--model-name", type=str, default="resnet50")
    parser.add_argument("--models", nargs="+", default=["resnet50", "vit-b/16"])
    parser.add_argument("--random-seed", type=int, default=20260607)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--data-root", type=str, default="/home/huangjie/.cache")
    parser.add_argument("--imbalance-factor", type=float, default=0.01)
    parser.add_argument("--error-factor", type=float, default=0.1)
    parser.add_argument("--sampling-strategy", type=str, default="compare")
    parser.add_argument(
        "--sampling-strategies",
        nargs="+",
        default=["none", "undersample", "oversample", "compare"],
    )
    parser.add_argument(
        "--loss-name",
        type=str,
        default="prior_sparse_ce",
        choices=["ce", "weighted_ce", "focal", "prior_sparse_ce", "prior_ce", "prior_categorical_ce"],
    )
    parser.add_argument(
        "--loss-names",
        nargs="+",
        default=None,
        help="仅 run-grid 生效；批量测试多个 loss_name（例如: ce weighted_ce focal prior_ce）",
    )
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--prior-tau", type=float, default=1.0)
    parser.add_argument(
        "--mixed-precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help="混合精度模式（accelerate）",
    )
    parser.add_argument(
        "--disable-accelerate",
        action="store_true",
        help="关闭 accelerate（默认开启）",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="输入分辨率；默认按模型自动设置(resnet50=224, vit-b/16=224)",
    )
    parser.add_argument(
        "--eval-logit-adjustment",
        action="store_true",
        help="在 evaluate 阶段对 logits 加 tau*log(prior) 再做预测",
    )
    parser.add_argument(
        "--eval-prior-tau",
        type=float,
        default=None,
        help="evaluate 阶段 Logit Adjustment 的 tau，默认复用 --prior-tau",
    )
    parser.add_argument(
        "--eval-prior-tau-list",
        nargs="+",
        default=None,
        help="仅 run-grid 生效；批量测试 evaluate 阶段 Logit Adjustment，支持 off 和数值(如: off 0.5 1.0 2.0)",
    )
    parser.add_argument("--best-metric", type=str, default="macro_f1")
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="早停容忍轮数；0 表示关闭早停",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=0.0,
        help="判定指标提升的最小阈值",
    )
    parser.add_argument("--best-ckpt-path", type=str, default="./best_model.pth")
    parser.add_argument("--result-txt-path", type=str, default="./train_result.txt")
    parser.add_argument("--output-dir", type=str, default="./runs")
    parser.add_argument(
        "--run-grid",
        action="store_true",
        help="批量测试不同模型+不同采样策略，生成 summary.txt",
    )
    parser.add_argument(
        "--include-normal-condition",
        action="store_true",
        help="run-grid 时额外加入 normal 条件(imbalance_factor=1.0, error_factor=0.0)做对照",
    )
    return parser


if __name__ == "__main__":
    """
    CUDA_VISIBLE_DEVICES=1 python3 data_distribute.py \
      --run-grid \
      --include-normal-condition \
      --models resnet50 vit-b/16 \
      --sampling-strategies none undersample oversample compare \
      --loss-names ce weighted_ce focal prior_ce \
      --eval-prior-tau-list off 0.5 1.0 2.0 \
      --imbalance-factor 0.1 \
      --error-factor 0.0 \
      --early-stop-patience 5 \
      --early-stop-min-delta 0.005 \
      --epochs 20 \
      --output-dir ./DataDistributeResult

    CUDA_VISIBLE_DEVICES=1,2 accelerate launch --num_processes 2 data_distribute.py \
      --run-grid \
      --include-normal-condition \
      --models resnet50\
      --sampling-strategies none undersample oversample \
      --loss-names ce focal prior_ce \
      --eval-prior-tau-list off 1.0 \
      --mixed-precision fp16 \
      --imbalance-factor 0.1 \
      --error-factor 0.0 \
      --epochs 30 \
      --batch-size 700 \
      --lr 1e-5 \
      --output-dir ./DataDistributeResult
    """
    parser = build_arg_parser()
    cli_args = parser.parse_args()

    if cli_args.run_grid:
        run_experiment_matrix(cli_args)
    else:
        trainer = DistributeTrainer(
            data_name=cli_args.data_name,
            model_name=cli_args.model_name,
            random_seed=cli_args.random_seed,
            error_factor=cli_args.error_factor,
            imbalance_factor=cli_args.imbalance_factor,
            sampling_strategy=cli_args.sampling_strategy,
            loss_name=cli_args.loss_name,
            focal_gamma=cli_args.focal_gamma,
            prior_tau=cli_args.prior_tau,
            image_size=cli_args.image_size,
            use_accelerate=(not cli_args.disable_accelerate),
            mixed_precision=cli_args.mixed_precision,
            eval_logit_adjustment=cli_args.eval_logit_adjustment,
            eval_prior_tau=cli_args.eval_prior_tau,
            best_metric=cli_args.best_metric,
            early_stop_patience=cli_args.early_stop_patience,
            early_stop_min_delta=cli_args.early_stop_min_delta,
            batch_size=cli_args.batch_size,
            lr=cli_args.lr,
            num_workers=cli_args.num_workers,
            data_root=cli_args.data_root,
            best_ckpt_path=cli_args.best_ckpt_path,
            result_txt_path=cli_args.result_txt_path,
        )
        trainer.trainer(epochs=cli_args.epochs)
