#!/usr/bin/env python3
# tools/train_resnet18_onnx.py
import os, json, argparse, random, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler

import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder

from sklearn.metrics import f1_score
from tqdm import tqdm

# -------------------- utils --------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def device_auto():
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# -------------------- data --------------------
def build_loaders(data_dir, img_size=224, batch_size=32, val_split=0.2, seed=42):
    # simple augs; works fine even without pretrained weights
    tfm_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomApply([transforms.ColorJitter(0.2,0.2,0.2,0.1)], p=0.3),
        transforms.ToTensor(),
        # ImageNet-ish normalization is fine even from scratch
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    tfm_val = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    full = ImageFolder(data_dir, transform=tfm_train)
    n_total = len(full)
    if n_total < 4:
        raise SystemExit(f"Not enough images found under {data_dir} (found {n_total}). Need at least a few per class.")
    n_val = max(1, int(n_total*val_split))
    n_train = n_total - n_val

    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full, [n_train, n_val], generator=g)

    # swap validation transform to non-aug
    val_ds.dataset = ImageFolder(data_dir, transform=tfm_val)
    val_ds.indices = val_ds.indices

    # ----- class-balanced sampler on the train subset -----
    targets = [full.targets[i] for i in train_ds.indices]
    counts = torch.bincount(torch.tensor(targets))
    weights = 1.0 / counts.float().clamp(min=1)
    sample_w = [float(weights[t]) for t in targets]
    sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=2, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,   num_workers=2, pin_memory=False)

    class_to_idx = full.class_to_idx  # e.g. {'clean':0,'dirty':1}
    return train_loader, val_loader, class_to_idx

# -------------------- model --------------------
def build_model(num_classes=2, use_pretrained=False):
    """
    use_pretrained=False avoids downloading ImageNet weights (works fully offline).
    """
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if use_pretrained else None
    m = torchvision.models.resnet18(weights=weights)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

# -------------------- train / eval --------------------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_targs = [], []
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
        all_preds.extend(pred.cpu().tolist())
        all_targs.extend(y.cpu().tolist())
    acc = correct / max(1, total)
    f1 = f1_score(all_targs, all_preds, average="macro")
    return acc, f1

def train_one_epoch(model, loader, device, criterion, optimizer):
    model.train()
    running = 0.0
    for x, y in tqdm(loader, leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        running += loss.item() * y.size(0)
    return running / max(1, len(loader.dataset))

# -------------------- export --------------------
def export_onnx(model, img_size, out_path):
    model_cpu = model.to("cpu").eval()
    dummy = torch.randn(1, 3, img_size, img_size, dtype=torch.float32)
    out_path = str(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.onnx.export(
        model_cpu, dummy, out_path,
        input_names=["input"],
        output_names=["logits"],
        opset_version=12,
        dynamic_axes=None,  # static (1,3,H,W) as our API resizes to model size
    )
    print(f"[OK] Exported ONNX to {out_path}")

# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser(description="Train a tiny ResNet18 (2-class) and export ONNX.")
    ap.add_argument("--data", default="data", help="Root folder with class subfolders, e.g. data/clean, data/dirty")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--val_split", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--export", default="service/model.onnx")
    ap.add_argument("--labels_out", default="service/labels.json")
    ap.add_argument("--pretrained", action="store_true", help="(Optional) use ImageNet weights (needs internet)")
    args = ap.parse_args()

    set_seed(args.seed)
    device = device_auto()
    print("Device:", device)

    # data
    train_loader, val_loader, class_to_idx = build_loaders(
        args.data, img_size=args.img, batch_size=args.batch, val_split=args.val_split, seed=args.seed
    )
    print("Classes:", class_to_idx)  # expect {'clean':0,'dirty':1}

    # model
    model = build_model(num_classes=2, use_pretrained=args.pretrained).to(device)

    # train
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1,args.epochs))

    best_f1, best_state = -1.0, None
    t0 = time.time()
    for epoch in range(1, args.epochs+1):
        loss = train_one_epoch(model, train_loader, device, criterion, optimizer)
        acc, f1 = evaluate(model, val_loader, device)
        scheduler.step()
        print(f"Epoch {epoch:02d}/{args.epochs} | loss {loss:.4f} | val_acc {acc:.3f} | val_f1 {f1:.3f}")
        if f1 > best_f1:
            best_f1, best_state = f1, {k: v.cpu() for k, v in model.state_dict().items()}

    # restore best and export
    if best_state is not None:
        model.load_state_dict(best_state)

    export_onnx(model, args.img, args.export)

    # labels.json (index -> name in correct order)
    # ImageFolder maps alphabetically: clean->0, dirty->1 (perfect for our API default CLASS_ORDER)
    inv = {idx: name for name, idx in class_to_idx.items()}   # e.g. {0:'clean',1:'dirty'}
    labels = [inv[i].upper() for i in range(len(inv))]
    os.makedirs(Path(args.labels_out).parent, exist_ok=True)
    with open(args.labels_out, "w") as f:
        json.dump(labels, f)
    print(f"[OK] Wrote labels to {args.labels_out}: {labels}")
    print(f"[DONE] Total time: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()

