import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import os
from torchvision import datasets 
from PIL import Image


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[OK] Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    # Optimize GPU for faster training
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True  # Auto-tune convolution algorithms

EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
NUM_CLASSES = len(EMOTIONS) 



Interval_Check = 30
EmotionDB = "Facial_Detection/Emotion_DB/"

class RGBImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):   # overrides the parent method
        path, target = self.samples[index]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, target

def build_model(num_classes=NUM_CLASSES, dropout=0.3):
    # Load MobileNetV3-Small with ImageNet weights 
    model = models.mobilenet_v3_small(weights="IMAGENET1K_V1")

    #Debug 
    #print(model.classifier)
 

    # Replace the final Linear layer only, keep the rest of the head
    model.classifier[3] = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(1024, num_classes)
    )

    for param in model.features.parameters():
        param.requires_grad = False

    for param in model.classifier[:3].parameters():
        param.requires_grad = False


    return model.to(DEVICE)
def count_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

def LoadData(db_path = EmotionDB, batch_size=64): #I'll try uping batch size later when i make sure my computer wont crash
    Train_Transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(), 
        transforms.Normalize([0.485, 0.456,0.406],[0.229, 0.224,0.225])]) #mean and std of the ImageNet ds that the original model was trained on
    
    Val_Transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]) 

    Train_ds = RGBImageFolder(os.path.join(db_path, "train"), transform=Train_Transform)
    Val_ds   = RGBImageFolder(os.path.join(db_path, "test"),  transform=Val_Transform)

    targets = [s[1] for s in Train_ds.samples]
    class_counts = np.bincount(targets)
    class_weights = 1.0 / class_counts   #handles inbalance between the classes
    sample_weights = [class_weights[t] for t in targets]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(Train_ds, batch_size=batch_size, sampler=sampler, num_workers=2)
    val_loader   = DataLoader(Val_ds,   batch_size=batch_size, shuffle=False,   num_workers=2)

    #debugs 
    print(f"Classes : {Train_ds.classes}")
    print(f"Train   : {len(Train_ds)} images")
    print(f"Val     : {len(Val_ds)} images")

    return train_loader, val_loader



def unfreeze_last_n(model, n=3 ): 
    blocks = list(model.features.children())
    for block in blocks[-n:]:
        for param in block.parameters():
            param.requires_grad = True


def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True



def make_optimizer(model, lr): 
     
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)


def make_scheduler(optimizer, epochs):
     return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=optimizer.param_groups[0]["lr"] * 0.01
    )

def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()
 
    total_loss, correct, total = 0.0, 0, 0
 
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
 
            logits = model(images)
            loss   = criterion(logits, labels)
 
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
 
            total_loss += loss.item() * labels.size(0)
            correct    += (logits.argmax(dim=1) == labels).sum().item()
            total      += labels.size(0)
 
    return total_loss / total, correct / total
def train(model, train_loader, val_loader):

    phases = [
        {
            "name":     "Phase 1 — head only", #first attempt was 5 plateaued at the end of phase 2 
            "epochs":   10, 
            "lr":       1e-3,
            "unfreeze": None
        },
        {
            "name":     "Phase 2 — partial unfreeze",
            "epochs":   5,
            "lr":       1e-4,
            "unfreeze": lambda m: unfreeze_last_n(m, n=3)
        },
        {
            "name":     "Phase 3 — full unfreeze",
            "epochs":   5,
            "lr":       1e-5,
            "unfreeze": lambda m: unfreeze_all(m)
        },
    ]

    
    targets = [s[1] for s in train_loader.dataset.samples]
    class_counts = np.bincount(targets)
    class_weights = torch.tensor(1.0 / class_counts, dtype=torch.float).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1) #first attempt stagnated at 0.567 trying label smoothing



    best_val_acc = 0.0
    
    for phase in phases:
        print(f"\n{phase['name']}")
        print("-" * 40)

        # unfreeze layers for this phase (phase 1 skips this)
        if phase["unfreeze"]:
            phase["unfreeze"](model)
            count_params(model)

        # rebuild optimizer after every unfreeze
        optimizer = make_optimizer(model, lr=phase["lr"])
        scheduler = make_scheduler(optimizer, phase["epochs"])

        # loop through each epoch 
        for epoch in range(1, phase["epochs"] + 1):

            # training pass weights are updated
            tr_loss, tr_acc = run_epoch(
                model, train_loader, criterion, optimizer
            )

            # validation pass weights arent updated here
            vl_loss, vl_acc = run_epoch(
                model, val_loader, criterion
            )

            # decay the learning rate
            scheduler.step()
            print(
                f"  Ep {epoch:02d} | "
                f"train loss {tr_loss:.3f}  acc {tr_acc:.3f} | "
                f"val loss {vl_loss:.3f}  acc {vl_acc:.3f}"
            )

            # save only when val accuracy improves
            if vl_acc > best_val_acc:
                best_val_acc = vl_acc
                torch.save(model.state_dict(), "best_model.pth")
                print(f"           Saved — best val acc: {best_val_acc:.3f}")


    print(f"\nDone. Best val acc: {best_val_acc:.3f}")



def main():

    model = build_model()
    count_params(model)
    train_loader, val_loader = LoadData()
    
    train(model, train_loader, val_loader)
    


if __name__ == "__main__":
    main()