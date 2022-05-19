import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from model import S2VT
from msvd import MSVD
from tqdm import tqdm
import numpy as np


# Config + hyperparameter
BATCH_SIZE = 16
NUM_EPOCH = 30
HIDDEN_SIZE = 512
VOCAB_SIZE = 1500
IMAGE_FEATURE_SIZE = 1280
CAPTION_MAX_LENGTH = 30
TRAIN_RATIO = 0.9
features_dir = "data/YoutubeClips_features"
caption_file = "data/AllVideoDescriptions.txt"
split_file = "data/train_split.txt"

# Train & Val subset split
train_dataset = MSVD(features_dir, caption_file, VOCAB_SIZE, CAPTION_MAX_LENGTH, split=split_file)
train_length = int(TRAIN_RATIO * len(train_dataset))
val_length = len(train_dataset) - train_length
train_subset, val_subset = random_split(train_dataset, [train_length, val_length],
                                        generator=torch.Generator().manual_seed(2022))
train_loader = DataLoader(train_subset, BATCH_SIZE)
val_loader = DataLoader(val_subset, BATCH_SIZE)

# Define model, loss, optimizer
device = "cuda" if torch.cuda.is_available() else "cpu"
s2vt_model = S2VT(IMAGE_FEATURE_SIZE, HIDDEN_SIZE, VOCAB_SIZE, CAPTION_MAX_LENGTH + 2, device)
s2vt_model.to(device)

criterion = nn.CrossEntropyLoss(reduction="none")
# optimizer = torch.optim.SGD(s2vt_model.parameters(), lr=0.001, momentum=0.9)
optimizer = torch.optim.Adam(s2vt_model.parameters())
loss_so_far = np.inf

for epoch in range(NUM_EPOCH):
    # Training
    epoch_train_avg_loss = 0
    s2vt_model.train()
    for vid_features, caption, caption_mask in (pbar := tqdm(train_loader)):
        caption_mask = caption_mask[:, 1:]  # Mask start after <BOS> to represent what should LSTM outputs
        vid_features, caption, caption_mask = vid_features.to(device), caption.to(device), caption_mask.to(device)
        pred_caption = s2vt_model(vid_features, caption)
        loss = criterion(pred_caption, caption[:, 1:])          # pred_caption: (B, 1500, 32), caption: (B, 32)
        loss = loss * caption_mask
        loss = torch.sum(loss) / torch.sum(caption_mask)        # loss on caption part only

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.set_description(f"Training [{epoch+1}/{NUM_EPOCH}] epoch")
        pbar.set_postfix({"Loss": loss.item()})
        epoch_train_avg_loss += loss.item()

    # Evaluation
    # Same procedure as above. Maybe we need another way to evaluate instead of computing the loss?
    epoch_val_avg_loss = 0
    with torch.no_grad():
        for vid_features, caption, caption_mask in (pbar := tqdm(val_loader)):
            caption_mask = caption_mask[:, 1:]
            vid_features, caption, caption_mask = vid_features.to(device), caption.to(device), caption_mask.to(device)
            pred_caption = s2vt_model(vid_features, caption)
            loss = criterion(pred_caption, caption[:, 1:])
            loss = loss * caption_mask
            loss = torch.sum(loss) / torch.sum(caption_mask)
            epoch_val_avg_loss += loss.item()
            pbar.set_description(f"Evaluating [{epoch + 1}/{NUM_EPOCH}] epoch")

    tqdm.write(f"Epoch [{epoch+1}/{NUM_EPOCH}]: loss_train: {epoch_train_avg_loss/len(train_loader)},"
               f" loss_val: {epoch_val_avg_loss/len(val_loader)}")
    if epoch_val_avg_loss < loss_so_far:
        tqdm.write("Best eval so far, saving model")
        loss_so_far = epoch_val_avg_loss
        torch.save(s2vt_model.state_dict(), "best.pth")
    torch.save(s2vt_model.state_dict(), "last.pth")

