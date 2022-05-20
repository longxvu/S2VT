import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from model import S2VT
from msvd import MSVD
from tqdm import tqdm
import config
import numpy as np


# Train & Val subset split
train_dataset = MSVD(config.FEATURES_DIR, config.CAPTION_FILE, config.VOCAB_SIZE,
                     config.CAPTION_MAX_LENGTH, split=config.SPLIT_FILE)
train_length = int(config.TRAIN_RATIO * len(train_dataset))
val_length = len(train_dataset) - train_length
train_subset, val_subset = random_split(train_dataset, [train_length, val_length],
                                        generator=torch.Generator().manual_seed(2022))
train_loader = DataLoader(train_subset, config.BATCH_SIZE)
val_loader = DataLoader(val_subset, config.BATCH_SIZE)

# Define model, loss, optimizer
device = "cuda" if torch.cuda.is_available() else "cpu"
s2vt_model = S2VT(config.IMAGE_FEATURE_SIZE, config.HIDDEN_SIZE, config.VOCAB_SIZE,
                  config.CAPTION_MAX_LENGTH + 2, device)
s2vt_model.to(device)

criterion = nn.CrossEntropyLoss(reduction="none")
# optimizer = torch.optim.SGD(s2vt_model.parameters(), lr=0.001, momentum=0.9)
optimizer = torch.optim.Adam(s2vt_model.parameters())
loss_so_far = np.inf

for epoch in range(config.NUM_EPOCH):
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
        pbar.set_description(f"Training [{epoch+1}/{config.NUM_EPOCH}] epoch")
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
            pbar.set_description(f"Evaluating [{epoch + 1}/{config.NUM_EPOCH}] epoch")

    tqdm.write(f"Epoch [{epoch+1}/{config.NUM_EPOCH}]: loss_train: {epoch_train_avg_loss/len(train_loader)},"
               f" loss_val: {epoch_val_avg_loss/len(val_loader)}")
    if epoch_val_avg_loss < loss_so_far:
        tqdm.write("Best eval so far, saving model")
        loss_so_far = epoch_val_avg_loss
        torch.save(s2vt_model.state_dict(), "best.pth")
    torch.save(s2vt_model.state_dict(), "last.pth")

