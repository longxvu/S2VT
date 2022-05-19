import torch
from torch import nn
from torch.utils.data import DataLoader
from model import S2VT
from msvd import MSVD
from tqdm import tqdm
import numpy as np


# Image features config

BATCH_SIZE = 16
NUM_EPOCH = 30
HIDDEN_SIZE = 512
VOCAB_SIZE = 1500
IMAGE_FEATURE_SIZE = 1280
CAPTION_MAX_LENGTH = 30
features_dir = "data/YoutubeClips_features"
caption_file = "data/AllVideoDescriptions.txt"

device = "cuda" if torch.cuda.is_available() else "cpu"
train_dataset = MSVD(features_dir, caption_file, VOCAB_SIZE, CAPTION_MAX_LENGTH)

s2vt_model = S2VT(IMAGE_FEATURE_SIZE, HIDDEN_SIZE, VOCAB_SIZE, CAPTION_MAX_LENGTH + 2, device)
s2vt_model.to(device)

train_loader = DataLoader(train_dataset, BATCH_SIZE)
criterion = nn.CrossEntropyLoss(reduction="none")
# optimizer = torch.optim.SGD(s2vt_model.parameters(), lr=0.001, momentum=0.9)
optimizer = torch.optim.Adam(s2vt_model.parameters())
loss_so_far = np.inf

for epoch in range(NUM_EPOCH):
    # Training
    s2vt_model.train()
    for vid_features, caption, caption_mask in (pbar := tqdm(train_loader)):
        caption_mask = caption_mask[:, 1:]  # Mask start after <BOS> to represent what should LSTM outputs
        vid_features, caption, caption_mask = vid_features.to(device), caption.to(device), caption_mask.to(device)
        pred_caption = s2vt_model(vid_features, caption, train_dataset.word2idx)
        # print("True caption")
        # print(caption)
        # print("Predicted caption")
        # # print(pred_caption)
        # print(pred_caption.max(dim=2)[1])
        pred_caption = torch.permute(pred_caption, (0, 2, 1))   # To follow cross entropy loss: (B, C, k...) format
        loss = criterion(pred_caption, caption[:, 1:])                 # pred_caption: (B, 1500, 32), caption: (B, 32)
        loss = loss * caption_mask
        loss = torch.sum(loss) / torch.sum(caption_mask)        # loss on caption part only

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.set_description(f"Epoch [{epoch+1}/{NUM_EPOCH}]")
        pbar.set_postfix({"Loss": loss.item()})

        if loss.item() < loss_so_far:
            tqdm.write("Best eval so far, saving model")
            loss_so_far = loss.item()
            torch.save(s2vt_model.state_dict(), "best.pth")
        torch.save(s2vt_model.state_dict(), "last.pth")

    # Eval on evaluation set
    # s2vt_model.eval()
    # with torch.no_grad():
    #     pred_caption = s2vt_model(vid_features, word2idx=train_dataset.word2idx)
    #     print("Eval caption")
    #     print(pred_caption)

