import torch
import os
from torch import nn
from torch.utils.data import DataLoader, random_split
from model import S2VT
from msvd import MSVD
from tqdm import tqdm
from eval import convert_pred_caption, convert_gt_caption, eval_collate_fn
from nltk.translate.meteor_score import meteor_score
import config


os.makedirs("runs", exist_ok=True)
# Train & Val subset split
train_dataset = MSVD(config.FEATURES_DIR, config.CAPTION_FILE, config.VOCAB_SIZE,
                     config.CAPTION_MAX_LENGTH, split=config.TRAIN_SPLIT_FILE)
val_dataset = MSVD(config.FEATURES_DIR, config.CAPTION_FILE, config.VOCAB_SIZE,
                   config.CAPTION_MAX_LENGTH, split=config.VAL_SPLIT_FILE, training=False)

print("Training data size:", len(train_dataset))
print("Validation data size:", len(val_dataset))
# train_subset, val_subset = random_split(train_dataset, [train_length, val_length],
#                                         generator=torch.Generator().manual_seed(2022))
train_loader = DataLoader(train_dataset, config.BATCH_SIZE)
val_loader = DataLoader(val_dataset, 1, collate_fn=eval_collate_fn)

# Define model, loss, optimizer
device = "cuda" if torch.cuda.is_available() else "cpu"
s2vt_model = S2VT(config.IMAGE_FEATURE_SIZE, config.HIDDEN_SIZE, config.VOCAB_SIZE,
                  config.CAPTION_MAX_LENGTH + 2, device)
s2vt_model.to(device)

criterion = nn.CrossEntropyLoss(reduction="none")
# optimizer = torch.optim.SGD(s2vt_model.parameters(), lr=0.001, momentum=0.9)
optimizer = torch.optim.Adam(s2vt_model.parameters())
best_meteor_score = 0

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
    avg_meteor_score = 0
    s2vt_model.eval()
    with torch.no_grad():
        for vid_features, caption in (pbar := tqdm(val_loader)):
            vid_features = vid_features.to(device)
            pred_caption = s2vt_model(vid_features, word2idx=train_dataset.word2idx)
            pred_caption = convert_pred_caption(pred_caption, train_dataset.idx2word)
            gt_caption = convert_gt_caption(caption)
            avg_meteor_score += meteor_score(gt_caption, pred_caption)
            pbar.set_description(f"Evaluating [{epoch + 1}/{config.NUM_EPOCH}] epoch")

    avg_meteor_score = avg_meteor_score / len(val_loader)
    tqdm.write(f"Epoch [{epoch+1}/{config.NUM_EPOCH}]: loss_train: {epoch_train_avg_loss/len(train_loader)},"
               f" eval_meteor_score: {avg_meteor_score}")
    if avg_meteor_score >= best_meteor_score:
        tqdm.write("Best eval so far, saving model")
        best_meteor_score = avg_meteor_score
        torch.save(s2vt_model.state_dict(), "runs/best.pth")
    if epoch % 5 == 0:
        torch.save(s2vt_model.state_dict(), f"runs/epoch_{epoch}.pth")
    torch.save(s2vt_model.state_dict(), f"runs/last.pth")

