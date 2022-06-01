import torch
import os
from torch import nn
from torch.utils.data import DataLoader, random_split
from model import S2VT
from msvd import MSVD, Vocab
from tqdm import tqdm
from eval import convert_pred_caption, convert_gt_caption, eval_collate_fn
from nltk.translate.meteor_score import meteor_score
import numpy as np
import config
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--checkpoint", type=str, help="Model checkpoint")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
args = parser.parse_args()
print(args)


os.makedirs("runs", exist_ok=True)
# Train & Val subset split
vocab = Vocab(config.CAPTION_FILE, config.VOCAB_SIZE, config.CAPTION_MAX_LENGTH)
train_dataset = MSVD(config.FEATURES_DIR, config.CAPTION_FILE, config.CAPTION_MAX_LENGTH,
                     vocab, split=config.TRAIN_SPLIT_FILE)
val_dataset = MSVD(config.FEATURES_DIR, config.CAPTION_FILE, config.CAPTION_MAX_LENGTH, vocab,
                   split=config.VAL_SPLIT_FILE, training=False)

print("Training data size:", len(train_dataset))
train_val_dataset = MSVD(config.FEATURES_DIR, config.CAPTION_FILE, config.CAPTION_MAX_LENGTH, vocab,
                         split=config.TRAIN_SPLIT_FILE, training=False)
train_loader = DataLoader(train_dataset, config.BATCH_SIZE, shuffle=True)
train_val_loader = DataLoader(val_dataset, 1, collate_fn=eval_collate_fn)

# Define model, loss, optimizer
device = "cuda" if torch.cuda.is_available() else "cpu"
s2vt_model = S2VT(config.IMAGE_FEATURE_SIZE, config.HIDDEN_SIZE, config.VOCAB_SIZE,
                  config.CAPTION_MAX_LENGTH + 2, device)
s2vt_model.to(device)

criterion = nn.CrossEntropyLoss(reduction="none")
# optimizer = torch.optim.SGD(s2vt_model.parameters(), lr=0.01, momentum=0.9)
optimizer = torch.optim.Adam(s2vt_model.parameters(), lr=args.lr)

if args.checkpoint and os.path.exists(args.checkpoint):
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint)
    s2vt_model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
else:
    print("No checkpoint provided, training from scratch...")

best_meteor_score = 0

for epoch in range(config.NUM_EPOCH):
    # Training
    epoch_train_avg_loss = 0
    s2vt_model.train()
    for vid_features, caption, caption_mask in (pbar := tqdm(train_loader)):
        caption_mask = caption_mask[:, 1:]  # Mask start after <BOS> to represent what should LSTM outputs
        vid_features, caption, caption_mask = vid_features.to(device), caption.to(device), caption_mask.to(device)
        pred_caption = s2vt_model(vid_features, caption, word2idx=vocab.word2idx)
        # pred_caption = torch.permute(pred_caption, (0, 2, 1))
        # print("Caption: ")
        # print(caption)
        # print("Predicted: ")
        # print(pred_caption.max(dim=1)[1])
        # print("Mask")
        # print(caption_mask)
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
    avg_meteor_score = 0
    sample_captions = []
    sample_gt_captions = []
    sample_idx = np.random.choice(len(train_val_loader), min(len(train_val_loader), 5), replace=False)
    print(sample_idx)
    s2vt_model.eval()
    with torch.no_grad():
        for idx, (vid_features, caption) in enumerate(pbar := tqdm(train_val_loader)):
            vid_features = vid_features.to(device)
            pred_caption = s2vt_model(vid_features, word2idx=vocab.word2idx)
            pred_caption = convert_pred_caption(pred_caption, vocab.idx2word)
            gt_caption = convert_gt_caption(caption)
            avg_meteor_score += meteor_score(gt_caption, pred_caption)
            pbar.set_description(f"Evaluating [{epoch + 1}/{config.NUM_EPOCH}] epoch")
            if idx in sample_idx:
                sample_captions.append(pred_caption)
                # Add 1 sample caption
                sample_gt_captions.append(gt_caption[0])

    avg_meteor_score = avg_meteor_score / len(train_val_loader)
    tqdm.write(f"Epoch [{epoch+1}/{config.NUM_EPOCH}]: loss_train: {epoch_train_avg_loss/len(train_loader)},"
               f" eval_meteor_score: {avg_meteor_score}")
    tqdm.write("Sample caption generated:")
    for sample_caption, sample_gt in zip(sample_captions, sample_gt_captions):
        tqdm.write(f"Pred: {' '.join(sample_caption)}. GT: {' '.join(sample_gt)}")

    # Model saving
    checkpoint = {
        "model": s2vt_model.state_dict(),
        "optimizer": optimizer.state_dict()
    }

    if avg_meteor_score >= best_meteor_score:
        tqdm.write("Best eval so far, saving model")
        best_meteor_score = avg_meteor_score
        torch.save(checkpoint, "runs/best.pth")
    if epoch % 10 == 0:
        torch.save(checkpoint, f"runs/epoch_{epoch}.pth")
    torch.save(checkpoint, f"runs/last.pth")

