import torch
from model import S2VT
from msvd import MSVD
import numpy as np


# Can replace features with video by replacing npy file with extract_images and features from preprocess
feature_name = "-4wsuPCjDBc_5_15"
features = np.load(f"data/YoutubeClips_features/{feature_name}.npy")
train_data = MSVD("data/YoutubeClips_features", "data/AllVideoDescriptions.txt")

HIDDEN_SIZE = 512
VOCAB_SIZE = 1500
IMAGE_FEATURE_SIZE = 1280
CAPTION_MAX_LENGTH = 30

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
s2vt_model = S2VT(IMAGE_FEATURE_SIZE, HIDDEN_SIZE, VOCAB_SIZE, CAPTION_MAX_LENGTH + 2, device)
s2vt_model.load_state_dict(torch.load("best.pth"))

features = torch.tensor(features).unsqueeze(dim=0)
s2vt_model.eval()
with torch.no_grad():
    pred_caption = s2vt_model(features, word2idx=train_data.word2idx)

pred_caption = pred_caption.flatten()
captions = [train_data.idx2word[token_idx.item()] for token_idx in pred_caption]

print(captions)
print(f"Cleaned caption for <{feature_name}>:")
print(" ".join([word for word in captions if word not in ["<BOS>", "<EOS>"]]))
