import torch
from model import S2VT
from msvd import MSVD
from preprocess import FeatureExtractor, extract_images, extract_images_features, image_transform
import config


# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"

# Define feature extractor
feature_extractor_model = FeatureExtractor()
feature_extractor_model.eval()
feature_extractor_model.to(device)

video_id = "edqyq4Q-7uU_103_109"
video_path = f"data/YoutubeClips/{video_id}.avi"
extracted_frames = extract_images(video_path)
features = extract_images_features(extracted_frames, feature_extractor_model, image_transform, device)

train_data = MSVD(config.FEATURES_DIR, config.CAPTION_FILE, config.VOCAB_SIZE, config.CAPTION_MAX_LENGTH)

# device = "cuda" if torch.cuda.is_available() else "cpu"
s2vt_model = S2VT(config.IMAGE_FEATURE_SIZE, config.HIDDEN_SIZE, config.VOCAB_SIZE,
                  config.CAPTION_MAX_LENGTH + 2, device)
s2vt_model.load_state_dict(torch.load("runs/best_mbv2_1500.pth"))
s2vt_model.to(device)

features = features.unsqueeze(dim=0)
s2vt_model.eval()
with torch.no_grad():
    pred_caption = s2vt_model(features, word2idx=train_data.word2idx)

pred_caption = pred_caption.flatten()
captions = [train_data.idx2word[token_idx.item()] for token_idx in pred_caption]

print(captions)
print(f"Cleaned caption for <{video_id}>:")
print(" ".join([word for word in captions if word not in ["<BOS>", "<EOS>"]]))
