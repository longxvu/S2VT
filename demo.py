import torch
from model import S2VT
from msvd import MSVD
from preprocess import FeatureExtractor, extract_images, extract_images_features, image_transform
from eval import convert_gt_caption, convert_pred_caption
from nltk.translate.meteor_score import meteor_score
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

train_data = MSVD(config.FEATURES_DIR, config.CAPTION_FILE, config.VOCAB_SIZE,
                  config.CAPTION_MAX_LENGTH, split=config.TRAIN_SPLIT_FILE, training=False)

# device = "cuda" if torch.cuda.is_available() else "cpu"
s2vt_model = S2VT(config.IMAGE_FEATURE_SIZE, config.HIDDEN_SIZE, config.VOCAB_SIZE,
                  config.CAPTION_MAX_LENGTH + 2, device)
s2vt_model.load_state_dict(torch.load("runs/epoch_20.pth"))
s2vt_model.to(device)

features = features.unsqueeze(dim=0)
s2vt_model.eval()
with torch.no_grad():
    pred_caption = s2vt_model(features, word2idx=train_data.word2idx)

print(f"Cleaned caption for <{video_id}>:")
pred_caption = convert_pred_caption(pred_caption, train_data.idx2word)
print(pred_caption)

print("All caption:")
gt_caption = convert_gt_caption(train_data.video_caption_list[video_id])
print(gt_caption)

print(f"Meteor score: {meteor_score(gt_caption, pred_caption)}")
