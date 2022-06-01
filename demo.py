import torch
from model import S2VT
from msvd import MSVD, Vocab
from preprocess import FeatureExtractor, extract_images, extract_images_features, image_transform
from eval import convert_gt_caption, convert_pred_caption
from nltk.translate.meteor_score import meteor_score
import cv2 as cv
import config


# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"

# Define feature extractor
feature_extractor_model = FeatureExtractor()
feature_extractor_model.eval()
feature_extractor_model.to(device)


video_id = "r0rmrbTb7fU_98_109"
video_path = f"data/YoutubeClips/{video_id}.avi"
extracted_frames = extract_images(video_path)
features = extract_images_features(extracted_frames, feature_extractor_model, image_transform, device)

vocab = Vocab(config.CAPTION_FILE, config.VOCAB_SIZE, config.CAPTION_MAX_LENGTH)
train_data = MSVD(config.FEATURES_DIR, config.CAPTION_FILE, config.CAPTION_MAX_LENGTH, vocab, training=False)

# device = "cuda" if torch.cuda.is_available() else "cpu"
s2vt_model = S2VT(config.IMAGE_FEATURE_SIZE, config.HIDDEN_SIZE, config.VOCAB_SIZE,
                  config.CAPTION_MAX_LENGTH + 2, device)

checkpoint = torch.load("runs/last_resnet_440.pth")
s2vt_model.load_state_dict(checkpoint["model"])
s2vt_model.to(device)

features = features.unsqueeze(dim=0)
s2vt_model.eval()
with torch.no_grad():
    pred_caption = s2vt_model(features, word2idx=vocab.word2idx)

print(f"Predicted caption for <{video_id}>:")
pred_caption = convert_pred_caption(pred_caption, vocab.idx2word)
print(" ".join(pred_caption))

print("All ground truth captions:")
gt_caption = convert_gt_caption(train_data.video_caption_list[video_id])
print([" ".join(cap) for cap in gt_caption])

print(f"Meteor score: {meteor_score(gt_caption, pred_caption)}")

cap = cv.VideoCapture(video_path)
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        cv.imshow(f"{video_id}", frame)
        if cv.waitKey(30) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv.destroyAllWindows()
