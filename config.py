# Model config
IMAGE_FEATURE_EXTRACT_MODEL = "resnet"
HIDDEN_SIZE = 512
VOCAB_SIZE = 2500
if IMAGE_FEATURE_EXTRACT_MODEL == "mobilenet":
    IMAGE_FEATURE_SIZE = 1280
else:
    IMAGE_FEATURE_SIZE = 2048
CAPTION_MAX_LENGTH = 30

# Training config
BATCH_SIZE = 32
NUM_EPOCH = 30
TRAIN_RATIO = 0.9
FEATURES_DIR = f"data/YoutubeClips_features_{IMAGE_FEATURE_EXTRACT_MODEL}"
CAPTION_FILE = "data/AllVideoDescriptions.txt"
SPLIT_FILE = "data/train_split.txt"

print(f"Model config: \n"
      f"hidden_size: {HIDDEN_SIZE}, vocab_size: {VOCAB_SIZE}, image_feat_size: {IMAGE_FEATURE_SIZE},"
      f" cap_max_len: {CAPTION_MAX_LENGTH}")
print(f"Training config: \n"
      f"batch_size: {BATCH_SIZE}, num_epoch: {NUM_EPOCH}, feature_dir: {FEATURES_DIR},"
      f" caption_file: {CAPTION_FILE}, split_file: {SPLIT_FILE}")
