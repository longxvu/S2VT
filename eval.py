import torch
from torch.utils.data import DataLoader
from msvd import MSVD
from model import S2VT
from tqdm import tqdm
from nltk.translate.meteor_score import meteor_score
import config


def convert_pred_caption(tokens, idx2word):
    tokens = tokens.flatten().tolist()
    generated_caption = []
    for token in tokens:
        if (word := idx2word[token]) != "<EOS>":
            generated_caption.append(word)
        else:
            break
    return generated_caption[1:]  # Remove <BOS>


def convert_gt_caption(captions):
    return [caption[1:-1] for caption in captions]   # Remove <BOS>, <SOS>


def eval_collate_fn(data):
    assert len(data) == 1, "Currently only support batch size = 1"
    data = data[0]
    return data[0].unsqueeze(dim=0), data[1]


if __name__ == "__main__":
    weight_file = "best.pth"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = S2VT(config.IMAGE_FEATURE_SIZE, config.HIDDEN_SIZE, config.VOCAB_SIZE,
                 config.CAPTION_MAX_LENGTH, device=device)
    model.load_state_dict(torch.load("runs/epoch_20.pth"))
    model.to(device)

    # Using training vocabulary to decode predicted caption
    train_dataset = MSVD(config.FEATURES_DIR, config.CAPTION_FILE, config.VOCAB_SIZE,
                         config.CAPTION_MAX_LENGTH, config.TRAIN_SPLIT_FILE)
    test_dataset = MSVD(config.FEATURES_DIR, config.CAPTION_FILE, config.VOCAB_SIZE,
                        config.CAPTION_MAX_LENGTH, config.TEST_SPLIT_FILE, training=False)

    test_loader = DataLoader(test_dataset, collate_fn=eval_collate_fn)

    score = 0

    model.eval()
    with torch.no_grad():
        for vid_features, gt_captions in (pbar := tqdm(test_loader)):
            vid_features = vid_features.to(device)
            pred_caption = model(vid_features, word2idx=train_dataset.word2idx)
            pred_caption = convert_pred_caption(pred_caption, train_dataset.idx2word)
            gt_captions = convert_gt_caption(gt_captions)
            score += meteor_score(gt_captions, pred_caption)

    print(f"METEOR for test set: {score / len(test_loader)}")
