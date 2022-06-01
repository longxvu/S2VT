import os.path

import torch
from torch.utils.data import DataLoader
from msvd import MSVD, Vocab
from model import S2VT
from tqdm import tqdm
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import corpus_bleu
import config


# Convert sequence of idx into a list of word, removing BOS and EOS
def convert_pred_caption(tokens, idx2word):
    tokens = tokens.flatten().tolist()
    generated_caption = []
    for token in tokens:
        if (word := idx2word[token]) != "<EOS>":
            generated_caption.append(word)
        else:
            break
    if not generated_caption:
        return []
    if generated_caption[0] == "<BOS>":
        generated_caption = generated_caption[1:]
    return generated_caption


# Assuming input is a list of captions, with BOS and EOS added
def convert_gt_caption(captions):
    return [cap[1:-1] for cap in captions]   # Remove <BOS>, <EOS>


def eval_collate_fn(data):
    assert len(data) == 1, "Currently only support batch size = 1"
    return_data = data[0]
    return return_data[0].unsqueeze(dim=0), return_data[1]


if __name__ == "__main__":
    # Change weight file to another file for different model
    weight_file = "runs/last_resnet_440.pth"
    if not os.path.exists(weight_file):
        print(f"Weight file {weight_file} doesn't exists")
        exit()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = S2VT(config.IMAGE_FEATURE_SIZE, config.HIDDEN_SIZE, config.VOCAB_SIZE,
                 config.CAPTION_MAX_LENGTH + 2, device)

    checkpoint = torch.load(weight_file)
    model.load_state_dict(checkpoint["model"])
    model.to(device)

    # Using training vocabulary to decode predicted caption
    print("Loading vocab")
    vocab = Vocab(config.CAPTION_FILE, config.VOCAB_SIZE, config.CAPTION_MAX_LENGTH)

    # Change config.TEST_SPLIT_FILE to config.TRAIN_SPLIT_FILE or config.VAL_SPLIT_FILE to run and get
    # evaluation results on train or val dataset instead
    eval_dataset = MSVD(config.FEATURES_DIR, config.CAPTION_FILE, config.CAPTION_MAX_LENGTH, vocab,
                        config.TEST_SPLIT_FILE, training=False)

    eval_loader = DataLoader(eval_dataset, collate_fn=eval_collate_fn)

    all_meteor_scores = []
    all_predicted_captions = []
    all_gt_captions = []

    model.eval()
    with torch.no_grad():
        for vid_features, gt_captions in (pbar := tqdm(eval_loader)):
            vid_features = vid_features.to(device)
            pred_caption = model(vid_features, word2idx=vocab.word2idx)
            pred_caption = convert_pred_caption(pred_caption, vocab.idx2word)
            gt_captions = convert_gt_caption(gt_captions)
            all_meteor_scores.append(meteor_score(gt_captions, pred_caption))
            # For BLEU calculation
            all_predicted_captions.append(pred_caption)
            all_gt_captions.append(gt_captions)

    meteor_score = sum(all_meteor_scores) / len(eval_loader)
    # BLEU-1 to BLEU-4
    bleu_scores = corpus_bleu(all_gt_captions, all_predicted_captions,
                              weights=[(1., 0), (0.5, 0.5), (0.333, 0.333, 0.334), (0.25, 0.25, 0.25, 0.25)])

    print(f"METEOR score: {meteor_score}")
    print(f"BLEU score: {bleu_scores}")

    video_score_map = [(eval_dataset.video_ids[idx], all_meteor_scores[idx], all_predicted_captions[idx])
                       for idx in range(len(eval_dataset))]
    video_score_map = list(sorted(video_score_map, key=lambda x: x[1], reverse=True))
    print("10 videos with highest score: ")
    print(video_score_map[:10])

    # Write prediction to file
    with open("eval_result.txt", "wt") as f:
        f.write(f"METEOR score: {meteor_score}\n")
        f.write(f"BLEU[1-4] scores: {bleu_scores}\n")

        for video_id, score, caption in video_score_map:
            f.write(f"{video_id} {score:.4f} {' '.join(caption)}\n")

    print("Evaluation result written to eval_result.txt")
