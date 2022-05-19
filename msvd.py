import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import os


class MSVD(Dataset):
    def __init__(self, features_dir, caption_path, max_vocab_size, caption_max_len, split=None):
        self.features_dir = features_dir
        self.max_vocab_size = max_vocab_size
        self.caption_max_len = caption_max_len
        self.video_ids = []

        if os.path.exists(features_dir):  # If not exists then maybe it's demo
            self.video_ids = [video_id.split(".")[0] for video_id in os.listdir(self.features_dir)]

        # Only choosing video_id in split
        if split is not None:
            with open(split) as file:
                lines = file.readlines()
                lines = {line.rstrip() for line in lines if line}

            self.video_ids = [video_id for video_id in self.video_ids if video_id in lines]

        captions = self.__load_caption(caption_path)
        self.word2idx, self.idx2word = self.__build_vocab(captions.values())

        # Creating video caption pair for each video and caption
        self.video_caption_pairs = []
        for video_id in self.video_ids:
            for label in captions[video_id]:
                self.video_caption_pairs.append((video_id, label))

    def __len__(self):
        return len(self.video_caption_pairs)

    def __getitem__(self, idx):
        video_id, caption = self.video_caption_pairs[idx]
        video_features = np.load(os.path.join(self.features_dir, video_id) + ".npy")
        video_features = torch.tensor(video_features)
        caption, caption_mask = self.__prepare_caption(caption)
        caption, caption_mask = torch.tensor(caption), torch.tensor(caption_mask)

        return video_features, caption, caption_mask

    # Load caption from files, remove caption which length > caption_max_len
    def __load_caption(self, labels_path):
        captions = {}
        print("Loading captions")
        with open(labels_path) as label_file:
            for line in label_file:
                if line.startswith("#") or len(line.strip()) == 0:
                    continue
                line = line.split()
                video_id, description = line[0], line[1:]
                # Some captions with length 30 just doesn't make sense, so we remove it from the caption set
                if len(description) > self.caption_max_len:
                    continue
                if video_id not in captions:
                    captions[video_id] = []
                captions[video_id].append(["<BOS>"] + description + ["<EOS>"])
        self.caption_max_len += 2  # to account for <BOS> and <EOS> token
        return captions

    # Create word count to get vocab with predefined max size. Create word <-> idx mapping
    def __build_vocab(self, caption_list):
        print("Building vocabulary from captions")
        word2idx = {}
        idx2word = {}
        word_count = {}
        for captions in caption_list:
            for caption in captions:
                for word in caption:
                    if word not in word_count:
                        word_count[word] = 0
                    word_count[word] += 1

        # sort dictionary by count
        word_count = {word: count for word, count in
                      sorted(word_count.items(), key=lambda item: item[1], reverse=True)}
        # Now we assume max_vocab_size is always less than vocab size
        for idx, word in enumerate(list(word_count.keys())[:self.max_vocab_size-1]):
            word2idx[word] = idx
            idx2word[idx] = word
        word2idx["<UNK>"] = self.max_vocab_size - 1
        idx2word[self.max_vocab_size - 1] = "<UNK>"

        return word2idx, idx2word

    # Prepare caption for training
    def __prepare_caption(self, caption):
        converted_caption = []
        # Pad caption to have desired length
        caption_length = len(caption)
        caption += ["<EOS>"] * (self.caption_max_len - caption_length)
        for word in caption:
            if word in self.word2idx:
                converted_caption.append(self.word2idx[word])
            else:
                converted_caption.append(self.word2idx["<UNK>"])

        # mask to signal our model to stop learning the padded EOS token
        caption_mask = [1] * caption_length + [0] * (self.caption_max_len - caption_length)
        return converted_caption, caption_mask


def split_train_test(dataset, output_dir="data/", test_ratio=0.1):
    all_videos = dataset.video_ids
    test_size = int(len(all_videos) * test_ratio)
    np.random.seed(2022)
    test_batch_idx = np.random.choice(len(all_videos), test_size, replace=False)
    train_ids = []
    test_ids = []
    for idx, video_id in enumerate(all_videos):
        if idx in test_batch_idx:
            test_ids.append(video_id)
        else:
            train_ids.append(video_id)

    with open(os.path.join(output_dir, "train_split.txt"), "wt") as f_out:
        for video_id in train_ids:
            f_out.write(video_id)
            f_out.write("\n")

    with open(os.path.join(output_dir, "test_split.txt"), "wt") as f_out:
        for video_id in test_ids:
            f_out.write(video_id)
            f_out.write("\n")


if __name__ == "__main__":
    data = MSVD("data/YoutubeClips_features", "data/AllVideoDescriptions.txt", 2500, 30, split="data/train_split.txt")

    # Run if we want to change train/test split
    # split_train_test(data)

    for features, cap, mask in tqdm(data):
        continue
