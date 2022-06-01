import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import os


class MSVD(Dataset):
    def __init__(self, features_dir, caption_path, caption_max_len, vocab, split=None, training=True):
        self.features_dir = features_dir
        self.caption_max_len = caption_max_len
        # demo uses training = False to get video ID
        self.training = training
        self.video_ids = []
        if os.path.exists(features_dir):  # If not exists then maybe it's demo
            self.video_ids = [video_id.split(".")[0] for video_id in os.listdir(self.features_dir)]

        self.vocab = vocab
        # Only choosing video_id in split
        if split is not None:
            with open(split) as file:
                lines = file.readlines()
                lines = {line.rstrip() for line in lines if line}

            self.video_ids = [video_id for video_id in self.video_ids if video_id in lines]

        print("Building caption data")
        self.video_caption_list = Vocab.load_caption(caption_path, caption_max_len)
        self.caption_max_len += 2  # Accounting for <BOS> and <EOS>

        # Creating video caption pair for each video and caption
        self.video_caption_pairs = []
        for video_id in self.video_ids:
            for label in self.video_caption_list[video_id]:
                self.video_caption_pairs.append((video_id, label))
                if self.training:
                    break  # One caption per video

        # Debugging stuff
        # np.random.seed(2022)
        # idx = np.random.choice(range(len(self.video_caption_pairs)), 64, replace=False)
        # self.video_caption_pairs = [self.video_caption_pairs[ids] for ids in idx]
        # print(self.video_caption_pairs)
        # self.video_ids = list({pair[0] for pair in self.video_caption_pairs})[:10]
        # print(self.video_ids)

    def __len__(self):
        if self.training:
            return len(self.video_caption_pairs)
        else:
            return len(self.video_ids)

    def __getitem__(self, idx):
        if self.training:
            video_id, caption = self.video_caption_pairs[idx]
            video_features = np.load(os.path.join(self.features_dir, video_id) + ".npy")
            video_features = torch.tensor(video_features)
            caption, caption_mask = self.__prepare_caption(caption)
            caption, caption_mask = torch.tensor(caption), torch.tensor(caption_mask)
            return video_features, caption, caption_mask
        else:
            video_id = self.video_ids[idx]
            video_features = np.load(os.path.join(self.features_dir, video_id) + ".npy")
            video_features = torch.tensor(video_features)
            return video_features, self.video_caption_list[video_id]

    # Prepare caption for training
    def __prepare_caption(self, caption):
        converted_caption = []
        # Pad caption to have desired length
        caption_length = len(caption)
        # Nice bug
        padded_caption = caption + ["<EOS>"] * (self.caption_max_len - caption_length)
        for word in padded_caption:
            if word in self.vocab.word2idx:
                converted_caption.append(self.vocab.word2idx[word])
            else:
                converted_caption.append(self.vocab.word2idx["<UNK>"])

        # mask to signal our model to stop learning the padded EOS token
        caption_mask = [1] * caption_length + [0] * (self.caption_max_len - caption_length)
        return converted_caption, caption_mask


# For all subset use training vocab only
class Vocab:
    def __init__(self, caption_file, max_vocab_size, caption_max_len, vocab_file="data/train_split.txt"):
        self.max_vocab_size = max_vocab_size

        with open(vocab_file) as file:
            lines = file.readlines()
            # videos id to extract vocab from
            self.vocab_videos_id = {line.rstrip() for line in lines if line}

        self._video_caption_list = self.load_caption(caption_file, caption_max_len)
        self._video_caption_list = {k: v for k, v in self._video_caption_list.items() if k in self.vocab_videos_id}
        self.word2idx, self.idx2word = self.__build_vocab(self._video_caption_list.values())

    # Load caption from files, remove caption which length > caption_max_len
    @staticmethod
    def load_caption(labels_path, caption_max_len):
        captions = {}
        with open(labels_path) as label_file:
            for line in label_file:
                if line.startswith("#") or len(line.strip()) == 0:
                    continue
                line = line.split()
                video_id, description = line[0], line[1:]
                # Some captions with length 30 just doesn't make sense, so we remove it from the caption set
                if len(description) > caption_max_len:
                    continue
                if video_id not in captions:
                    captions[video_id] = []
                captions[video_id].append(["<BOS>"] + [token.lower() for token in description] + ["<EOS>"])
        return captions

    # Create word count to get vocab with predefined max size. Create word <-> idx mapping
    def __build_vocab(self, caption_list):
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
        for idx, word in enumerate(list(word_count.keys())[:self.max_vocab_size - 1]):
            word2idx[word] = idx
            idx2word[idx] = word
        word2idx["<UNK>"] = self.max_vocab_size - 1
        idx2word[self.max_vocab_size - 1] = "<UNK>"

        return word2idx, idx2word


# Follows data split from the paper
def split_train_test(dataset, output_dir="data/", train_size=1200, test_size=670):
    all_videos = sorted(dataset.video_ids)

    train_ids = all_videos[:train_size]
    val_ids = all_videos[train_size:-test_size]
    test_ids = all_videos[-test_size:]
    print("Train set length:", len(train_ids))
    print("Validation set length:", len(val_ids))
    print("Test set length:", len(test_ids))

    subset_ids = [train_ids, val_ids, test_ids]
    names = ["train_split.txt", "val_split.txt", "test_split.txt"]

    for subset_id, name in zip(subset_ids, names):
        with open(os.path.join(output_dir, name), "wt") as f_out:
            for video_id in subset_id:
                f_out.write(video_id)
                f_out.write("\n")


if __name__ == "__main__":
    data = MSVD("data/YoutubeClips_features", "data/AllVideoDescriptions.txt", 5000, 30)

    # Run if we want to change train/test split
    # split_train_test(data)

    for features, cap, mask in tqdm(data):
        continue
