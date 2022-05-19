import torch
from torch import nn


class S2VT(nn.Module):
    def __init__(self, image_feature_size=1280, hidden_size=512, vocab_size=1500, caption_max_len=30, device="cpu"):
        super(S2VT, self).__init__()
        self.image_feature_size = image_feature_size
        self.caption_max_len = caption_max_len
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.device = device
        self.decoder_input_size = 2 * self.hidden_size  # output of image encoder + word embedding size

        self.lstm_v2h = nn.LSTM(self.image_feature_size, self.hidden_size, batch_first=True)
        self.lstm_h2c = nn.LSTM(self.decoder_input_size, self.hidden_size, batch_first=True)

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size, self.vocab_size)

    # caption for training, or index of <BOS> for inference
    def forward(self, video_features, caption=None, word2idx=None):
        # Video features -> hidden layer
        batch_size, num_video_features = video_features.size(0), video_features.size(1)
        video_padding = torch.zeros((batch_size, self.caption_max_len - 1, self.image_feature_size), device=self.device)
        padded_video_features = torch.cat([video_features, video_padding], dim=1)
        video_out, video_hidden = self.lstm_v2h(padded_video_features)

        if self.training:
            # Encoded videos feature -> caption decoder
            # caption input is from <BOS> to seq length -1
            embedded_caption = self.embedding(caption[:, :-1])          # size: (B, 31, 512)
            caption_padding = torch.zeros((batch_size, num_video_features, self.hidden_size), device=self.device)
            # Append empty embedding state (video part) before captions
            decoder_input = torch.cat([caption_padding, embedded_caption], dim=1)
            # Stack video feature and word embedding
            decoder_input = torch.cat([video_out, decoder_input], dim=2)
            caption_out, _ = self.lstm_h2c(decoder_input)
            caption_out = caption_out[:, num_video_features:]
            caption_out = self.linear(caption_out)
            caption_out = torch.permute(caption_out, (0, 2, 1))  # To follow cross entropy loss: (B, C, k...) format
            return caption_out
        else:
            caption_padding = torch.zeros((batch_size, num_video_features, self.hidden_size), device=self.device)
            vid_feature_input = torch.cat([video_out[:, :num_video_features, :], caption_padding], dim=2)
            # Video features -> captions
            caption_out, caption_hidden = self.lstm_h2c(vid_feature_input)
            # feeds <BOS> into LSTM
            bos_tensor = torch.full([batch_size, 1], word2idx["<BOS>"], device=self.device)
            results = bos_tensor.clone()

            bos_tensor = self.embedding(bos_tensor)
            bos_tensor = torch.cat([video_out[:, num_video_features, :].unsqueeze(dim=1), bos_tensor], dim=2)
            next_word, next_hidden = self.lstm_h2c(bos_tensor, caption_hidden)

            for idx in range(self.caption_max_len - 2):
                next_word = self.linear(next_word)
                word_out = next_word.max(dim=2)[1]
                results = torch.hstack([results, word_out])
                word_out = self.embedding(word_out)

                # concat previous word embedding with video features
                next_word = torch.cat([video_out[:, num_video_features+idx+1, :].unsqueeze(dim=1), word_out], dim=2)
                next_word, next_hidden = self.lstm_h2c(next_word, next_hidden)

            return results


# random_features = torch.ones((1, 80, 1024))
# random_caption = torch.ones((1, 30), dtype=torch.int)
# model = S2VT()
#
# print(model)
#
# output = model(random_features, random_caption)
# print(output.size())

