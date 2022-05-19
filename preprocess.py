import os
import cv2 as cv
import numpy as np
import glob
import torch
from torch import nn
from tqdm import tqdm
from torchvision.transforms import Normalize, ToTensor, Compose


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        model = torch.hub.load("pytorch/vision:v0.12.0", 'mobilenet_v2', pretrained=True)
        self.features = model.features
        self.pool = nn.AvgPool2d(7)

    def forward(self, x):
        out = self.features(x)
        out = self.pool(out)
        out = out.view(-1, 1280)
        return out


# Given video path, extract video into images, and resize if necessary
def extract_images(video_path, num_frame_per_video, image_resize=None):
    # Read video
    frames = []
    cap = cv.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    # Skip if somehow video decoding has errors
    if not frames:
        return

    selected_indices = np.linspace(0, len(frames) - 1, num_frame_per_video).round().astype(int)
    frames = [frames[idx] for idx in selected_indices]
    final_frames = []
    for frame in frames:
        if image_resize:
            frame = cv.resize(frame, image_resize)
        final_frames.append(frame)

    return final_frames


def extract_all_images(videos_path, output_dir, num_frame_per_video=80, image_size=(224, 224)):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Extracting videos into images from {videos_path}")

    for video_path in tqdm(glob.glob(os.path.join(videos_path, "*"))):
        extracted_frames = extract_images(video_path, num_frame_per_video, image_size)

        _, video_id = os.path.split(video_path)
        video_id = video_id.split(".")[0]
        images_output_dir = os.path.join(output_dir, video_id)
        os.makedirs(images_output_dir, exist_ok=True)

        for idx, frame in enumerate(extracted_frames):
            cv.imwrite(os.path.join(images_output_dir, f"{idx:03}.png"), frame)


def extract_features_images(images, model, transform, device="cpu"):
    transformed_images = []
    for image in images:
        transformed_images.append(transform(image))
    transformed_images = torch.stack(transformed_images)  # Now transformed images has size: (80, 3, 224, 224)
    transformed_images = transformed_images.to(device)
    features = model(transformed_images)
    return features


def extract_all_features(images_dir, features_output):
    # Create output dir and define transformation from numpy -> pytorch
    os.makedirs(features_output, exist_ok=True)
    transforms = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Feature extractor model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feature_extractor_model = FeatureExtractor()
    feature_extractor_model.eval()
    feature_extractor_model.to(device)

    # Extract features and saved into numpy images
    with torch.no_grad():
        for video_id in tqdm(os.listdir(images_dir)):
            video_path = os.path.join(images_dir, video_id)
            images = [cv.imread(image) for image in sorted(glob.glob(os.path.join(video_path, "*")))]
            features = extract_features_images(images, feature_extractor_model, transforms, device)
            features = features.cpu().numpy()
            np.save(os.path.join(features_output, f"{video_id}.npy"), features)


if __name__ == "__main__":
    # Videos -> images
    extract_all_images("data/YouTubeClips", "data/YoutubeClips_images")

    # Images -> Features
    extract_all_features("data/YoutubeClips_images", "data/YoutubeClips_features")
