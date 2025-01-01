import cv2
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models.video import r3d_18, R3D_18_Weights, MC3_18_Weights, mc3_18
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights, s3d, S3D_Weights
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights, mvit_v1_b, MViT_V1_B_Weights

class FeatureExtractor:
    def __init__(self, model_type='r3d_18', device='cpu'):
        self.model_type = model_type
        self.device = device
        
        if model_type == 'r3d_18':
            self.model = r3d_18(weights=R3D_18_Weights.DEFAULT)
            self.model.fc = torch.nn.Identity()
        elif model_type == 'mc3_18':
            self.model = mc3_18(weights=MC3_18_Weights.DEFAULT)
            self.model.fc = torch.nn.Identity()
        elif model_type == 'r2plus1d_18':
            self.model = r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT)
            self.model.fc = torch.nn.Identity()
        elif model_type == 's3d':
            self.model = s3d(weights=S3D_Weights.DEFAULT)
            self.model.classifier = torch.nn.Identity()
        elif model_type == 'mvit_v2_s':
            self.model = mvit_v2_s(weights=MViT_V2_S_Weights.DEFAULT)
            self.model.head = torch.nn.Identity()
        elif model_type == 'mvit_v1_b':
            self.model = mvit_v1_b(weights=MViT_V1_B_Weights.DEFAULT)
            self.model.head = torch.nn.Identity()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        self.model = self.model.to(device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], 
                              std=[0.22803, 0.22145, 0.216989])
        ])

    def preprocess_frames(self, frames):
        processed_frames = []
        for frame in frames:
            pil_image = Image.fromarray(frame)
            processed_frame = self.transform(pil_image)
            processed_frames.append(processed_frame)
        
        frames_tensor = torch.stack(processed_frames)

        return frames_tensor.permute(1, 0, 2, 3)

    def extract_features(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []

        for frame_idx in range(63, 88):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()

        if len(frames) != 25:
            raise ValueError(f"Expected 25 frames, got {len(frames)}")
        
        frames_tensor = self.preprocess_frames(frames).unsqueeze(0).to(self.device)
        
        # TODO: Fix for mvit models
        print(f"Shape of frames_tensor: {frames_tensor.shape}")
        
        with torch.no_grad():
            features = self.model(frames_tensor)
            
        return features

def main():
    video_path = 'data/dataset/train/action_0/clip_0.mp4'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for model_type in ['r3d_18', 'mc3_18', 'r2plus1d_18', 's3d', 'mvit_v2_s', 'mvit_v1_b']:
        extractor = FeatureExtractor(model_type=model_type, device=device)
        features = extractor.extract_features(video_path)
        print(f"{model_type} features shape: {features.shape}")

if __name__ == '__main__':
    main()