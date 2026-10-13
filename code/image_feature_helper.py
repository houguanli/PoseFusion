import os
import numpy as np
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from skimage import transform
from project_paths import ROOT

# ----------- CNN Rotation-Sensitive Feature Extractor -----------
class ResNetFeatureExtractor(torch.nn.Module):
    def __init__(self, layer="layer3"):
        super().__init__()
        weight_path = ROOT / 'dependencies/resnet18-f37072fd.pth'
        if not weight_path.is_file():
            raise FileNotFoundError(f'Missing bundled ResNet18 weights: {weight_path}')
        resnet = models.resnet18(weights=None)
        resnet.load_state_dict(torch.load(weight_path, map_location='cpu'))
        self.model = torch.nn.Sequential()
        for name, module in resnet.named_children():
            self.model.add_module(name, module)
            if name == layer:
                break

    def forward(self, x):
        return self.model(x)

# ------------------ Preprocessing ------------------
def read_rgb_image(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0

# ------------------ CNN Feature Extract ------------------
def extract_rotation_sensitive_feature(image, device='cpu'):
    extractor = ResNetFeatureExtractor(layer="layer3").to(device).eval()
    transform_pipeline = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img_t = transform_pipeline(image).unsqueeze(0).to(device)

    with torch.no_grad():
        feat = extractor(img_t).squeeze().cpu().numpy()

    return feat


# ------------------ Similarity ------------------
def cosine_similarity(f1, f2):
    f1 = f1.flatten()
    f2 = f2.flatten()
    return np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2) + 1e-8)


def _as_rgb_uint8(image):
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    image = np.asarray(image)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    if image.shape[-1] == 4:
        image = image[..., :3]
    if image.dtype != np.uint8:
        if image.max() <= 1.5:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def hsv_histogram(image, h_bins=36, s_bins=16, v_bins=16):
    rgb = _as_rgb_uint8(image)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h = hsv[..., 0].reshape(-1)
    s = hsv[..., 1].reshape(-1)
    v = hsv[..., 2].reshape(-1)

    # Downweight black background and nearly gray pixels. This keeps rendered
    # empty space from dominating OBB candidate appearance scoring.
    weights = ((v > 12).astype(np.float32) * (0.25 + 0.75 * (s.astype(np.float32) / 255.0))).reshape(-1)
    if float(weights.sum()) < 1e-6:
        weights = np.ones_like(weights, dtype=np.float32)

    h_hist, _ = np.histogram(h, bins=h_bins, range=(0, 180), weights=weights, density=False)
    s_hist, _ = np.histogram(s, bins=s_bins, range=(0, 256), weights=weights, density=False)
    v_hist, _ = np.histogram(v, bins=v_bins, range=(0, 256), weights=weights, density=False)
    hist = np.concatenate([h_hist, s_hist, v_hist]).astype(np.float32)
    hist /= float(hist.sum() + 1e-8)
    return hist


def histogram_chi_square(h1, h2):
    return float(0.5 * np.sum(((h1 - h2) ** 2) / (h1 + h2 + 1e-8)))


def hsv_hist_diff(img1, img2):
    return histogram_chi_square(hsv_histogram(img1), hsv_histogram(img2))

# ------------------ Log-Polar Feature ------------------
def to_log_polar(image):
    center = (image.shape[0] // 2, image.shape[1] // 2)
    radius = np.hypot(*center)
    return transform.warp_polar(image, center=center, radius=radius, scaling='log')

def normalized_cross_correlation(f1, f2):
    f1 = (f1 - np.mean(f1)) / (np.std(f1) + 1e-8)
    f2 = (f2 - np.mean(f2)) / (np.std(f2) + 1e-8)
    return np.sum(f1 * f2) / (np.sqrt(np.sum(f1**2)) * np.sqrt(np.sum(f2**2)))

# ------------------ Loss ------------------
def rotation_sensitive_loss(img1_path, img2_path):
    img1 = read_rgb_image(img1_path)
    img2 = read_rgb_image(img2_path)

    img1 = cv2.cvtColor((img1 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255.0
    img2 = cv2.cvtColor((img2 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255.0

    img1 = transform.resize(img1, (1024, 1024))
    img2 = transform.resize(img2, (1024, 1024))

    lp1 = to_log_polar(img1)
    lp2 = to_log_polar(img2)

    ncc = normalized_cross_correlation(lp1, lp2)
    loss = 1 - ncc
    return loss

# ------------------ CNN Color Feature Difference ------------------
def color_feature_difference(img1, img2, device='cpu'):
    feat1 = extract_rotation_sensitive_feature(img1, device=device)
    feat2 = extract_rotation_sensitive_feature(img2, device=device)
    sim = cosine_similarity(feat1, feat2)
    resnet_diff = 1 - sim
    hsv_diff = hsv_hist_diff(img1, img2)
    return 0.65 * hsv_diff + 0.35 * resnet_diff

# ------------------ Batch Compare ------------------
def batch_compare(folder1, folder2, output_csv="loss_results.csv", top_k=10):
    images1 = [os.path.join(folder1, f) for f in os.listdir(folder1) if f.lower().endswith('.png')]
    images2 = [os.path.join(folder2, f) for f in os.listdir(folder2) if f.lower().endswith('.png')]

    results = []

    for img1 in images1:
        for img2 in images2:
            loss = rotation_sensitive_loss(img1, img2)
            results.append([img1, img2, loss])
            print("cmp: ", img1, " ", img2)

    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Image1", "Image2", "Loss"])
        writer.writerows(results)

    print(f"CSV saved to {output_csv}")

    loss_matrix = np.array([[r[2] for r in results if os.path.basename(r[0]) == os.path.basename(img1)] for img1 in images1])
    plt.figure(figsize=(10, 8))
    sns.heatmap(loss_matrix, xticklabels=[os.path.basename(p) for p in images2], yticklabels=[os.path.basename(p) for p in images1], annot=True, fmt=".3f", cmap="viridis")
    plt.title("Rotation Sensitive Loss Heatmap")
    plt.xlabel("Folder2 Images")
    plt.ylabel("Folder1 Images")
    plt.tight_layout()
    plt.savefig("loss_heatmap.png")
    print("Heatmap saved to loss_heatmap.png")

if __name__ == "__main__":
    # ====== Default for debug ======
    folder1 = "C:/Users/guanl/Desktop/reg/public_data/bunny_pose1/image"
    folder2 = "C:/Users/guanl/Desktop/reg/public_data/bunny_pose2/image"
    top_k = 10
    batch_compare(folder1, folder2, top_k=top_k)
    # Example usage
    path1 = "C:/Users/guanl/Desktop/reg/exp/test_tmp/stand.png"
    path2 = "C:/Users/guanl/Desktop/reg/exp/test_tmp/near2.png"

    loss = rotation_sensitive_loss(path1, path2)
    print("Rotation Sensitive Loss:", loss)
