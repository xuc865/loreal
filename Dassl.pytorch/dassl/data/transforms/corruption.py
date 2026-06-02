import random
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
from io import BytesIO
import os 
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from scipy.ndimage import map_coordinates, gaussian_filter
 
class RandomCorruptionTransform: 
    def __init__(self, severity=5):  
        CORRS = [
            "gaussian_noise",
            "shot_noise",
            "impulse_noise",
            "speckle_noise",
            "defocus_blur",
            "glass_blur",
            "motion_blur",
            "zoom_blur",
            "snow",
            "frost",
            "fog",
            "brightness",
            "contrast",
            "jpeg_compression",
            "pixelate",
            "saturate",
            "elastic_transform" 
        ]
        self.corruption_names = CORRS
        self.default_severity = severity 

    def __call__(self, img):
        ctype = random.choice(self.corruption_names) 
        tx = CorruptionTransform(ctype, self.default_severity)
        return tx(img)

class CorruptionTransform: 
    def __init__(self, corruption_type: str, severity: int):
        self.corruption_type = corruption_type 
        self.severity = max(1, min(5, severity))

    def __call__(self, img: Image.Image) -> Image.Image: 
        c = self.corruption_type.lower()
        if c == "gaussian_noise":
            return self._gaussian_noise(img)
        elif c == "shot_noise":
            return self._shot_noise(img)
        elif c == "impulse_noise":
            return self._impulse_noise(img)
        elif c == "speckle_noise":
            return self._speckle_noise(img)
        elif c == "defocus_blur":
            return self._defocus_blur(img)
        elif c == "glass_blur":
            return self._glass_blur(img)
        elif c == "motion_blur":
            return self._motion_blur(img)
        elif c == "zoom_blur":
            return self._zoom_blur(img)
        elif c == "snow":
            return self._snow(img)
        elif c == "frost":
            return self._frost(img)
        elif c == "fog":
            return self._fog(img)
        elif c == "brightness":
            return self._brightness(img)
        elif c == "contrast":
            return self._contrast(img)
        elif c == "elastic_transform":
            return self._elastic_transform(img)
        elif c == "pixelate":
            return self._pixelate(img)
        elif c == "jpeg_compression" or c == "jpeg":
            return self._jpeg_compression(img)
        elif c == "saturate":
            return self._saturate(img)
        else: 
            return img 

    def _gaussian_noise(self, img: Image.Image) -> Image.Image:
        arr = np.array(img).astype(np.float32) / 255.0
        # sigma 随 severity 增大（可调系数）
        sigma = 0.05 * self.severity
        noise = np.random.randn(*arr.shape) * sigma
        arr = arr + noise
        arr = np.clip(arr, 0, 1)
        arr = (arr * 255).astype(np.uint8)
        return Image.fromarray(arr)

    def _shot_noise(self, img: Image.Image) -> Image.Image:
        # Shot / Poisson noise: 噪声方差 = 像素值本身
        arr = np.array(img).astype(np.float32) / 255.0
        # scale strength 与 severity 相关
        scale = self.severity * 0.04
        # 把像素视为期望值产生泊松噪声
        # 注意泊松只定义在非负整数，所以先 scale 再 poisson，再反 scale
        vals = arr * 255.0 * scale
        vals = np.random.poisson(vals) / scale
        vals = np.clip(vals / 255.0, 0, 1)
        arr = (vals * 255).astype(np.uint8)
        return Image.fromarray(arr)

    def _impulse_noise(self, img: Image.Image) -> Image.Image:
        # 又称 salt-and-pepper 噪声
        arr = np.array(img).astype(np.float32) / 255.0
        prob = 0.03 * self.severity  # 概率
        rnd = np.random.rand(*arr.shape[:2])
        mask_salt = rnd < (prob / 2)
        mask_pepper = (rnd >= (prob / 2)) & (rnd < prob)
        out = arr.copy()
        # 随机给每个通道置 1 或 0
        out[mask_salt] = 1.0
        out[mask_pepper] = 0.0
        out = np.clip(out, 0, 1)
        out = (out * 255).astype(np.uint8)
        return Image.fromarray(out)

    def _speckle_noise(self, img: Image.Image) -> Image.Image:
        arr = np.array(img).astype(np.float32) / 255.0
        # multiplicative noise: arr + arr * noise
        sigma = 0.05 * self.severity
        noise = np.random.randn(*arr.shape) * sigma
        arr = arr + arr * noise
        arr = np.clip(arr, 0, 1)
        arr = (arr * 255).astype(np.uint8)
        return Image.fromarray(arr)

    def _defocus_blur(self, img: Image.Image) -> Image.Image:
        # 用 PIL 的 GaussianBlur 近似 defocus blur
        # radius 与 severity 关联
        radius = self.severity * 1.5
        return img.filter(ImageFilter.GaussianBlur(radius=radius))

    def _glass_blur(self, img: Image.Image) -> Image.Image: 
        arr = np.array(img).astype(np.uint8) 
        radius = 0.5 * self.severity
        pil = img.filter(ImageFilter.GaussianBlur(radius=radius))
        arr_blur = np.array(pil).astype(np.uint8) 
        h, w, c = arr_blur.shape 
        max_shift = int(self.severity * 2)
        out = arr_blur.copy()
        for _ in range(int(h * w * 0.01 * self.severity)):
            i = random.randint(0, h - 1)
            j = random.randint(0, w - 1)
            di = random.randint(-max_shift, max_shift)
            dj = random.randint(-max_shift, max_shift)
            ni = np.clip(i + di, 0, h - 1)
            nj = np.clip(j + dj, 0, w - 1)
            out[i, j] = arr_blur[ni, nj]
        return Image.fromarray(out)

    def _motion_blur(self, img: Image.Image) -> Image.Image: 
        radius = self.severity * 1.5
        return img.filter(ImageFilter.GaussianBlur(radius=radius))

    def _zoom_blur(self, img: Image.Image) -> Image.Image: 
        width, height = img.size
        out = np.zeros((height, width, 3), dtype=np.float32)
        num_steps = self.severity * 2
        alpha = 1.0 / (num_steps + 1)
        arr_org = np.array(img).astype(np.float32)
        out += arr_org * alpha
        for i in range(1, num_steps + 1):
            factor = 1.0 - i * 0.02 * self.severity
            if factor <= 0:
                continue
            new_w = int(width * factor)
            new_h = int(height * factor)
            im2 = img.resize((new_w, new_h), Image.BILINEAR)
            im2 = im2.resize((width, height), Image.BILINEAR)
            arr2 = np.array(im2).astype(np.float32)
            out += arr2 * alpha
        out = np.clip(out, 0, 255).astype(np.uint8)
        return Image.fromarray(out)

    def _snow(self, img: Image.Image) -> Image.Image: 
        arr = np.array(img).astype(np.float32) / 255.0
        h, w, c = arr.shape 
        num_snow = int((h * w) * 0.0005 * self.severity)
        for _ in range(num_snow):
            x = random.randint(0, w - 1)
            y = random.randint(0, h - 1)
            arr[y, x, :] = 1.0  
        img2 = Image.fromarray((arr * 255).astype(np.uint8))
        img2 = img2.filter(ImageFilter.GaussianBlur(radius=self.severity * 0.3))
        return img2

    def _frost(self, img: Image.Image) -> Image.Image: 
        arr = np.array(img).astype(np.float32) / 255.0
        h, w, c = arr.shape 
        noise = np.random.randn(h, w, c).astype(np.float32) 
        yy = np.abs(np.linspace(-1, 1, h))[:, None]
        xx = np.abs(np.linspace(-1, 1, w))[None, :]
        weight = (yy + xx) / 2.0
        weight = np.expand_dims(weight, axis=2)
        arr = arr * (1 - weight * 0.5 * self.severity) + noise * (0.2 * self.severity) * weight
        arr = np.clip(arr, 0, 1)
        arr = (arr * 255).astype(np.uint8) 
        pil = Image.fromarray(arr)
        pil = pil.filter(ImageFilter.GaussianBlur(radius=self.severity * 0.2))
        return pil

    def _fog(self, img: Image.Image) -> Image.Image: 
        arr = np.array(img).astype(np.float32) / 255.0
        h, w, c = arr.shape 
        fog_strength = 0.5 * (self.severity / 5.0) 
        white = np.ones((h, w, c), dtype=np.float32)
        arr = arr * (1 - fog_strength) + white * fog_strength 
        pil = Image.fromarray((arr * 255).astype(np.uint8))
        pil = pil.filter(ImageFilter.GaussianBlur(radius=self.severity * 2))
        return pil

    def _brightness(self, img: Image.Image) -> Image.Image: 
        factor = 1.0 + 0.3 * (self.severity - 3)  
        return ImageEnhance.Brightness(img).enhance(factor)

    def _contrast(self, img: Image.Image) -> Image.Image: 
        factor = 1.0 + 0.3 * (self.severity - 3)
        return ImageEnhance.Contrast(img).enhance(factor)

    def _elastic_transform(self, img: Image.Image) -> Image.Image:
        arr = np.array(img).astype(np.float32)
        h, w, c = arr.shape
 
        magnitude = self.severity * 5
        dx = (np.random.rand(h, w) * 2 - 1) * magnitude
        dy = (np.random.rand(h, w) * 2 - 1) * magnitude 
        dx = gaussian_filter(dx, sigma=self.severity)
        dy = gaussian_filter(dy, sigma=self.severity)
 
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        indices_x = (y + dy).reshape(-1)
        indices_y = (x + dx).reshape(-1)

        out = np.zeros_like(arr) 
        for ch in range(c):
            channel = arr[..., ch]
            warped = map_coordinates(channel, [indices_x, indices_y], order=1, mode='reflect')
            out[..., ch] = warped.reshape((h, w))

        out = np.clip(out, 0, 255).astype(np.uint8)
        return Image.fromarray(out)

    def _pixelate(self, img: Image.Image) -> Image.Image:
        # 像素化：把图像缩小后再拉大
        width, height = img.size
        # 缩小比例
        factor = 1 + self.severity * 1  # severity 越大缩小比例越大
        new_w = max(1, width // factor)
        new_h = max(1, height // factor)
        img_small = img.resize((new_w, new_h), Image.NEAREST)
        img_large = img_small.resize((width, height), Image.NEAREST)
        return img_large

    def _jpeg_compression(self, img: Image.Image) -> Image.Image:
        buffer = BytesIO()
        # 质量 = 100 - severity * 15 （可调）
        quality = max(5, 100 - self.severity * 15)
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer)

    def _saturate(self, img: Image.Image) -> Image.Image:
        factor = 1.0 + 0.3 * (self.severity - 3)
        return ImageEnhance.Color(img).enhance(factor)


 
def visualize_corruptions(img_path, corruption_types, severity=3, out_path=None): 
    img = Image.open(img_path).convert("RGB")

    n = len(corruption_types) + 1  
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    axes = axes.flatten()
 
    axes[0].imshow(img)
    axes[0].axis("off")
    axes[0].set_title("original")
 
    for i, ctype in enumerate(corruption_types):
        tx = CorruptionTransform(ctype, severity)
        img_c = tx(img)
        axes[i+1].imshow(img_c)
        axes[i+1].axis("off")
        axes[i+1].set_title(f"{ctype}, s={severity}")
 
    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path)
    plt.show()

if __name__ == "__main__":
    sample_path = "/root/dummy.jpg"

    corruption_types = [
        "gaussian_noise",
        "shot_noise",
        "impulse_noise",
        "speckle_noise",
        "defocus_blur",
        "glass_blur",
        "motion_blur",
        "zoom_blur",
        "frost",
        "snow",
        "fog",
        "brightness",
        "contrast",
        "jpeg_compression",
        "pixelate",
        "saturate",
        "elastic_transform" 
    ]
 
    subset = corruption_types 
    visualize_corruptions(sample_path, subset, severity=5, out_path="/root/res.jpg")
