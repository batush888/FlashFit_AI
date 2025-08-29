import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2
from typing import Tuple, List, Union, Optional
import torch
import torchvision.transforms as transforms
from sklearn.cluster import KMeans
import os

class ImagePreprocessor:
    """
    图像预处理器 - 为CLIP和分类器准备图像
    """
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size = target_size
        
        # CLIP标准预处理
        self.clip_transform = transforms.Compose([
            transforms.Resize(target_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])
        
        # 分类器预处理
        self.classifier_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        print("图像预处理器初始化完成")
    
    def load_and_validate_image(self, image_path: str) -> Image.Image:
        """
        加载并验证图像
        
        Args:
            image_path: 图像路径
            
        Returns:
            PIL图像对象
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
        
        try:
            image = Image.open(image_path)
            
            # 转换为RGB格式
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # 验证图像尺寸
            width, height = image.size
            if width < 32 or height < 32:
                raise ValueError(f"图像尺寸太小: {width}x{height}")
            
            if width > 4096 or height > 4096:
                # 如果图像太大，先缩放
                max_size = 2048
                ratio = min(max_size / width, max_size / height)
                new_size = (int(width * ratio), int(height * ratio))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            return image
            
        except Exception as e:
            raise ValueError(f"无法加载图像: {str(e)}")
    
    def enhance_image_quality(self, image: Image.Image, 
                            enhance_contrast: bool = True,
                            enhance_sharpness: bool = True,
                            enhance_color: bool = True) -> Image.Image:
        """
        增强图像质量
        
        Args:
            image: 输入图像
            enhance_contrast: 是否增强对比度
            enhance_sharpness: 是否增强锐度
            enhance_color: 是否增强色彩
            
        Returns:
            增强后的图像
        """
        enhanced_image = image.copy()
        
        if enhance_contrast:
            enhancer = ImageEnhance.Contrast(enhanced_image)
            enhanced_image = enhancer.enhance(1.1)  # 轻微增强对比度
        
        if enhance_sharpness:
            enhancer = ImageEnhance.Sharpness(enhanced_image)
            enhanced_image = enhancer.enhance(1.1)  # 轻微增强锐度
        
        if enhance_color:
            enhancer = ImageEnhance.Color(enhanced_image)
            enhanced_image = enhancer.enhance(1.05)  # 轻微增强色彩饱和度
        
        return enhanced_image
    
    def remove_background(self, image: Image.Image, method: str = "simple") -> Image.Image:
        """
        简单的背景移除（MVP版本）
        
        Args:
            image: 输入图像
            method: 背景移除方法
            
        Returns:
            处理后的图像
        """
        if method == "simple":
            # 简单的边缘检测和形态学操作
            img_array = np.array(image)
            
            # 转换为灰度图
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # 高斯模糊
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # 边缘检测
            edges = cv2.Canny(blurred, 50, 150)
            
            # 形态学闭运算
            kernel = np.ones((3, 3), np.uint8)
            closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            
            # 找到轮廓
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # 找到最大的轮廓（假设是主要物体）
                largest_contour = max(contours, key=cv2.contourArea)
                
                # 创建掩码
                mask = np.zeros(gray.shape, np.uint8)
                cv2.fillPoly(mask, [largest_contour], 255)
                
                # 应用掩码
                result = img_array.copy()
                result[mask == 0] = [255, 255, 255]  # 白色背景
                
                return Image.fromarray(result)
        
        # 如果处理失败，返回原图
        return image
    
    def crop_to_content(self, image: Image.Image, padding: int = 20) -> Image.Image:
        """
        裁剪到内容区域
        
        Args:
            image: 输入图像
            padding: 边距
            
        Returns:
            裁剪后的图像
        """
        img_array = np.array(image)
        
        # 转换为灰度图
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # 二值化
        _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        
        # 找到非零像素的边界
        coords = np.column_stack(np.where(binary > 0))
        
        if len(coords) > 0:
            # 计算边界框
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            
            # 添加边距
            height, width = img_array.shape[:2]
            y_min = max(0, y_min - padding)
            x_min = max(0, x_min - padding)
            y_max = min(height, y_max + padding)
            x_max = min(width, x_max + padding)
            
            # 裁剪图像
            cropped = img_array[y_min:y_max, x_min:x_max]
            return Image.fromarray(cropped)
        
        return image
    
    def preprocess_for_clip(self, image: Image.Image) -> torch.Tensor:
        """
        为CLIP模型预处理图像
        
        Args:
            image: 输入图像
            
        Returns:
            预处理后的张量
        """
        return self.clip_transform(image)
    
    def preprocess_for_classifier(self, image: Image.Image) -> torch.Tensor:
        """
        为分类器预处理图像
        
        Args:
            image: 输入图像
            
        Returns:
            预处理后的张量
        """
        return self.classifier_transform(image)
    
    def extract_dominant_colors(self, image: Image.Image, n_colors: int = 5) -> List[dict]:
        """
        提取主要颜色
        
        Args:
            image: 输入图像
            n_colors: 提取的颜色数量
            
        Returns:
            颜色信息列表
        """
        # 转换为numpy数组
        img_array = np.array(image)
        
        # 重塑为像素列表
        pixels = img_array.reshape(-1, 3)
        
        # 移除纯白色像素（可能是背景）
        non_white_pixels = pixels[np.sum(pixels, axis=1) < 750]  # RGB总和小于750
        
        if len(non_white_pixels) < 100:
            non_white_pixels = pixels  # 如果过滤后像素太少，使用全部像素
        
        # K-means聚类
        kmeans = KMeans(n_clusters=min(n_colors, len(non_white_pixels)), 
                       random_state=42, n_init=10)
        kmeans.fit(non_white_pixels)
        
        colors = []
        labels = kmeans.labels_
        
        for i, color in enumerate(kmeans.cluster_centers_):
            # 计算该颜色的占比
            percentage = np.sum(labels == i) / len(labels)
            
            # 转换为整数RGB值
            rgb = (int(color[0]), int(color[1]), int(color[2]))
            
            # 获取颜色名称
            color_name = self._get_color_name(rgb)
            
            colors.append({
                "rgb": rgb,
                "hex": "#{:02x}{:02x}{:02x}".format(*rgb),
                "name": color_name,
                "percentage": round(percentage * 100, 1)
            })
        
        # 按占比排序
        colors.sort(key=lambda x: x["percentage"], reverse=True)
        
        return colors
    
    def _get_color_name(self, rgb: Tuple[int, int, int]) -> str:
        """
        根据RGB值获取颜色名称
        
        Args:
            rgb: RGB颜色值
            
        Returns:
            颜色名称
        """
        r, g, b = rgb
        
        # 颜色分类规则
        if r > 200 and g > 200 and b > 200:
            return "white"
        elif r < 50 and g < 50 and b < 50:
            return "black"
        elif abs(r - g) < 30 and abs(g - b) < 30:
            if r > 150:
                return "white"
            elif r > 100:
                return "gray"
            else:
                return "black"
        elif r > max(g, b) + 30:
            if r > 200 and g < 100:
                return "red"
            elif r > 150 and g > 100:
                return "orange"
            else:
                return "brown"
        elif g > max(r, b) + 30:
            return "green"
        elif b > max(r, g) + 30:
            return "blue"
        elif r > 150 and g > 100 and b < 100:
            return "yellow"
        elif r > 150 and b > 100 and g < 150:
            return "purple"
        elif r > 150 and g > 100 and b > 100:
            return "pink"
        elif r > 100 and g > 80 and b > 60:
            return "beige"
        else:
            return "gray"
    
    def create_thumbnail(self, image: Image.Image, size: Tuple[int, int] = (150, 150)) -> Image.Image:
        """
        创建缩略图
        
        Args:
            image: 输入图像
            size: 缩略图尺寸
            
        Returns:
            缩略图
        """
        # 保持宽高比的缩略图
        image.thumbnail(size, Image.Resampling.LANCZOS)
        
        # 创建白色背景
        thumbnail = Image.new('RGB', size, (255, 255, 255))
        
        # 居中粘贴
        offset = ((size[0] - image.size[0]) // 2, (size[1] - image.size[1]) // 2)
        thumbnail.paste(image, offset)
        
        return thumbnail
    
    def batch_preprocess(self, image_paths: List[str], 
                        output_dir: str,
                        enhance_quality: bool = True,
                        create_thumbnails: bool = True) -> List[dict]:
        """
        批量预处理图像
        
        Args:
            image_paths: 图像路径列表
            output_dir: 输出目录
            enhance_quality: 是否增强质量
            create_thumbnails: 是否创建缩略图
            
        Returns:
            处理结果列表
        """
        os.makedirs(output_dir, exist_ok=True)
        
        results = []
        
        for i, image_path in enumerate(image_paths):
            try:
                # 加载图像
                image = self.load_and_validate_image(image_path)
                
                # 增强质量
                if enhance_quality:
                    image = self.enhance_image_quality(image)
                
                # 保存处理后的图像
                filename = f"processed_{i:04d}.jpg"
                output_path = os.path.join(output_dir, filename)
                image.save(output_path, "JPEG", quality=95)
                
                # 创建缩略图
                thumbnail_path = None
                if create_thumbnails:
                    thumbnail = self.create_thumbnail(image)
                    thumbnail_filename = f"thumb_{i:04d}.jpg"
                    thumbnail_path = os.path.join(output_dir, thumbnail_filename)
                    thumbnail.save(thumbnail_path, "JPEG", quality=85)
                
                # 提取颜色
                colors = self.extract_dominant_colors(image)
                
                results.append({
                    "original_path": image_path,
                    "processed_path": output_path,
                    "thumbnail_path": thumbnail_path,
                    "colors": colors,
                    "size": image.size,
                    "status": "success"
                })
                
            except Exception as e:
                results.append({
                    "original_path": image_path,
                    "error": str(e),
                    "status": "failed"
                })
        
        return results

# 全局预处理器实例
_preprocessor = None

def get_preprocessor() -> ImagePreprocessor:
    """
    获取全局预处理器实例（单例模式）
    """
    global _preprocessor
    if _preprocessor is None:
        _preprocessor = ImagePreprocessor()
    return _preprocessor