from fastapi import UploadFile, HTTPException
from PIL import Image
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import uuid
import shutil

# 导入我们的模型
from models.clip_model import get_clip_model
from models.classifier import get_classifier

class UploadHandler:
    """
    文件上传处理器
    """
    
    def __init__(self):
        # 存储路径
        self.upload_dir = Path("data/uploads")
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        
        # 用户数据文件
        self.users_file = Path("data/users.json")
        
        # 支持的图像格式
        self.allowed_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
        
        # 最大文件大小 (5MB)
        self.max_file_size = 5 * 1024 * 1024
        
        print("上传处理器初始化完成")
    
    def _load_users(self) -> dict:
        """
        加载用户数据
        """
        try:
            with open(self.users_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    
    def _save_users(self, users: dict):
        """
        保存用户数据
        """
        with open(self.users_file, 'w', encoding='utf-8') as f:
            json.dump(users, f, ensure_ascii=False, indent=2)
    
    def _validate_image(self, file: UploadFile) -> None:
        """
        验证上传的图像文件
        """
        # 检查文件扩展名
        file_ext = Path(file.filename or "").suffix.lower()
        if file_ext not in self.allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"不支持的文件格式。支持的格式: {', '.join(self.allowed_extensions)}"
            )
        
        # 检查文件大小
        if file.size and file.size > self.max_file_size:
            raise HTTPException(
                status_code=400,
                detail=f"文件大小超过限制 ({self.max_file_size // (1024*1024)}MB)"
            )
    
    def _save_uploaded_file(self, file: UploadFile, user_id: str) -> str:
        """
        保存上传的文件
        
        Returns:
            保存的文件路径
        """
        # 生成唯一文件名
        file_ext = Path(file.filename or "").suffix.lower()
        unique_filename = f"{user_id}_{uuid.uuid4().hex}{file_ext}"
        
        # 创建用户目录
        user_dir = self.upload_dir / user_id
        user_dir.mkdir(exist_ok=True)
        
        # 保存文件
        file_path = user_dir / unique_filename
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        return str(file_path)
    
    def _process_image(self, image_path: str) -> Dict[str, Any]:
        """
        处理图像：分类和特征提取
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            处理结果字典
        """
        try:
            # 加载图像
            image = Image.open(image_path).convert('RGB')
            
            # 获取模型实例
            clip_model = get_clip_model()
            classifier = get_classifier()
            
            # 服装分类
            classification_result = classifier.classify_garment(image)
            
            # CLIP特征提取
            embeddings = clip_model.encode_image(image)
            
            # 生成风格关键词
            style_keywords = classifier.get_style_keywords(classification_result)
            
            return {
                "classification": classification_result,
                "embeddings": embeddings.tolist(),  # 转换为列表以便JSON序列化
                "style_keywords": style_keywords,
                "image_size": image.size,
                "processed_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"图像处理失败: {str(e)}"
            )
    
    async def process_upload(self, file: UploadFile, user_id: str) -> Dict[str, Any]:
        """
        处理文件上传的主要方法
        
        Args:
            file: 上传的文件
            user_id: 用户ID
            
        Returns:
            处理结果
        """
        # 验证文件
        self._validate_image(file)
        
        # 保存文件
        file_path = self._save_uploaded_file(file, user_id)
        
        try:
            # 处理图像
            processing_result = self._process_image(file_path)
            
            # 生成物品ID
            item_id = f"item_{uuid.uuid4().hex}"
            
            # 创建物品记录
            item_record = {
                "item_id": item_id,
                "user_id": user_id,
                "filename": file.filename,
                "file_path": file_path,
                "url": f"/api/images/{user_id}/{Path(file_path).name}",
                "upload_time": datetime.now().isoformat(),
                "garment_type": processing_result["classification"]["category"],
                "garment_type_cn": processing_result["classification"]["category_cn"],
                "colors": processing_result["classification"]["dominant_colors"],
                "embeddings": processing_result["embeddings"],
                "style_keywords": processing_result["style_keywords"],
                "confidence": processing_result["classification"]["confidence"],
                "image_size": processing_result["image_size"],
                "tags": [],  # 用户可以添加的标签
                "is_favorite": False
            }
            
            # 更新用户数据
            users = self._load_users()
            
            # 找到用户并添加物品
            user_found = False
            for email, user in users.items():
                if user["user_id"] == user_id:
                    if "wardrobe_items" not in user:
                        user["wardrobe_items"] = []
                    user["wardrobe_items"].append(item_record)
                    user_found = True
                    break
            
            if not user_found:
                raise HTTPException(status_code=404, detail="用户不存在")
            
            # 保存更新后的用户数据
            self._save_users(users)
            
            # 返回结果（不包含嵌入向量以减少响应大小）
            response_data = item_record.copy()
            response_data.pop("embeddings", None)  # 移除嵌入向量
            
            return {
                "message": "上传成功",
                "item": response_data
            }
            
        except Exception as e:
            # 如果处理失败，删除已保存的文件
            try:
                os.remove(file_path)
            except:
                pass
            raise e
    
    def get_image_url(self, user_id: str, filename: str) -> str:
        """
        获取图像URL
        
        Args:
            user_id: 用户ID
            filename: 文件名
            
        Returns:
            图像URL
        """
        return f"/api/images/{user_id}/{filename}"
    
    def delete_uploaded_file(self, file_path: str) -> bool:
        """
        删除上传的文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            是否删除成功
        """
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                return True
            return False
        except Exception as e:
            print(f"删除文件失败: {e}")
            return False
    
    def get_upload_stats(self) -> Dict[str, Any]:
        """
        获取上传统计信息
        
        Returns:
            统计信息
        """
        users = self._load_users()
        
        total_items = 0
        category_counts = {}
        
        for user in users.values():
            items = user.get("wardrobe_items", [])
            total_items += len(items)
            
            for item in items:
                category = item.get("garment_type", "unknown")
                category_counts[category] = category_counts.get(category, 0) + 1
        
        return {
            "total_items": total_items,
            "category_distribution": category_counts,
            "total_users_with_items": sum(1 for user in users.values() if user.get("wardrobe_items"))
        }