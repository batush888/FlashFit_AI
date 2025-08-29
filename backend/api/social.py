import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import uuid
from urllib.parse import quote

class SocialFeaturesHandler:
    """
    社交功能处理器
    管理用户的搭配分享、评分和社区互动功能
    """
    
    def __init__(self):
        self.shared_outfits_file = Path("data/shared_outfits.json")
        self.outfit_ratings_file = Path("data/outfit_ratings.json")
        self.social_interactions_file = Path("data/social_interactions.json")
        
        # 确保数据目录存在
        os.makedirs("data", exist_ok=True)
        
        # 初始化数据文件
        self._init_data_files()
        
        print("社交功能处理器初始化完成")
    
    def _init_data_files(self):
        """初始化数据文件"""
        if not self.shared_outfits_file.exists():
            with open(self.shared_outfits_file, 'w', encoding='utf-8') as f:
                json.dump({}, f, ensure_ascii=False, indent=2)
        
        if not self.outfit_ratings_file.exists():
            with open(self.outfit_ratings_file, 'w', encoding='utf-8') as f:
                json.dump({}, f, ensure_ascii=False, indent=2)
                
        if not self.social_interactions_file.exists():
            with open(self.social_interactions_file, 'w', encoding='utf-8') as f:
                json.dump({}, f, ensure_ascii=False, indent=2)
    
    def _load_shared_outfits(self) -> dict:
        """加载分享的搭配"""
        try:
            with open(self.shared_outfits_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    
    def _save_shared_outfits(self, shared_outfits: dict):
        """保存分享的搭配"""
        with open(self.shared_outfits_file, 'w', encoding='utf-8') as f:
            json.dump(shared_outfits, f, ensure_ascii=False, indent=2)
    
    def _load_outfit_ratings(self) -> dict:
        """加载搭配评分"""
        try:
            with open(self.outfit_ratings_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    
    def _save_outfit_ratings(self, ratings: dict):
        """保存搭配评分"""
        with open(self.outfit_ratings_file, 'w', encoding='utf-8') as f:
            json.dump(ratings, f, ensure_ascii=False, indent=2)
    
    def _load_social_interactions(self) -> dict:
        """加载社交互动数据"""
        try:
            with open(self.social_interactions_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    
    def _save_social_interactions(self, interactions: dict):
        """保存社交互动数据"""
        with open(self.social_interactions_file, 'w', encoding='utf-8') as f:
            json.dump(interactions, f, ensure_ascii=False, indent=2)
    
    async def share_outfit(self, user_id: str, outfit_data: Dict[str, Any], share_options: Dict[str, Any]) -> Dict[str, Any]:
        """
        分享搭配到社区
        
        Args:
            user_id: 用户ID
            outfit_data: 搭配数据
            share_options: 分享选项 (privacy_level, allow_comments, tags)
            
        Returns:
            分享结果和分享链接
        """
        shared_outfits = self._load_shared_outfits()
        
        # 创建分享ID
        share_id = str(uuid.uuid4())
        
        # 创建分享记录
        shared_outfit = {
            "share_id": share_id,
            "user_id": user_id,
            "outfit_id": outfit_data.get("id", str(uuid.uuid4())),
            "title": outfit_data.get("title", "我的搭配"),
            "description": share_options.get("description", ""),
            "items": outfit_data.get("items", []),
            "occasion": outfit_data.get("occasion", "日常"),
            "style_tags": outfit_data.get("style_tags", []),
            "tips": outfit_data.get("tips", []),
            "collage_url": outfit_data.get("collage_url", ""),
            "privacy_level": share_options.get("privacy_level", "public"),  # public, friends, private
            "allow_comments": share_options.get("allow_comments", True),
            "share_tags": share_options.get("tags", []),
            "created_at": datetime.now().isoformat(),
            "view_count": 0,
            "like_count": 0,
            "comment_count": 0,
            "share_count": 0,
            "average_rating": 0.0,
            "rating_count": 0,
            "is_featured": False,
            "is_active": True
        }
        
        shared_outfits[share_id] = shared_outfit
        self._save_shared_outfits(shared_outfits)
        
        # 生成分享链接
        share_url = f"/shared/{share_id}"
        social_share_urls = {
            "facebook": f"https://www.facebook.com/sharer/sharer.php?u={quote(f'https://flashfit.ai{share_url}')}",
            "twitter": f"https://twitter.com/intent/tweet?url={quote(f'https://flashfit.ai{share_url}')}&text={quote(shared_outfit['title'])}",
            "instagram": f"https://www.instagram.com/",  # Instagram doesn't support direct URL sharing
            "pinterest": f"https://pinterest.com/pin/create/button/?url={quote(f'https://flashfit.ai{share_url}')}&description={quote(shared_outfit['title'])}"
        }
        
        return {
            "success": True,
            "message": "搭配分享成功",
            "data": {
                "share_id": share_id,
                "share_url": share_url,
                "social_share_urls": social_share_urls,
                "shared_outfit": shared_outfit
            }
        }
    
    async def get_shared_outfit(self, share_id: str, viewer_user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        获取分享的搭配详情
        
        Args:
            share_id: 分享ID
            viewer_user_id: 查看者用户ID (可选)
            
        Returns:
            分享的搭配详情
        """
        shared_outfits = self._load_shared_outfits()
        
        if share_id not in shared_outfits:
            return {
                "success": False,
                "message": "分享的搭配不存在",
                "error": "SHARED_OUTFIT_NOT_FOUND"
            }
        
        shared_outfit = shared_outfits[share_id]
        
        # 检查隐私设置
        if shared_outfit["privacy_level"] == "private" and viewer_user_id != shared_outfit["user_id"]:
            return {
                "success": False,
                "message": "无权访问此分享",
                "error": "ACCESS_DENIED"
            }
        
        # 增加浏览次数
        if viewer_user_id != shared_outfit["user_id"]:
            shared_outfit["view_count"] += 1
            shared_outfits[share_id] = shared_outfit
            self._save_shared_outfits(shared_outfits)
        
        return {
            "success": True,
            "data": shared_outfit
        }
    
    async def rate_shared_outfit(self, user_id: str, share_id: str, rating: float, review: Optional[str] = None) -> Dict[str, Any]:
        """
        为分享的搭配评分
        
        Args:
            user_id: 评分用户ID
            share_id: 分享ID
            rating: 评分 (1-5)
            review: 评价文字 (可选)
            
        Returns:
            评分结果
        """
        if not (1 <= rating <= 5):
            return {
                "success": False,
                "message": "评分必须在1-5之间",
                "error": "INVALID_RATING"
            }
        
        shared_outfits = self._load_shared_outfits()
        outfit_ratings = self._load_outfit_ratings()
        
        if share_id not in shared_outfits:
            return {
                "success": False,
                "message": "分享的搭配不存在",
                "error": "SHARED_OUTFIT_NOT_FOUND"
            }
        
        # 不能给自己的分享评分
        if shared_outfits[share_id]["user_id"] == user_id:
            return {
                "success": False,
                "message": "不能为自己的分享评分",
                "error": "CANNOT_RATE_OWN_SHARE"
            }
        
        # 初始化评分数据结构
        if share_id not in outfit_ratings:
            outfit_ratings[share_id] = {
                "ratings": {},
                "reviews": {},
                "total_rating": 0.0,
                "rating_count": 0,
                "average_rating": 0.0
            }
        
        rating_data = outfit_ratings[share_id]
        
        # 检查是否已经评分过
        is_update = user_id in rating_data["ratings"]
        old_rating = rating_data["ratings"].get(user_id, 0)
        
        # 更新评分
        rating_data["ratings"][user_id] = rating
        if review:
            rating_data["reviews"][user_id] = {
                "review": review,
                "created_at": datetime.now().isoformat()
            }
        
        # 重新计算平均评分
        if is_update:
            rating_data["total_rating"] = rating_data["total_rating"] - old_rating + rating
        else:
            rating_data["total_rating"] += rating
            rating_data["rating_count"] += 1
        
        rating_data["average_rating"] = rating_data["total_rating"] / rating_data["rating_count"]
        
        # 更新分享搭配的评分信息
        shared_outfits[share_id]["average_rating"] = rating_data["average_rating"]
        shared_outfits[share_id]["rating_count"] = rating_data["rating_count"]
        
        # 保存数据
        self._save_outfit_ratings(outfit_ratings)
        self._save_shared_outfits(shared_outfits)
        
        return {
            "success": True,
            "message": "评分成功" if not is_update else "评分更新成功",
            "data": {
                "user_rating": rating,
                "average_rating": rating_data["average_rating"],
                "rating_count": rating_data["rating_count"]
            }
        }
    
    async def get_popular_shared_outfits(self, limit: int = 20, offset: int = 0, filter_options: Optional[Dict] = None) -> Dict[str, Any]:
        """
        获取热门分享搭配
        
        Args:
            limit: 限制数量
            offset: 偏移量
            filter_options: 过滤选项 (occasion, style_tags, min_rating)
            
        Returns:
            热门分享搭配列表
        """
        shared_outfits = self._load_shared_outfits()
        filter_options = filter_options or {}
        
        # 过滤公开的活跃分享
        filtered_outfits = []
        for share_id, outfit in shared_outfits.items():
            if not outfit.get("is_active", True) or outfit.get("privacy_level") != "public":
                continue
            
            # 应用过滤条件
            if filter_options.get("occasion") and outfit.get("occasion") != filter_options["occasion"]:
                continue
            
            if filter_options.get("style_tags"):
                required_tags = set(filter_options["style_tags"])
                outfit_tags = set(outfit.get("style_tags", []) + outfit.get("share_tags", []))
                if not required_tags.intersection(outfit_tags):
                    continue
            
            if filter_options.get("min_rating") and outfit.get("average_rating", 0) < filter_options["min_rating"]:
                continue
            
            # 计算热门度分数 (综合浏览量、点赞数、评分等)
            popularity_score = (
                outfit.get("view_count", 0) * 0.1 +
                outfit.get("like_count", 0) * 0.3 +
                outfit.get("share_count", 0) * 0.4 +
                outfit.get("average_rating", 0) * outfit.get("rating_count", 0) * 0.2
            )
            
            outfit_with_score = outfit.copy()
            outfit_with_score["popularity_score"] = popularity_score
            outfit_with_score["share_id"] = share_id
            filtered_outfits.append(outfit_with_score)
        
        # 按热门度排序
        filtered_outfits.sort(key=lambda x: x["popularity_score"], reverse=True)
        
        # 分页
        total = len(filtered_outfits)
        paginated_outfits = filtered_outfits[offset:offset + limit]
        
        return {
            "success": True,
            "data": {
                "outfits": paginated_outfits,
                "total": total,
                "limit": limit,
                "offset": offset,
                "has_more": offset + limit < total
            }
        }
    
    async def like_shared_outfit(self, user_id: str, share_id: str) -> Dict[str, Any]:
        """
        点赞分享的搭配
        
        Args:
            user_id: 用户ID
            share_id: 分享ID
            
        Returns:
            点赞结果
        """
        shared_outfits = self._load_shared_outfits()
        social_interactions = self._load_social_interactions()
        
        if share_id not in shared_outfits:
            return {
                "success": False,
                "message": "分享的搭配不存在",
                "error": "SHARED_OUTFIT_NOT_FOUND"
            }
        
        # 初始化用户互动数据
        if user_id not in social_interactions:
            social_interactions[user_id] = {
                "liked_shares": [],
                "shared_outfits": [],
                "following": [],
                "followers": []
            }
        
        user_interactions = social_interactions[user_id]
        
        # 检查是否已经点赞
        if share_id in user_interactions["liked_shares"]:
            # 取消点赞
            user_interactions["liked_shares"].remove(share_id)
            shared_outfits[share_id]["like_count"] = max(0, shared_outfits[share_id].get("like_count", 0) - 1)
            action = "unliked"
            message = "取消点赞成功"
        else:
            # 添加点赞
            user_interactions["liked_shares"].append(share_id)
            shared_outfits[share_id]["like_count"] = shared_outfits[share_id].get("like_count", 0) + 1
            action = "liked"
            message = "点赞成功"
        
        # 保存数据
        self._save_social_interactions(social_interactions)
        self._save_shared_outfits(shared_outfits)
        
        return {
            "success": True,
            "message": message,
            "data": {
                "action": action,
                "like_count": shared_outfits[share_id]["like_count"],
                "is_liked": action == "liked"
            }
        }
    
    async def get_user_shared_outfits(self, user_id: str, limit: int = 20, offset: int = 0) -> Dict[str, Any]:
        """
        获取用户的分享搭配列表
        
        Args:
            user_id: 用户ID
            limit: 限制数量
            offset: 偏移量
            
        Returns:
            用户分享搭配列表
        """
        shared_outfits = self._load_shared_outfits()
        
        # 筛选用户的分享
        user_shares = []
        for share_id, outfit in shared_outfits.items():
            if outfit["user_id"] == user_id and outfit.get("is_active", True):
                outfit_with_id = outfit.copy()
                outfit_with_id["share_id"] = share_id
                user_shares.append(outfit_with_id)
        
        # 按创建时间排序
        user_shares.sort(key=lambda x: x["created_at"], reverse=True)
        
        # 分页
        total = len(user_shares)
        paginated_shares = user_shares[offset:offset + limit]
        
        return {
            "success": True,
            "data": {
                "shares": paginated_shares,
                "total": total,
                "limit": limit,
                "offset": offset,
                "has_more": offset + limit < total
            }
        }
    
    async def delete_shared_outfit(self, user_id: str, share_id: str) -> Dict[str, Any]:
        """
        删除分享的搭配
        
        Args:
            user_id: 用户ID
            share_id: 分享ID
            
        Returns:
            删除结果
        """
        shared_outfits = self._load_shared_outfits()
        
        if share_id not in shared_outfits:
            return {
                "success": False,
                "message": "分享的搭配不存在",
                "error": "SHARED_OUTFIT_NOT_FOUND"
            }
        
        # 检查权限
        if shared_outfits[share_id]["user_id"] != user_id:
            return {
                "success": False,
                "message": "无权删除此分享",
                "error": "ACCESS_DENIED"
            }
        
        # 软删除 - 标记为不活跃
        shared_outfits[share_id]["is_active"] = False
        shared_outfits[share_id]["deleted_at"] = datetime.now().isoformat()
        
        self._save_shared_outfits(shared_outfits)
        
        return {
            "success": True,
            "message": "分享删除成功"
        }
    
    async def get_social_stats(self, user_id: str) -> Dict[str, Any]:
        """
        获取用户的社交统计数据
        
        Args:
            user_id: 用户ID
            
        Returns:
            社交统计数据
        """
        shared_outfits = self._load_shared_outfits()
        social_interactions = self._load_social_interactions()
        
        # 计算用户分享统计
        user_shares = [outfit for outfit in shared_outfits.values() 
                      if outfit["user_id"] == user_id and outfit.get("is_active", True)]
        
        total_views = sum(share.get("view_count", 0) for share in user_shares)
        total_likes = sum(share.get("like_count", 0) for share in user_shares)
        total_shares = len(user_shares)
        avg_rating = sum(share.get("average_rating", 0) for share in user_shares) / max(1, total_shares)
        
        # 获取用户互动数据
        user_interactions = social_interactions.get(user_id, {
            "liked_shares": [],
            "shared_outfits": [],
            "following": [],
            "followers": []
        })
        
        return {
            "success": True,
            "data": {
                "shares_count": total_shares,
                "total_views": total_views,
                "total_likes": total_likes,
                "average_rating": round(avg_rating, 2),
                "liked_shares_count": len(user_interactions["liked_shares"]),
                "following_count": len(user_interactions["following"]),
                "followers_count": len(user_interactions["followers"]),
                "engagement_rate": round((total_likes / max(1, total_views)) * 100, 2)
            }
        }

# 创建全局实例
social_handler = SocialFeaturesHandler()