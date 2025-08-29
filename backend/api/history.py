import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import uuid

class OutfitHistoryHandler:
    """
    服装搭配历史记录处理器
    管理用户的搭配历史、收藏夹和搭配集合
    """
    
    def __init__(self):
        self.history_file = Path("data/outfit_history.json")
        self.collections_file = Path("data/outfit_collections.json")
        
        # 确保数据目录存在
        os.makedirs("data", exist_ok=True)
        
        # 初始化数据文件
        self._init_data_files()
        
        print("搭配历史处理器初始化完成")
    
    def _init_data_files(self):
        """初始化数据文件"""
        if not self.history_file.exists():
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump({}, f, ensure_ascii=False, indent=2)
        
        if not self.collections_file.exists():
            with open(self.collections_file, 'w', encoding='utf-8') as f:
                json.dump({}, f, ensure_ascii=False, indent=2)
    
    def _load_history(self) -> dict:
        """加载历史记录"""
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    
    def _save_history(self, history: dict):
        """保存历史记录"""
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    
    def _load_collections(self) -> dict:
        """加载搭配集合"""
        try:
            with open(self.collections_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    
    def _save_collections(self, collections: dict):
        """保存搭配集合"""
        with open(self.collections_file, 'w', encoding='utf-8') as f:
            json.dump(collections, f, ensure_ascii=False, indent=2)
    
    async def save_outfit_to_history(self, user_id: str, outfit_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        保存搭配到历史记录
        
        Args:
            user_id: 用户ID
            outfit_data: 搭配数据
            
        Returns:
            保存结果
        """
        history = self._load_history()
        
        if user_id not in history:
            history[user_id] = {
                "outfits": [],
                "favorites": [],
                "worn_outfits": [],
                "created_at": datetime.now().isoformat()
            }
        
        # 创建历史记录条目
        history_entry = {
            "id": str(uuid.uuid4()),
            "outfit_id": outfit_data.get("suggestion_id", str(uuid.uuid4())),
            "title": outfit_data.get("title_cn", "未命名搭配"),
            "items": outfit_data.get("items", []),
            "occasion": outfit_data.get("occasion", "日常"),
            "style_tags": outfit_data.get("style_tags", []),
            "tips": outfit_data.get("tips_cn", []),
            "similarity_score": outfit_data.get("similarity_score", 0.0),
            "created_at": datetime.now().isoformat(),
            "is_favorite": False,
            "is_worn": False,
            "wear_count": 0,
            "last_worn": None,
            "user_rating": None,
            "user_notes": ""
        }
        
        # 添加到历史记录
        history[user_id]["outfits"].append(history_entry)
        
        # 保持最近100条记录
        if len(history[user_id]["outfits"]) > 100:
            history[user_id]["outfits"] = history[user_id]["outfits"][-100:]
        
        self._save_history(history)
        
        return {
            "success": True,
            "message": "搭配已保存到历史记录",
            "history_id": history_entry["id"]
        }
    
    async def get_outfit_history(self, user_id: str, limit: int = 20, offset: int = 0) -> Dict[str, Any]:
        """
        获取用户搭配历史
        
        Args:
            user_id: 用户ID
            limit: 返回数量限制
            offset: 偏移量
            
        Returns:
            历史记录列表
        """
        history = self._load_history()
        user_history = history.get(user_id, {"outfits": []})
        
        outfits = user_history.get("outfits", [])
        
        # 按创建时间倒序排列
        sorted_outfits = sorted(
            outfits,
            key=lambda x: x.get("created_at", ""),
            reverse=True
        )
        
        # 分页
        start = offset
        end = offset + limit
        paginated_outfits = sorted_outfits[start:end]
        
        return {
            "outfits": paginated_outfits,
            "total": len(sorted_outfits),
            "limit": limit,
            "offset": offset,
            "has_more": end < len(sorted_outfits)
        }
    
    async def toggle_favorite_outfit(self, user_id: str, history_id: str) -> Dict[str, Any]:
        """
        切换搭配收藏状态
        
        Args:
            user_id: 用户ID
            history_id: 历史记录ID
            
        Returns:
            操作结果
        """
        history = self._load_history()
        
        if user_id not in history:
            return {"success": False, "error": "用户历史记录不存在"}
        
        user_outfits = history[user_id]["outfits"]
        
        # 查找目标搭配
        target_outfit = None
        for outfit in user_outfits:
            if outfit["id"] == history_id:
                target_outfit = outfit
                break
        
        if not target_outfit:
            return {"success": False, "error": "搭配记录不存在"}
        
        # 切换收藏状态
        target_outfit["is_favorite"] = not target_outfit.get("is_favorite", False)
        target_outfit["updated_at"] = datetime.now().isoformat()
        
        # 更新收藏列表
        favorites = history[user_id].get("favorites", [])
        if target_outfit["is_favorite"]:
            if history_id not in favorites:
                favorites.append(history_id)
        else:
            if history_id in favorites:
                favorites.remove(history_id)
        
        history[user_id]["favorites"] = favorites
        
        self._save_history(history)
        
        return {
            "success": True,
            "is_favorite": target_outfit["is_favorite"],
            "message": "已添加到收藏" if target_outfit["is_favorite"] else "已取消收藏"
        }
    
    async def get_favorite_outfits(self, user_id: str) -> Dict[str, Any]:
        """
        获取用户收藏的搭配
        
        Args:
            user_id: 用户ID
            
        Returns:
            收藏搭配列表
        """
        history = self._load_history()
        user_history = history.get(user_id, {"outfits": [], "favorites": []})
        
        outfits = user_history.get("outfits", [])
        favorite_ids = user_history.get("favorites", [])
        
        # 筛选收藏的搭配
        favorite_outfits = [
            outfit for outfit in outfits
            if outfit["id"] in favorite_ids or outfit.get("is_favorite", False)
        ]
        
        # 按收藏时间倒序排列
        sorted_favorites = sorted(
            favorite_outfits,
            key=lambda x: x.get("updated_at", x.get("created_at", "")),
            reverse=True
        )
        
        return {
            "favorites": sorted_favorites,
            "total": len(sorted_favorites)
        }
    
    async def mark_outfit_worn(self, user_id: str, history_id: str, worn_date: Optional[str] = None) -> Dict[str, Any]:
        """
        标记搭配为已穿着
        
        Args:
            user_id: 用户ID
            history_id: 历史记录ID
            worn_date: 穿着日期（可选）
            
        Returns:
            操作结果
        """
        history = self._load_history()
        
        if user_id not in history:
            return {"success": False, "error": "用户历史记录不存在"}
        
        user_outfits = history[user_id]["outfits"]
        
        # 查找目标搭配
        target_outfit = None
        for outfit in user_outfits:
            if outfit["id"] == history_id:
                target_outfit = outfit
                break
        
        if not target_outfit:
            return {"success": False, "error": "搭配记录不存在"}
        
        # 更新穿着信息
        target_outfit["is_worn"] = True
        target_outfit["wear_count"] = target_outfit.get("wear_count", 0) + 1
        target_outfit["last_worn"] = worn_date or datetime.now().isoformat()
        target_outfit["updated_at"] = datetime.now().isoformat()
        
        # 更新已穿着列表
        worn_outfits = history[user_id].get("worn_outfits", [])
        if history_id not in worn_outfits:
            worn_outfits.append(history_id)
        history[user_id]["worn_outfits"] = worn_outfits
        
        self._save_history(history)
        
        return {
            "success": True,
            "message": "已标记为穿着",
            "wear_count": target_outfit["wear_count"]
        }
    
    async def create_outfit_collection(self, user_id: str, collection_name: str, description: str = "") -> Dict[str, Any]:
        """
        创建搭配集合
        
        Args:
            user_id: 用户ID
            collection_name: 集合名称
            description: 集合描述
            
        Returns:
            创建结果
        """
        collections = self._load_collections()
        
        if user_id not in collections:
            collections[user_id] = {}
        
        collection_id = str(uuid.uuid4())
        
        collections[user_id][collection_id] = {
            "id": collection_id,
            "name": collection_name,
            "description": description,
            "outfit_ids": [],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        self._save_collections(collections)
        
        return {
            "success": True,
            "message": "搭配集合创建成功",
            "collection_id": collection_id
        }
    
    async def add_outfit_to_collection(self, user_id: str, collection_id: str, history_id: str) -> Dict[str, Any]:
        """
        添加搭配到集合
        
        Args:
            user_id: 用户ID
            collection_id: 集合ID
            history_id: 历史记录ID
            
        Returns:
            操作结果
        """
        collections = self._load_collections()
        
        if user_id not in collections or collection_id not in collections[user_id]:
            return {"success": False, "error": "集合不存在"}
        
        collection = collections[user_id][collection_id]
        
        if history_id not in collection["outfit_ids"]:
            collection["outfit_ids"].append(history_id)
            collection["updated_at"] = datetime.now().isoformat()
            
            self._save_collections(collections)
            
            return {
                "success": True,
                "message": "搭配已添加到集合"
            }
        else:
            return {
                "success": False,
                "error": "搭配已存在于集合中"
            }
    
    async def get_user_collections(self, user_id: str) -> Dict[str, Any]:
        """
        获取用户的搭配集合
        
        Args:
            user_id: 用户ID
            
        Returns:
            集合列表
        """
        collections = self._load_collections()
        user_collections = collections.get(user_id, {})
        
        # 转换为列表格式
        collections_list = list(user_collections.values())
        
        # 按更新时间倒序排列
        sorted_collections = sorted(
            collections_list,
            key=lambda x: x.get("updated_at", x.get("created_at", "")),
            reverse=True
        )
        
        return {
            "collections": sorted_collections,
            "total": len(sorted_collections)
        }
    
    async def get_outfit_statistics(self, user_id: str) -> Dict[str, Any]:
        """
        获取用户搭配统计信息
        
        Args:
            user_id: 用户ID
            
        Returns:
            统计信息
        """
        history = self._load_history()
        collections = self._load_collections()
        
        user_history = history.get(user_id, {"outfits": [], "favorites": [], "worn_outfits": []})
        user_collections = collections.get(user_id, {})
        
        outfits = user_history.get("outfits", [])
        
        # 统计各种数据
        total_outfits = len(outfits)
        favorite_count = len([o for o in outfits if o.get("is_favorite", False)])
        worn_count = len([o for o in outfits if o.get("is_worn", False)])
        collection_count = len(user_collections)
        
        # 统计场合分布
        occasion_stats = {}
        for outfit in outfits:
            occasion = outfit.get("occasion", "未知")
            occasion_stats[occasion] = occasion_stats.get(occasion, 0) + 1
        
        # 统计风格标签
        style_stats = {}
        for outfit in outfits:
            for tag in outfit.get("style_tags", []):
                style_stats[tag] = style_stats.get(tag, 0) + 1
        
        # 最近活动
        recent_outfits = sorted(
            outfits,
            key=lambda x: x.get("created_at", ""),
            reverse=True
        )[:5]
        
        return {
            "total_outfits": total_outfits,
            "favorite_count": favorite_count,
            "worn_count": worn_count,
            "collection_count": collection_count,
            "occasion_distribution": occasion_stats,
            "style_distribution": style_stats,
            "recent_outfits": recent_outfits,
            "wear_rate": round(worn_count / total_outfits * 100, 1) if total_outfits > 0 else 0
        }