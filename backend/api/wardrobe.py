import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

class WardrobeHandler:
    """
    衣橱管理处理器
    """
    
    def __init__(self):
        self.users_file = Path("data/users.json")
        print("衣橱管理器初始化完成")
    
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
    
    async def get_user_wardrobe(self, user_id: str) -> Dict[str, Any]:
        """
        获取用户衣橱
        
        Args:
            user_id: 用户ID
            
        Returns:
            衣橱数据
        """
        users = self._load_users()
        
        # 查找用户
        user_wardrobe = None
        for user in users.values():
            if user.get("user_id") == user_id:
                user_wardrobe = user.get("wardrobe_items", [])
                break
        
        if user_wardrobe is None:
            raise ValueError("用户不存在")
        
        # 统计信息
        category_counts = {}
        color_counts = {}
        total_items = len(user_wardrobe)
        
        for item in user_wardrobe:
            # 统计类别
            category = item.get("garment_type", "unknown")
            category_counts[category] = category_counts.get(category, 0) + 1
            
            # 统计颜色
            colors = item.get("colors", [])
            if colors:
                main_color = colors[0].get("name", "unknown")
                color_counts[main_color] = color_counts.get(main_color, 0) + 1
        
        # 按上传时间排序（最新的在前）
        sorted_items = sorted(
            user_wardrobe, 
            key=lambda x: x.get("upload_time", ""), 
            reverse=True
        )
        
        return {
            "items": sorted_items,
            "stats": {
                "total_items": total_items,
                "categories": category_counts,
                "colors": color_counts,
                "recent_uploads": len([item for item in user_wardrobe if (datetime.now() - datetime.fromisoformat(item.get("upload_time", "2024-01-01T00:00:00").replace('Z', '+00:00'))).days <= 7])
            },
            "total_count": total_items,
            "last_updated": datetime.now().isoformat()
        }
    
    async def get_item_by_id(self, item_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """
        根据ID获取特定物品
        
        Args:
            item_id: 物品ID
            user_id: 用户ID
            
        Returns:
            物品详情
        """
        users = self._load_users()
        
        for user in users.values():
            if user.get("user_id") == user_id:
                for item in user.get("wardrobe_items", []):
                    if item.get("item_id") == item_id:
                        return item
        
        return None
    
    async def update_item(self, item_id: str, user_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        更新物品信息
        
        Args:
            item_id: 物品ID
            user_id: 用户ID
            updates: 更新的字段
            
        Returns:
            更新结果
        """
        users = self._load_users()
        
        # 查找并更新物品
        item_found = False
        for user in users.values():
            if user.get("user_id") == user_id:
                for item in user.get("wardrobe_items", []):
                    if item.get("item_id") == item_id:
                        # 允许更新的字段
                        allowed_fields = {
                            "tags", "is_favorite", "notes", "custom_name"
                        }
                        
                        for field, value in updates.items():
                            if field in allowed_fields:
                                item[field] = value
                        
                        item["updated_at"] = datetime.now().isoformat()
                        item_found = True
                        break
                
                if item_found:
                    break
        
        if not item_found:
            raise ValueError("物品不存在")
        
        # 保存更新
        self._save_users(users)
        
        return {
            "message": "物品更新成功",
            "item_id": item_id,
            "updated_fields": list(updates.keys())
        }
    
    async def delete_item(self, item_id: str, user_id: str) -> Dict[str, Any]:
        """
        删除物品
        
        Args:
            item_id: 物品ID
            user_id: 用户ID
            
        Returns:
            删除结果
        """
        users = self._load_users()
        
        # 查找并删除物品
        item_found = False
        deleted_item = None
        
        for user in users.values():
            if user.get("user_id") == user_id:
                wardrobe_items = user.get("wardrobe_items", [])
                
                for i, item in enumerate(wardrobe_items):
                    if item.get("item_id") == item_id:
                        deleted_item = wardrobe_items.pop(i)
                        item_found = True
                        break
                
                if item_found:
                    break
        
        if not item_found:
            raise ValueError("物品不存在")
        
        # 保存更新
        self._save_users(users)
        
        # 删除物理文件
        if deleted_item and "file_path" in deleted_item:
            try:
                file_path = deleted_item["file_path"]
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"删除文件失败: {e}")
        
        return {
            "message": "物品删除成功",
            "item_id": item_id,
            "deleted_item": {
                "filename": deleted_item.get("filename") if deleted_item else None,
                "garment_type": deleted_item.get("garment_type_cn") if deleted_item else None
            }
        }
    
    async def search_items(self, user_id: str, query: str = "", 
                          category: Optional[str] = None,
                          color: Optional[str] = None,
                          tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        搜索衣橱物品
        
        Args:
            user_id: 用户ID
            query: 搜索关键词
            category: 物品类别筛选
            color: 颜色筛选
            tags: 标签筛选
            
        Returns:
            搜索结果
        """
        wardrobe_data = await self.get_user_wardrobe(user_id)
        items = wardrobe_data["items"]
        
        filtered_items = []
        
        for item in items:
            # 类别筛选
            if category and item.get("garment_type") != category:
                continue
            
            # 颜色筛选
            if color:
                item_colors = [c.get("name", "") for c in item.get("colors", [])]
                if color not in item_colors:
                    continue
            
            # 标签筛选
            if tags:
                item_tags = item.get("tags", [])
                if not any(tag in item_tags for tag in tags):
                    continue
            
            # 关键词搜索
            if query:
                searchable_text = " ".join([
                    item.get("filename", ""),
                    item.get("garment_type_cn", ""),
                    item.get("custom_name", ""),
                    " ".join(item.get("tags", [])),
                    " ".join(item.get("style_keywords", [])),
                    " ".join([c.get("name_cn", "") for c in item.get("colors", [])])
                ]).lower()
                
                if query.lower() not in searchable_text:
                    continue
            
            filtered_items.append(item)
        
        return {
            "items": filtered_items,
            "total_count": len(filtered_items),
            "search_params": {
                "query": query,
                "category": category,
                "color": color,
                "tags": tags
            }
        }
    
    async def get_favorites(self, user_id: str) -> Dict[str, Any]:
        """
        获取收藏的物品
        
        Args:
            user_id: 用户ID
            
        Returns:
            收藏物品列表
        """
        wardrobe_data = await self.get_user_wardrobe(user_id)
        items = wardrobe_data["items"]
        
        favorite_items = [item for item in items if item.get("is_favorite", False)]
        
        return {
            "items": favorite_items,
            "total_count": len(favorite_items)
        }
    
    async def get_category_items(self, user_id: str, category: str) -> Dict[str, Any]:
        """
        获取特定类别的物品
        
        Args:
            user_id: 用户ID
            category: 物品类别
            
        Returns:
            该类别的物品列表
        """
        return await self.search_items(user_id, category=category)
    
    async def bulk_update_tags(self, user_id: str, item_ids: List[str], 
                              tags: List[str], action: str = "add") -> Dict[str, Any]:
        """
        批量更新物品标签
        
        Args:
            user_id: 用户ID
            item_ids: 物品ID列表
            tags: 标签列表
            action: 操作类型 (add/remove/replace)
            
        Returns:
            批量更新结果
        """
        users = self._load_users()
        updated_count = 0
        
        for user in users.values():
            if user.get("user_id") == user_id:
                for item in user.get("wardrobe_items", []):
                    if item.get("item_id") in item_ids:
                        current_tags = item.get("tags", [])
                        
                        if action == "add":
                            # 添加标签（去重）
                            new_tags = list(set(current_tags + tags))
                        elif action == "remove":
                            # 移除标签
                            new_tags = [tag for tag in current_tags if tag not in tags]
                        elif action == "replace":
                            # 替换标签
                            new_tags = tags
                        else:
                            continue
                        
                        item["tags"] = new_tags
                        item["updated_at"] = datetime.now().isoformat()
                        updated_count += 1
                
                break
        
        if updated_count > 0:
            self._save_users(users)
        
        return {
            "message": f"批量更新完成",
            "updated_count": updated_count,
            "total_requested": len(item_ids),
            "action": action,
            "tags": tags
        }