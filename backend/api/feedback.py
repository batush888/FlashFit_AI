import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import uuid

class FeedbackHandler:
    """
    用户反馈处理器
    """
    
    def __init__(self):
        self.users_file = Path("data/users.json")
        self.feedback_file = Path("data/feedback.json")
        
        # 初始化反馈数据文件
        self._init_feedback_data()
        
        print("反馈处理器初始化完成")
    
    def _init_feedback_data(self):
        """
        初始化反馈数据文件
        """
        if not self.feedback_file.exists():
            self.feedback_file.parent.mkdir(exist_ok=True)
            with open(self.feedback_file, 'w', encoding='utf-8') as f:
                json.dump([], f, ensure_ascii=False, indent=2)
    
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
    
    def _load_feedback(self) -> List[dict]:
        """
        加载反馈数据
        """
        try:
            with open(self.feedback_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []
    
    def _save_feedback(self, feedback_list: List[dict]):
        """
        保存反馈数据
        """
        with open(self.feedback_file, 'w', encoding='utf-8') as f:
            json.dump(feedback_list, f, ensure_ascii=False, indent=2)
    
    async def record_feedback(self, suggestion_id: str, user_id: str, 
                            liked: bool, notes: Optional[str] = None) -> Dict[str, Any]:
        """
        记录用户反馈
        
        Args:
            suggestion_id: 建议ID
            user_id: 用户ID
            liked: 是否喜欢
            notes: 备注
            
        Returns:
            反馈记录结果
        """
        # 生成反馈ID
        feedback_id = f"feedback_{uuid.uuid4().hex}"
        
        # 创建反馈记录
        feedback_record = {
            "feedback_id": feedback_id,
            "suggestion_id": suggestion_id,
            "user_id": user_id,
            "liked": liked,
            "notes": notes,
            "created_at": datetime.now().isoformat(),
            "feedback_type": "suggestion_rating"
        }
        
        # 保存到全局反馈文件
        feedback_list = self._load_feedback()
        feedback_list.append(feedback_record)
        self._save_feedback(feedback_list)
        
        # 更新用户数据中的反馈历史
        users = self._load_users()
        user_found = False
        
        for user in users.values():
            if user.get("user_id") == user_id:
                if "feedback_history" not in user:
                    user["feedback_history"] = []
                
                user["feedback_history"].append(feedback_record)
                user_found = True
                break
        
        if not user_found:
            raise ValueError("用户不存在")
        
        # 保存用户数据
        self._save_users(users)
        
        return {
            "message": "反馈记录成功",
            "feedback_id": feedback_id,
            "suggestion_id": suggestion_id,
            "liked": liked
        }
    
    async def get_user_feedback_history(self, user_id: str, 
                                       limit: int = 50) -> Dict[str, Any]:
        """
        获取用户反馈历史
        
        Args:
            user_id: 用户ID
            limit: 返回数量限制
            
        Returns:
            反馈历史
        """
        users = self._load_users()
        
        user_feedback = []
        for user in users.values():
            if user.get("user_id") == user_id:
                user_feedback = user.get("feedback_history", [])
                break
        
        # 按时间倒序排列
        sorted_feedback = sorted(
            user_feedback,
            key=lambda x: x.get("created_at", ""),
            reverse=True
        )[:limit]
        
        # 统计信息
        total_feedback = len(user_feedback)
        liked_count = sum(1 for f in user_feedback if f.get("liked", False))
        disliked_count = total_feedback - liked_count
        
        return {
            "feedback_history": sorted_feedback,
            "statistics": {
                "total_feedback": total_feedback,
                "liked_count": liked_count,
                "disliked_count": disliked_count,
                "satisfaction_rate": round(liked_count / total_feedback * 100, 1) if total_feedback > 0 else 0
            }
        }
    
    async def get_suggestion_feedback(self, suggestion_id: str) -> Dict[str, Any]:
        """
        获取特定建议的反馈统计
        
        Args:
            suggestion_id: 建议ID
            
        Returns:
            建议反馈统计
        """
        feedback_list = self._load_feedback()
        
        suggestion_feedback = [
            f for f in feedback_list 
            if f.get("suggestion_id") == suggestion_id
        ]
        
        if not suggestion_feedback:
            return {
                "suggestion_id": suggestion_id,
                "feedback_count": 0,
                "statistics": {
                    "liked_count": 0,
                    "disliked_count": 0,
                    "satisfaction_rate": 0
                },
                "feedback_list": []
            }
        
        liked_count = sum(1 for f in suggestion_feedback if f.get("liked", False))
        total_count = len(suggestion_feedback)
        
        return {
            "suggestion_id": suggestion_id,
            "feedback_count": total_count,
            "statistics": {
                "liked_count": liked_count,
                "disliked_count": total_count - liked_count,
                "satisfaction_rate": round(liked_count / total_count * 100, 1)
            },
            "feedback_list": suggestion_feedback
        }
    
    async def get_feedback_analytics(self, days: int = 30) -> Dict[str, Any]:
        """
        获取反馈分析数据
        
        Args:
            days: 分析天数
            
        Returns:
            反馈分析结果
        """
        feedback_list = self._load_feedback()
        
        # 计算时间范围
        from datetime import timedelta
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # 筛选时间范围内的反馈
        recent_feedback = []
        for feedback in feedback_list:
            try:
                feedback_date = datetime.fromisoformat(feedback.get("created_at", ""))
                if start_date <= feedback_date <= end_date:
                    recent_feedback.append(feedback)
            except ValueError:
                continue
        
        if not recent_feedback:
            return {
                "period": f"最近{days}天",
                "total_feedback": 0,
                "statistics": {
                    "overall_satisfaction_rate": 0,
                    "daily_average": 0,
                    "liked_count": 0,
                    "disliked_count": 0
                },
                "trends": [],
                "top_issues": []
            }
        
        # 基础统计
        total_feedback = len(recent_feedback)
        liked_count = sum(1 for f in recent_feedback if f.get("liked", False))
        disliked_count = total_feedback - liked_count
        
        # 按日期分组统计趋势
        daily_stats = {}
        for feedback in recent_feedback:
            try:
                date_str = feedback.get("created_at", "")[:10]  # 取日期部分
                if date_str not in daily_stats:
                    daily_stats[date_str] = {"total": 0, "liked": 0}
                
                daily_stats[date_str]["total"] += 1
                if feedback.get("liked", False):
                    daily_stats[date_str]["liked"] += 1
            except:
                continue
        
        # 转换为趋势数据
        trends = []
        for date, stats in sorted(daily_stats.items()):
            satisfaction_rate = (stats["liked"] / stats["total"] * 100) if stats["total"] > 0 else 0
            trends.append({
                "date": date,
                "total_feedback": stats["total"],
                "liked_count": stats["liked"],
                "satisfaction_rate": round(satisfaction_rate, 1)
            })
        
        # 分析负面反馈的常见问题
        negative_feedback = [f for f in recent_feedback if not f.get("liked", False)]
        top_issues = []
        
        if negative_feedback:
            # 简单的关键词统计（实际项目中可以使用更复杂的NLP分析）
            issue_keywords = {}
            for feedback in negative_feedback:
                notes = feedback.get("notes", "")
                if notes:
                    # 简单的关键词提取
                    words = notes.lower().split()
                    for word in words:
                        if len(word) > 2:  # 忽略太短的词
                            issue_keywords[word] = issue_keywords.get(word, 0) + 1
            
            # 取前5个最常见的问题关键词
            top_issues = sorted(issue_keywords.items(), key=lambda x: x[1], reverse=True)[:5]
            top_issues = [{"keyword": k, "count": v} for k, v in top_issues]
        
        return {
            "period": f"最近{days}天",
            "total_feedback": total_feedback,
            "statistics": {
                "overall_satisfaction_rate": round(liked_count / total_feedback * 100, 1),
                "daily_average": round(total_feedback / days, 1),
                "liked_count": liked_count,
                "disliked_count": disliked_count
            },
            "trends": trends,
            "top_issues": top_issues
        }
    
    async def delete_feedback(self, feedback_id: str, user_id: str) -> Dict[str, Any]:
        """
        删除反馈记录
        
        Args:
            feedback_id: 反馈ID
            user_id: 用户ID
            
        Returns:
            删除结果
        """
        # 从全局反馈文件中删除
        feedback_list = self._load_feedback()
        original_count = len(feedback_list)
        
        feedback_list = [
            f for f in feedback_list 
            if not (f.get("feedback_id") == feedback_id and f.get("user_id") == user_id)
        ]
        
        if len(feedback_list) == original_count:
            raise ValueError("反馈记录不存在或无权限删除")
        
        self._save_feedback(feedback_list)
        
        # 从用户数据中删除
        users = self._load_users()
        for user in users.values():
            if user.get("user_id") == user_id:
                feedback_history = user.get("feedback_history", [])
                user["feedback_history"] = [
                    f for f in feedback_history 
                    if f.get("feedback_id") != feedback_id
                ]
                break
        
        self._save_users(users)
        
        return {
            "message": "反馈删除成功",
            "feedback_id": feedback_id
        }