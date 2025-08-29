from datetime import datetime, timedelta
from typing import Optional
from jose import jwt, JWTError
from passlib.context import CryptContext
import os
import json
from pathlib import Path

class AuthHandler:
    """
    用户认证处理器
    """
    
    def __init__(self):
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.secret_key = os.getenv("JWT_SECRET_KEY", "flashfit-ai-secret-key-2024")
        self.algorithm = "HS256"
        self.access_token_expire_minutes = 60 * 24 * 7  # 7天
        
        # 简单的文件存储（MVP版本）
        self.users_file = Path("data/users.json")
        self.users_file.parent.mkdir(exist_ok=True)
        
        # 初始化用户数据
        self._init_users_data()
        
        print("认证处理器初始化完成")
    
    def _init_users_data(self):
        """
        初始化用户数据文件
        """
        if not self.users_file.exists():
            with open(self.users_file, 'w', encoding='utf-8') as f:
                json.dump({}, f, ensure_ascii=False, indent=2)
    
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
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """
        验证密码
        """
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """
        生成密码哈希
        """
        return self.pwd_context.hash(password)
    
    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """
        创建访问令牌
        """
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str) -> Optional[str]:
        """
        验证令牌并返回用户ID
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            user_id = payload.get("sub")
            if user_id is None or not isinstance(user_id, str):
                return None
            return user_id
        except JWTError:
            return None
    
    async def register(self, email: str, password: str) -> dict:
        """
        用户注册
        """
        users = self._load_users()
        
        # 检查用户是否已存在
        if email in users:
            raise ValueError("用户已存在")
        
        # 验证邮箱格式（简单验证）
        if "@" not in email or "." not in email:
            raise ValueError("邮箱格式不正确")
        
        # 验证密码强度
        if len(password) < 6:
            raise ValueError("密码长度至少6位")
        
        # 创建用户
        user_id = f"user_{len(users) + 1}_{int(datetime.now().timestamp())}"
        hashed_password = self.get_password_hash(password)
        
        users[email] = {
            "user_id": user_id,
            "email": email,
            "password_hash": hashed_password,
            "created_at": datetime.now().isoformat(),
            "consent_given": True,  # 注册时默认同意
            "wardrobe_items": [],
            "feedback_history": []
        }
        
        self._save_users(users)
        
        # 创建访问令牌
        access_token = self.create_access_token(data={"sub": user_id})
        
        return {
            "message": "注册成功",
            "user_id": user_id,
            "access_token": access_token,
            "token_type": "bearer"
        }
    
    async def login(self, email: str, password: str) -> dict:
        """
        用户登录
        """
        users = self._load_users()
        
        # 检查用户是否存在
        if email not in users:
            raise ValueError("用户不存在")
        
        user = users[email]
        
        # 验证密码
        if not self.verify_password(password, user["password_hash"]):
            raise ValueError("密码错误")
        
        # 创建访问令牌
        access_token = self.create_access_token(data={"sub": user["user_id"]})
        
        # 更新最后登录时间
        user["last_login"] = datetime.now().isoformat()
        self._save_users(users)
        
        # 构建用户信息对象
        user_info = {
            "id": user["user_id"],
            "email": email,
            "created_at": user["created_at"],
            "consent_given": user.get("consent_given", False)
        }
        
        return {
            "success": True,
            "data": {
                "token": access_token,
                "user": user_info
            },
            "message": "登录成功"
        }
    
    def get_user_by_id(self, user_id: str) -> Optional[dict]:
        """
        根据用户ID获取用户信息
        """
        users = self._load_users()
        
        for email, user in users.items():
            if user["user_id"] == user_id:
                # 返回用户信息（不包含密码哈希）
                user_info = user.copy()
                user_info.pop("password_hash", None)
                return user_info
        
        return None
    
    def update_user_consent(self, user_id: str, consent: bool) -> bool:
        """
        更新用户同意状态
        """
        users = self._load_users()
        
        for email, user in users.items():
            if user["user_id"] == user_id:
                user["consent_given"] = consent
                user["consent_updated_at"] = datetime.now().isoformat()
                self._save_users(users)
                return True
        
        return False
    
    def delete_user(self, user_id: str) -> bool:
        """
        删除用户账户
        """
        users = self._load_users()
        
        for email, user in users.items():
            if user["user_id"] == user_id:
                del users[email]
                self._save_users(users)
                return True
        
        return False
    
    def get_user_stats(self) -> dict:
        """
        获取用户统计信息
        """
        users = self._load_users()
        
        total_users = len(users)
        consented_users = sum(1 for user in users.values() if user.get("consent_given", False))
        
        return {
            "total_users": total_users,
            "consented_users": consented_users,
            "consent_rate": round(consented_users / total_users * 100, 1) if total_users > 0 else 0
        }