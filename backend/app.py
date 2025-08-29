from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import os
from datetime import datetime
import json

# Import our modules
from models.clip_model import CLIPModel
from models.classifier import GarmentClassifier
from api.auth import AuthHandler
from api.upload import UploadHandler
from api.match import MatchHandler
from api.wardrobe import WardrobeHandler
from api.feedback import FeedbackHandler
from api.enhanced_match import get_enhanced_match_handler
from api.history import OutfitHistoryHandler
from api.social import social_handler

app = FastAPI(
    title="FlashFit AI - 服装搭配助手",
    description="AI驱动的服装搭配建议系统",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境中应该限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Initialize handlers
auth_handler = AuthHandler()
upload_handler = UploadHandler()
match_handler = MatchHandler()
wardrobe_handler = WardrobeHandler()
feedback_handler = FeedbackHandler()
enhanced_match_handler = get_enhanced_match_handler()
history_handler = OutfitHistoryHandler()

# Pydantic models
class UserRegister(BaseModel):
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

class MatchRequest(BaseModel):
    item_id: str
    occasion: Optional[str] = None
    target_count: int = 3

class FeedbackRequest(BaseModel):
    suggestion_id: str
    liked: bool
    notes: Optional[str] = None

class DeleteImageRequest(BaseModel):
    item_id: str

class EnhancedMatchRequest(BaseModel):
    item_id: str
    occasion: Optional[str] = None
    season: Optional[str] = None
    style_preference: Optional[str] = None

class UserFeedbackRequest(BaseModel):
    user_id: str
    suggestion_id: str
    rating: int  # 1-5 stars
    feedback_text: Optional[str] = None

class SaveOutfitRequest(BaseModel):
    outfit_data: dict

class ToggleFavoriteRequest(BaseModel):
    history_id: str

class MarkWornRequest(BaseModel):
    history_id: str
    worn_date: Optional[str] = None

class CreateCollectionRequest(BaseModel):
    collection_name: str
    description: str = ""

class AddToCollectionRequest(BaseModel):
    collection_id: str
    history_id: str

class ShareOutfitRequest(BaseModel):
    outfit_data: dict
    share_options: dict

class RateOutfitRequest(BaseModel):
    rating: float
    review: Optional[str] = None

# Auth dependency
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    user_id = auth_handler.verify_token(token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")
    return user_id

# Routes
@app.get("/")
async def root():
    return {"message": "FlashFit AI - 服装搭配助手 API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Auth endpoints
@app.post("/api/auth/register")
async def register(user: UserRegister):
    try:
        result = await auth_handler.register(user.email, user.password)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/auth/login")
async def login(user: UserLogin):
    try:
        result = await auth_handler.login(user.email, user.password)
        return result
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))

# Upload endpoint
@app.post("/api/upload")
async def upload_image(
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user)
):
    try:
        result = await upload_handler.process_upload(file, user_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Wardrobe endpoint
@app.get("/api/wardrobe")
async def get_wardrobe(user_id: str = Depends(get_current_user)):
    try:
        result = await wardrobe_handler.get_user_wardrobe(user_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Match endpoint
@app.post("/api/match")
async def get_outfit_suggestions(
    request: MatchRequest,
    user_id: str = Depends(get_current_user)
):
    try:
        result = await match_handler.generate_suggestions(
            request.item_id, user_id, request.occasion, request.target_count
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Feedback endpoint
@app.post("/api/feedback")
async def submit_feedback(
    request: FeedbackRequest,
    user_id: str = Depends(get_current_user)
):
    try:
        result = await feedback_handler.record_feedback(
            request.suggestion_id, user_id, request.liked, request.notes
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Delete image endpoint
@app.post("/api/user/delete_image")
async def delete_image(
    request: DeleteImageRequest,
    user_id: str = Depends(get_current_user)
):
    try:
        result = await wardrobe_handler.delete_item(user_id, request.item_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Enhanced ML endpoints
@app.post("/api/enhanced/match")
async def get_enhanced_outfit_suggestions(
    request: EnhancedMatchRequest,
    user_id: str = Depends(get_current_user)
):
    """获取增强版AI搭配建议"""
    try:
        result = await enhanced_match_handler.generate_enhanced_suggestions(
            item_id=request.item_id,
            user_id=user_id,
            occasion=request.occasion,
            season=request.season,
            style_preference=request.style_preference
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/enhanced/feedback")
async def submit_enhanced_feedback(
    request: UserFeedbackRequest,
    user_id: str = Depends(get_current_user)
):
    """提交用户反馈以改进AI模型"""
    try:
        success = enhanced_match_handler.save_user_feedback(
            user_id=user_id,
            suggestion_id=request.suggestion_id,
            rating=request.rating,
            feedback_text=request.feedback_text
        )
        return {"success": success, "message": "反馈已保存"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/enhanced/analytics")
async def get_suggestion_analytics(
    user_id: str = Depends(get_current_user)
):
    """获取用户的搭配建议分析数据"""
    try:
        analytics = enhanced_match_handler.get_suggestion_analytics(user_id=user_id)
        return analytics
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/enhanced/analytics/global")
async def get_global_analytics(
    user_id: str = Depends(get_current_user)
):
    try:
        result = enhanced_match_handler.get_suggestion_analytics()
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Outfit History endpoints
@app.post("/api/history/save")
async def save_outfit_to_history(
    request: SaveOutfitRequest,
    user_id: str = Depends(get_current_user)
):
    try:
        result = await history_handler.save_outfit_to_history(user_id, request.outfit_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/history")
async def get_outfit_history(
    limit: int = 20,
    offset: int = 0,
    user_id: str = Depends(get_current_user)
):
    try:
        result = await history_handler.get_outfit_history(user_id, limit, offset)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/history/favorite")
async def toggle_favorite_outfit(
    request: ToggleFavoriteRequest,
    user_id: str = Depends(get_current_user)
):
    try:
        result = await history_handler.toggle_favorite_outfit(user_id, request.history_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/history/favorites")
async def get_favorite_outfits(
    user_id: str = Depends(get_current_user)
):
    try:
        result = await history_handler.get_favorite_outfits(user_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/history/worn")
async def mark_outfit_worn(
    request: MarkWornRequest,
    user_id: str = Depends(get_current_user)
):
    try:
        result = await history_handler.mark_outfit_worn(user_id, request.history_id, request.worn_date)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/collections/create")
async def create_outfit_collection(
    request: CreateCollectionRequest,
    user_id: str = Depends(get_current_user)
):
    try:
        result = await history_handler.create_outfit_collection(user_id, request.collection_name, request.description)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/collections/add")
async def add_outfit_to_collection(
    request: AddToCollectionRequest,
    user_id: str = Depends(get_current_user)
):
    try:
        result = await history_handler.add_outfit_to_collection(user_id, request.collection_id, request.history_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/collections")
async def get_user_collections(
    user_id: str = Depends(get_current_user)
):
    try:
        result = await history_handler.get_user_collections(user_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/history/statistics")
async def get_outfit_statistics(
    user_id: str = Depends(get_current_user)
):
    try:
        result = await history_handler.get_outfit_statistics(user_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Social Features Endpoints
@app.post("/api/social/share")
async def share_outfit(
    request: ShareOutfitRequest,
    user_id: str = Depends(get_current_user)
):
    try:
        result = await social_handler.share_outfit(user_id, request.outfit_data, request.share_options)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/social/shared/{share_id}")
async def get_shared_outfit(
    share_id: str,
    user_id: Optional[str] = Depends(get_current_user)
):
    try:
        result = await social_handler.get_shared_outfit(share_id, user_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/social/shared/{share_id}/rate")
async def rate_shared_outfit(
    share_id: str,
    request: RateOutfitRequest,
    user_id: str = Depends(get_current_user)
):
    try:
        result = await social_handler.rate_shared_outfit(user_id, share_id, request.rating, request.review)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/social/shared/{share_id}/like")
async def like_shared_outfit(
    share_id: str,
    user_id: str = Depends(get_current_user)
):
    try:
        result = await social_handler.like_shared_outfit(user_id, share_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/social/popular")
async def get_popular_shared_outfits(
    limit: int = 20,
    offset: int = 0,
    occasion: Optional[str] = None,
    style_tags: Optional[str] = None,
    min_rating: Optional[float] = None
):
    try:
        filter_options = {}
        if occasion:
            filter_options["occasion"] = occasion
        if style_tags:
            filter_options["style_tags"] = style_tags.split(",")
        if min_rating:
            filter_options["min_rating"] = min_rating
        
        result = await social_handler.get_popular_shared_outfits(limit, offset, filter_options)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/social/user/shares")
async def get_user_shared_outfits(
    limit: int = 20,
    offset: int = 0,
    user_id: str = Depends(get_current_user)
):
    try:
        result = await social_handler.get_user_shared_outfits(user_id, limit, offset)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/api/social/shared/{share_id}")
async def delete_shared_outfit(
    share_id: str,
    user_id: str = Depends(get_current_user)
):
    try:
        result = await social_handler.delete_shared_outfit(user_id, share_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/social/stats")
async def get_social_stats(
    user_id: str = Depends(get_current_user)
):
    try:
        result = await social_handler.get_social_stats(user_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )