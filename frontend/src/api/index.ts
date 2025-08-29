// API客户端
export { api, apiClient, API_ENDPOINTS, ApiClient } from './client';

// 认证服务
export {
  AuthService,
  authService,
  register,
  login,
  logout,
  refreshToken,
  getCurrentUser,
  isAuthenticated,
  getStoredUser,
  getStoredToken,
  validateToken,
  autoRefreshToken,
} from './auth';

// 上传服务
export {
  UploadService,
  uploadService,
  uploadImage,
  uploadBatch,
  deleteImage,
  compressImage,
  createPreviewUrl,
  revokePreviewUrl,
  createFileFromUrl,
  getImageMetadata,
} from './upload';

// 衣橱服务
export {
  WardrobeService,
  wardrobeService,
  getWardrobe,
  getItem,
  updateItem,
  deleteItem,
  searchItems,
  getCategories,
  getFavorites,
  bulkUpdateTags,
  getItemsByCategory,
  getItemsByColor,
  getItemsByTags,
  getRecentItems,
  getSuggestedTags,
  getColorStats,
  itemExists,
  validateItem,
  formatItemForDisplay,
} from './wardrobe';

// 搭配服务
export {
  MatchService,
  matchService,
  generateSuggestions,
  getMatchHistory,
  saveSuggestion,
  deleteSuggestion,
  generateByOccasion,
  quickMatch,
  getPopularSuggestions,
  getHistoryByOccasion,
  searchSuggestions,
  getMatchStats,
  validateMatchRequest,
  formatSuggestionForDisplay,
} from './match';

// 反馈服务
export {
  FeedbackService,
  feedbackService,
  submitFeedback,
  getFeedbackHistory,
  getFeedbackStats,
  likeSuggestion,
  dislikeSuggestion,
  toggleFeedback,
  deleteFeedback,
  getFeedbackAnalysis,
  getSuggestionFeedback,
  validateFeedbackRequest,
  formatFeedbackForDisplay,
  getFeedbackSuggestions,
} from './feedback';

// 社交服务
export {
  SocialService,
  socialService,
  shareOutfit,
  getSharedOutfit,
  rateSharedOutfit,
  likeSharedOutfit,
  getPopularSharedOutfits,
  getUserSharedOutfits,
  deleteSharedOutfit,
  getSocialStats,
  generateSocialShareUrl,
  copyShareLink,
  validateShareOptions,
  formatSharedOutfitForDisplay,
  getShareStatsSummary,
} from './social';

// 导入服务实例用于集合
import { authService } from './auth';
import { uploadService } from './upload';
import { wardrobeService } from './wardrobe';
import { matchService } from './match';
import { feedbackService } from './feedback';
import { socialService } from './social';

// 便捷的API服务集合
export const apiServices = {
  auth: authService,
  upload: uploadService,
  wardrobe: wardrobeService,
  match: matchService,
  feedback: feedbackService,
  social: socialService,
} as const;

// 类型导出
export type { ApiResponse } from '../types';