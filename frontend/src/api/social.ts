import { api, API_ENDPOINTS } from './client';
import { ApiResponse } from '../types';

// 社交功能相关类型定义
export interface ShareOptions {
  description?: string;
  privacy_level: 'public' | 'friends' | 'private';
  allow_comments: boolean;
  tags: string[];
}

export interface SharedOutfit {
  share_id: string;
  user_id: string;
  outfit_id: string;
  title: string;
  description: string;
  items: any[];
  occasion: string;
  style_tags: string[];
  tips: string[];
  collage_url: string;
  privacy_level: string;
  allow_comments: boolean;
  share_tags: string[];
  created_at: string;
  view_count: number;
  like_count: number;
  comment_count: number;
  share_count: number;
  average_rating: number;
  rating_count: number;
  is_featured: boolean;
  is_active: boolean;
  popularity_score?: number;
}

export interface SocialStats {
  shares_count: number;
  total_views: number;
  total_likes: number;
  average_rating: number;
  liked_shares_count: number;
  following_count: number;
  followers_count: number;
  engagement_rate: number;
}

export interface ShareResponse {
  share_id: string;
  share_url: string;
  social_share_urls: {
    facebook: string;
    twitter: string;
    instagram: string;
    pinterest: string;
  };
  shared_outfit: SharedOutfit;
}

/**
 * 社交功能相关API服务
 */
export class SocialService {
  /**
   * 分享搭配到社区
   */
  async shareOutfit(
    outfitData: any,
    shareOptions: ShareOptions
  ): Promise<ApiResponse<ShareResponse>> {
    return await api.post('/api/social/share', {
      outfit_data: outfitData,
      share_options: shareOptions
    });
  }

  /**
   * 获取分享的搭配详情
   */
  async getSharedOutfit(shareId: string): Promise<ApiResponse<SharedOutfit>> {
    return await api.get(`/api/social/shared/${shareId}`);
  }

  /**
   * 为分享的搭配评分
   */
  async rateSharedOutfit(
    shareId: string,
    rating: number,
    review?: string
  ): Promise<ApiResponse<{
    user_rating: number;
    average_rating: number;
    rating_count: number;
  }>> {
    return await api.post(`/api/social/shared/${shareId}/rate`, {
      rating,
      review
    });
  }

  /**
   * 点赞/取消点赞分享的搭配
   */
  async likeSharedOutfit(shareId: string): Promise<ApiResponse<{
    action: 'liked' | 'unliked';
    like_count: number;
    is_liked: boolean;
  }>> {
    return await api.post(`/api/social/shared/${shareId}/like`);
  }

  /**
   * 获取热门分享搭配
   */
  async getPopularSharedOutfits(
    limit: number = 20,
    offset: number = 0,
    filters?: {
      occasion?: string;
      style_tags?: string[];
      min_rating?: number;
    }
  ): Promise<ApiResponse<{
    outfits: SharedOutfit[];
    total: number;
    limit: number;
    offset: number;
    has_more: boolean;
  }>> {
    const params = new URLSearchParams();
    params.append('limit', limit.toString());
    params.append('offset', offset.toString());
    
    if (filters?.occasion) {
      params.append('occasion', filters.occasion);
    }
    if (filters?.style_tags && filters.style_tags.length > 0) {
      params.append('style_tags', filters.style_tags.join(','));
    }
    if (filters?.min_rating) {
      params.append('min_rating', filters.min_rating.toString());
    }

    return await api.get(`/api/social/popular?${params.toString()}`);
  }

  /**
   * 获取用户的分享搭配列表
   */
  async getUserSharedOutfits(
    limit: number = 20,
    offset: number = 0
  ): Promise<ApiResponse<{
    shares: SharedOutfit[];
    total: number;
    limit: number;
    offset: number;
    has_more: boolean;
  }>> {
    const params = new URLSearchParams();
    params.append('limit', limit.toString());
    params.append('offset', offset.toString());

    return await api.get(`/api/social/user/shares?${params.toString()}`);
  }

  /**
   * 删除分享的搭配
   */
  async deleteSharedOutfit(shareId: string): Promise<ApiResponse<void>> {
    return await api.delete(`/api/social/shared/${shareId}`);
  }

  /**
   * 获取用户的社交统计数据
   */
  async getSocialStats(): Promise<ApiResponse<SocialStats>> {
    return await api.get('/api/social/stats');
  }

  /**
   * 生成社交媒体分享链接
   */
  generateSocialShareUrl(platform: string, shareUrl: string, title: string): string {
    const encodedUrl = encodeURIComponent(`https://flashfit.ai${shareUrl}`);
    const encodedTitle = encodeURIComponent(title);
    
    switch (platform) {
      case 'facebook':
        return `https://www.facebook.com/sharer/sharer.php?u=${encodedUrl}`;
      case 'twitter':
        return `https://twitter.com/intent/tweet?url=${encodedUrl}&text=${encodedTitle}`;
      case 'pinterest':
        return `https://pinterest.com/pin/create/button/?url=${encodedUrl}&description=${encodedTitle}`;
      case 'linkedin':
        return `https://www.linkedin.com/sharing/share-offsite/?url=${encodedUrl}`;
      case 'whatsapp':
        return `https://wa.me/?text=${encodedTitle}%20${encodedUrl}`;
      default:
        return shareUrl;
    }
  }

  /**
   * 复制分享链接到剪贴板
   */
  async copyShareLink(shareUrl: string): Promise<boolean> {
    try {
      const fullUrl = `https://flashfit.ai${shareUrl}`;
      await navigator.clipboard.writeText(fullUrl);
      return true;
    } catch (error) {
      console.error('Failed to copy share link:', error);
      return false;
    }
  }

  /**
   * 验证分享选项
   */
  validateShareOptions(shareOptions: ShareOptions): { isValid: boolean; errors: string[] } {
    const errors: string[] = [];

    if (!['public', 'friends', 'private'].includes(shareOptions.privacy_level)) {
      errors.push('隐私级别必须是 public、friends 或 private');
    }

    if (typeof shareOptions.allow_comments !== 'boolean') {
      errors.push('允许评论必须是布尔值');
    }

    if (!Array.isArray(shareOptions.tags)) {
      errors.push('标签必须是数组');
    }

    if (shareOptions.description && shareOptions.description.length > 500) {
      errors.push('描述不能超过500个字符');
    }

    return {
      isValid: errors.length === 0,
      errors
    };
  }

  /**
   * 格式化分享数据用于显示
   */
  formatSharedOutfitForDisplay(sharedOutfit: SharedOutfit) {
    return {
      ...sharedOutfit,
      formatted_created_at: new Date(sharedOutfit.created_at).toLocaleDateString('zh-CN'),
      engagement_rate: sharedOutfit.view_count > 0 
        ? Math.round((sharedOutfit.like_count / sharedOutfit.view_count) * 100)
        : 0,
      rating_display: sharedOutfit.rating_count > 0
        ? `${sharedOutfit.average_rating.toFixed(1)} (${sharedOutfit.rating_count})`
        : '暂无评分'
    };
  }

  /**
   * 获取分享统计摘要
   */
  getShareStatsSummary(stats: SocialStats) {
    return {
      total_engagement: stats.total_likes + stats.total_views,
      avg_views_per_share: stats.shares_count > 0 ? Math.round(stats.total_views / stats.shares_count) : 0,
      avg_likes_per_share: stats.shares_count > 0 ? Math.round(stats.total_likes / stats.shares_count) : 0,
      social_influence_score: Math.round(
        (stats.followers_count * 0.3 + 
         stats.total_likes * 0.4 + 
         stats.engagement_rate * 0.3)
      )
    };
  }
}

// 创建服务实例
export const socialService = new SocialService();

// 便捷函数导出
export const shareOutfit = socialService.shareOutfit.bind(socialService);
export const getSharedOutfit = socialService.getSharedOutfit.bind(socialService);
export const rateSharedOutfit = socialService.rateSharedOutfit.bind(socialService);
export const likeSharedOutfit = socialService.likeSharedOutfit.bind(socialService);
export const getPopularSharedOutfits = socialService.getPopularSharedOutfits.bind(socialService);
export const getUserSharedOutfits = socialService.getUserSharedOutfits.bind(socialService);
export const deleteSharedOutfit = socialService.deleteSharedOutfit.bind(socialService);
export const getSocialStats = socialService.getSocialStats.bind(socialService);
export const generateSocialShareUrl = socialService.generateSocialShareUrl.bind(socialService);
export const copyShareLink = socialService.copyShareLink.bind(socialService);
export const validateShareOptions = socialService.validateShareOptions.bind(socialService);
export const formatSharedOutfitForDisplay = socialService.formatSharedOutfitForDisplay.bind(socialService);
export const getShareStatsSummary = socialService.getShareStatsSummary.bind(socialService);