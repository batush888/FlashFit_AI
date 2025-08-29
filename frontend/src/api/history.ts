import { api } from './client';
import { ApiResponse } from '../types';

// 类型定义
export interface OutfitHistoryItem {
  id: string;
  outfit_id: string;
  title: string;
  items: any[];
  occasion: string;
  style_tags: string[];
  tips: string[];
  similarity_score: number;
  created_at: string;
  is_favorite: boolean;
  is_worn: boolean;
  wear_count: number;
  last_worn?: string;
  user_rating?: number;
  user_notes: string;
}

export interface OutfitCollection {
  id: string;
  name: string;
  description: string;
  outfit_ids: string[];
  created_at: string;
  updated_at: string;
}

export interface OutfitStatistics {
  total_outfits: number;
  favorite_count: number;
  worn_count: number;
  collection_count: number;
  occasion_distribution: Record<string, number>;
  style_distribution: Record<string, number>;
  recent_outfits: OutfitHistoryItem[];
  wear_rate: number;
}

// API 端点
const API_ENDPOINTS = {
  HISTORY: {
    SAVE: '/api/history/save',
    LIST: '/api/history',
    FAVORITE: '/api/history/favorite',
    FAVORITES: '/api/history/favorites',
    WORN: '/api/history/worn',
    STATISTICS: '/api/history/statistics'
  },
  COLLECTIONS: {
    CREATE: '/api/collections/create',
    LIST: '/api/collections',
    ADD: '/api/collections/add'
  }
};

export class HistoryAPI {
  /**
   * 保存搭配到历史记录
   */
  async saveOutfitToHistory(outfitData: any): Promise<ApiResponse<{ success: boolean; message: string; history_id: string }>> {
    return await api.post(API_ENDPOINTS.HISTORY.SAVE, {
      outfit_data: outfitData
    });
  }

  /**
   * 获取搭配历史
   */
  async getOutfitHistory(
    limit: number = 20,
    offset: number = 0
  ): Promise<ApiResponse<{
    outfits: OutfitHistoryItem[];
    total: number;
    limit: number;
    offset: number;
    has_more: boolean;
  }>> {
    const params = new URLSearchParams();
    params.append('limit', limit.toString());
    params.append('offset', offset.toString());
    
    return await api.get(`${API_ENDPOINTS.HISTORY.LIST}?${params.toString()}`);
  }

  /**
   * 切换搭配收藏状态
   */
  async toggleFavoriteOutfit(historyId: string): Promise<ApiResponse<{
    success: boolean;
    is_favorite: boolean;
    message: string;
  }>> {
    return await api.post(API_ENDPOINTS.HISTORY.FAVORITE, {
      history_id: historyId
    });
  }

  /**
   * 获取收藏的搭配
   */
  async getFavoriteOutfits(): Promise<ApiResponse<{
    favorites: OutfitHistoryItem[];
    total: number;
  }>> {
    return await api.get(API_ENDPOINTS.HISTORY.FAVORITES);
  }

  /**
   * 标记搭配为已穿着
   */
  async markOutfitWorn(
    historyId: string,
    wornDate?: string
  ): Promise<ApiResponse<{
    success: boolean;
    message: string;
    wear_count: number;
  }>> {
    return await api.post(API_ENDPOINTS.HISTORY.WORN, {
      history_id: historyId,
      worn_date: wornDate
    });
  }

  /**
   * 创建搭配集合
   */
  async createOutfitCollection(
    collectionName: string,
    description: string = ''
  ): Promise<ApiResponse<{
    success: boolean;
    message: string;
    collection_id: string;
  }>> {
    return await api.post(API_ENDPOINTS.COLLECTIONS.CREATE, {
      collection_name: collectionName,
      description: description
    });
  }

  /**
   * 添加搭配到集合
   */
  async addOutfitToCollection(
    collectionId: string,
    historyId: string
  ): Promise<ApiResponse<{
    success: boolean;
    message: string;
  }>> {
    return await api.post(API_ENDPOINTS.COLLECTIONS.ADD, {
      collection_id: collectionId,
      history_id: historyId
    });
  }

  /**
   * 获取用户的搭配集合
   */
  async getUserCollections(): Promise<ApiResponse<{
    collections: OutfitCollection[];
    total: number;
  }>> {
    return await api.get(API_ENDPOINTS.COLLECTIONS.LIST);
  }

  /**
   * 获取搭配统计信息
   */
  async getOutfitStatistics(): Promise<ApiResponse<OutfitStatistics>> {
    return await api.get(API_ENDPOINTS.HISTORY.STATISTICS);
  }
}

// 导出单例实例
export const historyAPI = new HistoryAPI();