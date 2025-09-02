import { api, API_ENDPOINTS } from './client';
import {
  MatchRequest,
  MatchResponse,
  OutfitSuggestion,
  ApiResponse,
} from '../types';

/**
 * 搭配建议相关API服务
 */
export class MatchService {
  /**
   * 生成搭配建议
   */
  async generateSuggestions(
    request: MatchRequest
  ): Promise<ApiResponse<MatchResponse>> {
    return await api.post<MatchResponse>(
      API_ENDPOINTS.MATCH.GENERATE,
      request
    );
  }

  /**
   * 获取搭配历史
   */
  async getMatchHistory(
    limit?: number,
    offset?: number
  ): Promise<ApiResponse<{
    suggestions: OutfitSuggestion[];
    total: number;
    has_more: boolean;
  }>> {
    const params = new URLSearchParams();
    if (limit) params.append('limit', limit.toString());
    if (offset) params.append('offset', offset.toString());
    
    const url = `${API_ENDPOINTS.MATCH.HISTORY}?${params.toString()}`;
    return await api.get(url);
  }

  /**
   * 保存搭配建议
   */
  async saveSuggestion(
    suggestion: OutfitSuggestion
  ): Promise<ApiResponse<{ suggestion_id: string }>> {
    return await api.post(API_ENDPOINTS.MATCH.SAVE, suggestion);
  }

  /**
   * 基于图片生成融合推荐
   */
  async generateFusionRecommendations(
    imageFile: File,
    targetCount: number = 5
  ): Promise<ApiResponse<MatchResponse>> {
    const formData = new FormData();
    formData.append('file', imageFile);
    formData.append('target_count', targetCount.toString());

    return await api.upload('/api/fusion/match', formData);
  }

  /**
   * 基于用户ID生成融合推荐
   */
  async generateFusionByUserId(
    userId: string,
    targetCount: number = 5
  ): Promise<ApiResponse<MatchResponse>> {
    const params = new URLSearchParams();
    params.append('user_id', userId);
    params.append('target_count', targetCount.toString());
    
    const url = `/api/fusion/match?${params.toString()}`;
    return await api.get(url);
  }

  /**
   * 删除搭配建议
   */
  async deleteSuggestion(
    suggestionId: string
  ): Promise<ApiResponse<void>> {
    return await api.delete(`${API_ENDPOINTS.MATCH.SAVE}/${suggestionId}`);
  }

  /**
   * 根据场合生成搭配
   */
  async generateByOccasion(
    itemId: string,
    occasion: string,
    targetCount?: number
  ): Promise<ApiResponse<MatchResponse>> {
    return await this.generateSuggestions({
      item_id: itemId,
      occasion,
      target_count: targetCount,
    });
  }

  /**
   * 快速搭配（使用默认参数）
   */
  async quickMatch(
    itemId: string
  ): Promise<ApiResponse<MatchResponse>> {
    return await this.generateSuggestions({
      item_id: itemId,
      target_count: 3, // 默认生成3个建议
    });
  }

  /**
   * 获取热门搭配
   */
  async getPopularSuggestions(
    limit: number = 10
  ): Promise<ApiResponse<OutfitSuggestion[]>> {
    const response = await this.getMatchHistory(limit * 2); // 获取更多数据用于筛选
    
    if (response.success && response.data) {
      // 这里可以根据点赞数、查看次数等指标筛选热门搭配
      // 目前简单按时间排序
      const popularSuggestions = response.data.suggestions
        .sort((a, b) => b.similarity_score - a.similarity_score)
        .slice(0, limit);
      
      return {
        success: true,
        data: popularSuggestions,
      };
    }
    
    return {
      success: false,
      error: response.error || '获取热门搭配失败',
      data: [],
    };
  }

  /**
   * 按场合筛选搭配历史
   */
  async getHistoryByOccasion(
    occasion: string,
    limit?: number
  ): Promise<ApiResponse<OutfitSuggestion[]>> {
    const response = await this.getMatchHistory(limit || 50);
    
    if (response.success && response.data) {
      const filteredSuggestions = response.data.suggestions
        .filter(suggestion => suggestion.occasion === occasion);
      
      return {
        success: true,
        data: filteredSuggestions,
      };
    }
    
    return {
      success: false,
      error: response.error || '获取场合搭配失败',
      data: [],
    };
  }

  /**
   * 搜索搭配建议
   */
  async searchSuggestions(
    query: string,
    filters?: {
      occasion?: string;
      minScore?: number;
      dateFrom?: string;
      dateTo?: string;
    }
  ): Promise<ApiResponse<OutfitSuggestion[]>> {
    const response = await this.getMatchHistory(100); // 获取更多数据用于搜索
    
    if (response.success && response.data) {
      let filteredSuggestions = response.data.suggestions;
      
      // 按标题和提示搜索
      if (query.trim()) {
        const searchTerm = query.toLowerCase();
        filteredSuggestions = filteredSuggestions.filter(suggestion => 
          suggestion.title_cn.toLowerCase().includes(searchTerm) ||
          suggestion.tips_cn.some(tip => tip.toLowerCase().includes(searchTerm))
        );
      }
      
      // 按场合筛选
      if (filters?.occasion) {
        filteredSuggestions = filteredSuggestions.filter(
          suggestion => suggestion.occasion === filters.occasion
        );
      }
      
      // 按相似度分数筛选
      if (filters?.minScore !== undefined) {
        filteredSuggestions = filteredSuggestions.filter(
          suggestion => suggestion.similarity_score >= filters.minScore!
        );
      }
      
      // 按日期范围筛选
      if (filters?.dateFrom || filters?.dateTo) {
        filteredSuggestions = filteredSuggestions.filter(suggestion => {
          const suggestionDate = new Date(suggestion.created_at);
          const fromDate = filters?.dateFrom ? new Date(filters.dateFrom) : null;
          const toDate = filters?.dateTo ? new Date(filters.dateTo) : null;
          
          if (fromDate && suggestionDate < fromDate) return false;
          if (toDate && suggestionDate > toDate) return false;
          return true;
        });
      }
      
      return {
        success: true,
        data: filteredSuggestions,
      };
    }
    
    return {
      success: false,
      error: response.error || '搜索搭配失败',
      data: [],
    };
  }

  /**
   * 获取搭配统计信息
   */
  async getMatchStats(): Promise<ApiResponse<{
    total_suggestions: number;
    occasions: Record<string, number>;
    avg_similarity_score: number;
    recent_matches: number;
  }>> {
    const response = await this.getMatchHistory(1000); // 获取大量数据用于统计
    
    if (response.success && response.data) {
      const suggestions = response.data.suggestions;
      
      // 统计场合分布
      const occasions = suggestions.reduce((acc, suggestion) => {
        acc[suggestion.occasion] = (acc[suggestion.occasion] || 0) + 1;
        return acc;
      }, {} as Record<string, number>);
      
      // 计算平均相似度分数
      const avgScore = suggestions.length > 0 
        ? suggestions.reduce((sum, s) => sum + s.similarity_score, 0) / suggestions.length
        : 0;
      
      // 统计最近7天的搭配数量
      const sevenDaysAgo = new Date();
      sevenDaysAgo.setDate(sevenDaysAgo.getDate() - 7);
      const recentMatches = suggestions.filter(
        s => new Date(s.created_at) >= sevenDaysAgo
      ).length;
      
      return {
        success: true,
        data: {
          total_suggestions: suggestions.length,
          occasions,
          avg_similarity_score: Math.round(avgScore * 100) / 100,
          recent_matches: recentMatches,
        },
      };
    }
    
    return {
      success: false,
      error: response.error || '获取统计信息失败',
      data: {
        total_suggestions: 0,
        occasions: {},
        avg_similarity_score: 0,
        recent_matches: 0,
      },
    };
  }

  /**
   * 验证搭配请求
   */
  validateMatchRequest(request: MatchRequest): {
    isValid: boolean;
    errors: string[];
  } {
    const errors: string[] = [];
    
    if (!request.item_id || request.item_id.trim() === '') {
      errors.push('必须指定要搭配的物品');
    }
    
    if (request.target_count !== undefined) {
      if (request.target_count < 1 || request.target_count > 10) {
        errors.push('搭配建议数量必须在1-10之间');
      }
    }
    
    if (request.occasion) {
      const validOccasions = [
        'casual', 'formal', 'business', 'party', 
        'sport', 'date', 'travel'
      ];
      
      if (!validOccasions.includes(request.occasion)) {
        errors.push('无效的场合类型');
      }
    }
    
    return {
      isValid: errors.length === 0,
      errors,
    };
  }

  /**
   * 格式化搭配建议用于显示
   */
  formatSuggestionForDisplay(suggestion: OutfitSuggestion): {
    displayTitle: string;
    occasionLabel: string;
    scoreText: string;
    itemCount: number;
    createDate: string;
    tipsText: string;
  } {
    const occasionLabels: Record<string, string> = {
      casual: '休闲',
      formal: '正式',
      business: '商务',
      party: '聚会',
      sport: '运动',
      date: '约会',
      travel: '旅行',
    };
    
    return {
      displayTitle: suggestion.title_cn,
      occasionLabel: occasionLabels[suggestion.occasion] || suggestion.occasion,
      scoreText: `${Math.round(suggestion.similarity_score * 100)}%`,
      itemCount: suggestion.items.length,
      createDate: new Date(suggestion.created_at).toLocaleDateString('zh-CN'),
      tipsText: suggestion.tips_cn.join(' • '),
    };
  }
}

// 导出单例实例
export const matchService = new MatchService();

// 导出便捷方法
export const {
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
  generateFusionRecommendations,
  generateFusionByUserId,
} = matchService;