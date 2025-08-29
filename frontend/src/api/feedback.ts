import { api, API_ENDPOINTS } from './client';
import {
  FeedbackRequest,
  FeedbackResponse,
  ApiResponse,
} from '../types';

/**
 * 反馈相关API服务
 */
export class FeedbackService {
  /**
   * 提交反馈
   */
  async submitFeedback(
    request: FeedbackRequest
  ): Promise<ApiResponse<FeedbackResponse>> {
    return await api.post<FeedbackResponse>(
      API_ENDPOINTS.FEEDBACK.SUBMIT,
      request
    );
  }

  /**
   * 获取反馈历史
   */
  async getFeedbackHistory(
    limit?: number,
    offset?: number
  ): Promise<ApiResponse<{
    feedbacks: Array<{
      id: string;
      suggestion_id: string;
      liked: boolean;
      notes?: string;
      created_at: string;
    }>;
    total: number;
    has_more: boolean;
  }>> {
    const params = new URLSearchParams();
    if (limit) params.append('limit', limit.toString());
    if (offset) params.append('offset', offset.toString());
    
    const url = `${API_ENDPOINTS.FEEDBACK.HISTORY}?${params.toString()}`;
    return await api.get(url);
  }

  /**
   * 获取反馈统计
   */
  async getFeedbackStats(
    suggestionId?: string
  ): Promise<ApiResponse<{
    total_feedbacks: number;
    positive_feedbacks: number;
    negative_feedbacks: number;
    positive_rate: number;
    recent_feedbacks: number;
  }>> {
    const params = new URLSearchParams();
    if (suggestionId) params.append('suggestion_id', suggestionId);
    
    const url = `${API_ENDPOINTS.FEEDBACK.STATS}?${params.toString()}`;
    return await api.get(url);
  }

  /**
   * 点赞搭配建议
   */
  async likeSuggestion(
    suggestionId: string,
    notes?: string
  ): Promise<ApiResponse<FeedbackResponse>> {
    return await this.submitFeedback({
      suggestion_id: suggestionId,
      liked: true,
      notes,
    });
  }

  /**
   * 不喜欢搭配建议
   */
  async dislikeSuggestion(
    suggestionId: string,
    notes?: string
  ): Promise<ApiResponse<FeedbackResponse>> {
    return await this.submitFeedback({
      suggestion_id: suggestionId,
      liked: false,
      notes,
    });
  }

  /**
   * 切换反馈状态
   */
  async toggleFeedback(
    suggestionId: string,
    currentLiked?: boolean,
    notes?: string
  ): Promise<ApiResponse<FeedbackResponse>> {
    return await this.submitFeedback({
      suggestion_id: suggestionId,
      liked: !currentLiked,
      notes,
    });
  }

  /**
   * 删除反馈
   */
  async deleteFeedback(
    feedbackId: string
  ): Promise<ApiResponse<void>> {
    return await api.delete(`${API_ENDPOINTS.FEEDBACK.SUBMIT}/${feedbackId}`);
  }

  /**
   * 获取用户的反馈偏好分析
   */
  async getFeedbackAnalysis(): Promise<ApiResponse<{
    total_feedbacks: number;
    liked_categories: Record<string, number>;
    disliked_categories: Record<string, number>;
    preferred_occasions: string[];
    feedback_trends: Array<{
      date: string;
      positive: number;
      negative: number;
    }>;
  }>> {
    const response = await this.getFeedbackHistory(1000); // 获取大量数据用于分析
    
    if (response.success && response.data) {
      const feedbacks = response.data.feedbacks;
      
      // 分析反馈数据
      const analysis = {
        total_feedbacks: feedbacks.length,
        liked_categories: {} as Record<string, number>,
        disliked_categories: {} as Record<string, number>,
        preferred_occasions: [] as string[],
        feedback_trends: [] as Array<{
          date: string;
          positive: number;
          negative: number;
        }>,
      };
      
      // 这里可以根据实际需求进行更复杂的分析
      // 目前返回基础统计
      const positiveCount = feedbacks.filter(f => f.liked).length;
      const negativeCount = feedbacks.filter(f => !f.liked).length;
      
      // 按日期分组统计趋势
      const dateGroups = feedbacks.reduce((acc, feedback) => {
        const date = new Date(feedback.created_at).toISOString().split('T')[0];
        if (!acc[date]) {
          acc[date] = { positive: 0, negative: 0 };
        }
        if (feedback.liked) {
          acc[date].positive++;
        } else {
          acc[date].negative++;
        }
        return acc;
      }, {} as Record<string, { positive: number; negative: number }>);
      
      analysis.feedback_trends = Object.entries(dateGroups)
        .map(([date, counts]) => ({ date, ...counts }))
        .sort((a, b) => a.date.localeCompare(b.date))
        .slice(-30); // 最近30天
      
      return {
        success: true,
        data: analysis,
      };
    }
    
    return {
      success: false,
      error: response.error || '获取反馈分析失败',
      data: {
        total_feedbacks: 0,
        liked_categories: {},
        disliked_categories: {},
        preferred_occasions: [],
        feedback_trends: [],
      },
    };
  }

  /**
   * 获取建议的反馈状态
   */
  async getSuggestionFeedback(
    suggestionId: string
  ): Promise<ApiResponse<{
    user_feedback?: {
      liked: boolean;
      notes?: string;
      created_at: string;
    };
    total_likes: number;
    total_dislikes: number;
  }>> {
    const response = await this.getFeedbackStats(suggestionId);
    
    if (response.success && response.data) {
      // 获取用户自己的反馈
      const historyResponse = await this.getFeedbackHistory(100);
      let userFeedback = undefined;
      
      if (historyResponse.success && historyResponse.data) {
        const userFeedbackItem = historyResponse.data.feedbacks.find(
          f => f.suggestion_id === suggestionId
        );
        
        if (userFeedbackItem) {
          userFeedback = {
            liked: userFeedbackItem.liked,
            notes: userFeedbackItem.notes,
            created_at: userFeedbackItem.created_at,
          };
        }
      }
      
      return {
        success: true,
        data: {
          user_feedback: userFeedback,
          total_likes: response.data.positive_feedbacks,
          total_dislikes: response.data.negative_feedbacks,
        },
      };
    }
    
    return {
      success: false,
      error: response.error || '获取反馈状态失败',
      data: {
        total_likes: 0,
        total_dislikes: 0,
      },
    };
  }

  /**
   * 验证反馈请求
   */
  validateFeedbackRequest(request: FeedbackRequest): {
    isValid: boolean;
    errors: string[];
  } {
    const errors: string[] = [];
    
    if (!request.suggestion_id || request.suggestion_id.trim() === '') {
      errors.push('必须指定搭配建议ID');
    }
    
    if (typeof request.liked !== 'boolean') {
      errors.push('必须指定是否喜欢');
    }
    
    if (request.notes && request.notes.length > 500) {
      errors.push('反馈备注不能超过500个字符');
    }
    
    return {
      isValid: errors.length === 0,
      errors,
    };
  }

  /**
   * 格式化反馈用于显示
   */
  formatFeedbackForDisplay(feedback: {
    id: string;
    suggestion_id: string;
    liked: boolean;
    notes?: string;
    created_at: string;
  }): {
    statusText: string;
    statusIcon: string;
    dateText: string;
    hasNotes: boolean;
    notesPreview: string;
  } {
    return {
      statusText: feedback.liked ? '喜欢' : '不喜欢',
      statusIcon: feedback.liked ? '👍' : '👎',
      dateText: new Date(feedback.created_at).toLocaleDateString('zh-CN'),
      hasNotes: !!feedback.notes,
      notesPreview: feedback.notes 
        ? feedback.notes.length > 50 
          ? feedback.notes.substring(0, 50) + '...'
          : feedback.notes
        : '',
    };
  }

  /**
   * 获取反馈建议
   */
  getFeedbackSuggestions(): string[] {
    return [
      '颜色搭配很棒',
      '风格很适合我',
      '很实用的搭配',
      '颜色不太合适',
      '风格不是我喜欢的',
      '搭配太复杂了',
      '很有创意',
      '经典的搭配',
      '适合这个场合',
      '不适合我的身材',
    ];
  }
}

// 导出单例实例
export const feedbackService = new FeedbackService();

// 导出便捷方法
export const {
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
} = feedbackService;