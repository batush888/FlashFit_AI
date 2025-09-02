/**
 * Generative Feedback Service
 * Handles user feedback for generative recommendations
 */

export interface GenerativeFeedbackRequest {
  suggestion_id: string;
  query_embedding: number[];
  generated_embedding: number[];
  rating: number; // 0.0 to 1.0
  feedback_type?: string;
  context?: Record<string, any>;
  model_version?: string;
}

export interface GenerativeFeedbackResponse {
  status: string;
  feedback_id: string;
  queued_for_training: boolean;
  message?: string;
}

export interface FeedbackStats {
  total_feedback: number;
  average_rating: number;
  rating_std: number;
  feedback_distribution: Record<string, number>;
  recent_feedback_count: number;
  training_queue_size: number;
}

export interface UserPreferences {
  preferences: Record<string, Record<string, string[]>>;
  confidence: number;
  total_feedback: number;
  average_rating: number;
}

export interface TrainingStatus {
  training_queue_size: number;
  total_feedback: number;
  recent_feedback: number;
  min_feedback_for_training: number;
  ready_for_training: boolean;
}

class GenerativeFeedbackService {
  private baseUrl = 'http://localhost:8080/api/generative/feedback';

  private async getAuthHeaders(): Promise<HeadersInit> {
    const token = localStorage.getItem('token');
    return {
      'Content-Type': 'application/json',
      'Authorization': token ? `Bearer ${token}` : '',
    };
  }

  /**
   * Submit feedback for a generative recommendation
   */
  async submitFeedback(request: GenerativeFeedbackRequest): Promise<GenerativeFeedbackResponse> {
    try {
      const response = await fetch(`${this.baseUrl}/submit`, {
        method: 'POST',
        headers: await this.getAuthHeaders(),
        body: JSON.stringify(request),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error submitting generative feedback:', error);
      throw error;
    }
  }

  /**
   * Quick like action for a suggestion
   */
  async likeSuggestion(
    suggestionId: string,
    queryEmbedding: number[],
    generatedEmbedding: number[],
    context?: Record<string, any>
  ): Promise<{ status: string; message: string; feedback_id: string }> {
    try {
      const url = new URL(`${this.baseUrl}/like/${suggestionId}`);
      
      const response = await fetch(url.toString(), {
        method: 'POST',
        headers: await this.getAuthHeaders(),
        body: JSON.stringify({
          query_embedding: queryEmbedding,
          generated_embedding: generatedEmbedding,
          context: context || {},
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error liking suggestion:', error);
      throw error;
    }
  }

  /**
   * Quick dislike action for a suggestion
   */
  async dislikeSuggestion(
    suggestionId: string,
    queryEmbedding: number[],
    generatedEmbedding: number[],
    context?: Record<string, any>
  ): Promise<{ status: string; message: string; feedback_id: string }> {
    try {
      const url = new URL(`${this.baseUrl}/dislike/${suggestionId}`);
      
      const response = await fetch(url.toString(), {
        method: 'POST',
        headers: await this.getAuthHeaders(),
        body: JSON.stringify({
          query_embedding: queryEmbedding,
          generated_embedding: generatedEmbedding,
          context: context || {},
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error disliking suggestion:', error);
      throw error;
    }
  }

  /**
   * Get user feedback statistics
   */
  async getFeedbackStats(): Promise<FeedbackStats> {
    try {
      const response = await fetch(`${this.baseUrl}/stats`, {
        method: 'GET',
        headers: await this.getAuthHeaders(),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error getting feedback stats:', error);
      throw error;
    }
  }

  /**
   * Get global feedback statistics
   */
  async getGlobalFeedbackStats(): Promise<FeedbackStats> {
    try {
      const response = await fetch(`${this.baseUrl}/stats/global`, {
        method: 'GET',
        headers: await this.getAuthHeaders(),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error getting global feedback stats:', error);
      throw error;
    }
  }

  /**
   * Get learned user preferences
   */
  async getUserPreferences(): Promise<UserPreferences> {
    try {
      const response = await fetch(`${this.baseUrl}/preferences`, {
        method: 'GET',
        headers: await this.getAuthHeaders(),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error getting user preferences:', error);
      throw error;
    }
  }

  /**
   * Get training status
   */
  async getTrainingStatus(): Promise<TrainingStatus> {
    try {
      const response = await fetch(`${this.baseUrl}/training/status`, {
        method: 'GET',
        headers: await this.getAuthHeaders(),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error getting training status:', error);
      throw error;
    }
  }
}

export const generativeFeedbackService = new GenerativeFeedbackService();
export default generativeFeedbackService;