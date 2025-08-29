import { api, API_ENDPOINTS } from './client';
import {
  LoginRequest,
  RegisterRequest,
  AuthResponse,
  User,
  ApiResponse,
} from '../types';

/**
 * 认证相关API服务
 */
export class AuthService {
  /**
   * 用户注册
   */
  async register(data: RegisterRequest): Promise<ApiResponse<AuthResponse>> {
    const response = await api.post<AuthResponse>(
      API_ENDPOINTS.AUTH.REGISTER,
      data
    );
    
    if (response.success && response.data) {
      // 保存token和用户信息
      this.saveAuthData(response.data);
    }
    
    return response;
  }

  /**
   * 用户登录
   */
  async login(data: LoginRequest): Promise<ApiResponse<AuthResponse>> {
    const response = await api.post<AuthResponse>(
      API_ENDPOINTS.AUTH.LOGIN,
      data
    );
    
    if (response.success && response.data) {
      // 保存token和用户信息
      this.saveAuthData(response.data);
    }
    
    return response;
  }

  /**
   * 用户登出
   */
  async logout(): Promise<void> {
    try {
      // 调用后端登出接口（如果需要）
      await api.post(API_ENDPOINTS.AUTH.LOGOUT);
    } catch (error) {
      console.warn('Logout API call failed:', error);
    } finally {
      // 清除本地存储的认证信息
      this.clearAuthData();
    }
  }

  /**
   * 刷新token
   */
  async refreshToken(): Promise<ApiResponse<AuthResponse>> {
    const response = await api.post<AuthResponse>(API_ENDPOINTS.AUTH.REFRESH);
    
    if (response.success && response.data) {
      this.saveAuthData(response.data);
    } else {
      // 刷新失败，清除认证信息
      this.clearAuthData();
    }
    
    return response;
  }

  /**
   * 获取当前用户信息
   */
  async getCurrentUser(): Promise<ApiResponse<User>> {
    return await api.get<User>(API_ENDPOINTS.USER.PROFILE);
  }

  /**
   * 检查是否已认证
   */
  isAuthenticated(): boolean {
    return api.isAuthenticated() && !!this.getStoredUser();
  }

  /**
   * 获取存储的用户信息
   */
  getStoredUser(): User | null {
    try {
      const userData = localStorage.getItem('user_data');
      return userData ? JSON.parse(userData) : null;
    } catch (error) {
      console.error('Failed to parse stored user data:', error);
      return null;
    }
  }

  /**
   * 获取存储的token
   */
  getStoredToken(): string | null {
    return api.getAuthToken();
  }

  /**
   * 保存认证数据
   */
  private saveAuthData(authData: AuthResponse): void {
    // 保存token
    api.setAuthToken(authData.token);
    
    // 保存用户信息
    localStorage.setItem('user_data', JSON.stringify(authData.user));
  }

  /**
   * 清除认证数据
   */
  private clearAuthData(): void {
    // 清除token
    api.clearAuthToken();
    
    // 清除用户信息
    localStorage.removeItem('user_data');
    
    // 清除其他相关数据
    localStorage.removeItem('wardrobe_cache');
    localStorage.removeItem('suggestions_cache');
  }

  /**
   * 验证token是否有效
   */
  async validateToken(): Promise<boolean> {
    if (!this.isAuthenticated()) {
      return false;
    }

    try {
      const response = await this.getCurrentUser();
      return response.success;
    } catch (error) {
      console.error('Token validation failed:', error);
      this.clearAuthData();
      return false;
    }
  }

  /**
   * 自动刷新token（如果需要）
   */
  async autoRefreshToken(): Promise<boolean> {
    try {
      const response = await this.refreshToken();
      return response.success;
    } catch (error) {
      console.error('Auto refresh token failed:', error);
      this.clearAuthData();
      return false;
    }
  }
}

// 导出单例实例
export const authService = new AuthService();

// 导出便捷方法
export const {
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
} = authService;