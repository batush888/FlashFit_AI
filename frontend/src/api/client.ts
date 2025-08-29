import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse } from 'axios';
import { ApiResponse } from '../types';

// API基础配置
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8080';
const API_TIMEOUT = 30000; // 30秒超时

// 创建axios实例
const apiClient: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: API_TIMEOUT,
  headers: {
    'Content-Type': 'application/json',
  },
});

// 请求拦截器 - 添加认证token
apiClient.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('auth_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// 响应拦截器 - 统一错误处理
apiClient.interceptors.response.use(
  (response: AxiosResponse) => {
    return response;
  },
  (error) => {
    // 处理401未授权错误
    if (error.response?.status === 401) {
      localStorage.removeItem('auth_token');
      localStorage.removeItem('user_data');
      window.location.href = '/login';
    }
    
    // 处理网络错误
    if (!error.response) {
      error.message = '网络连接失败，请检查网络设置';
    }
    
    return Promise.reject(error);
  }
);

// API请求封装类
export class ApiClient {
  private client: AxiosInstance;

  constructor() {
    this.client = apiClient;
  }

  // GET请求
  async get<T = any>(
    url: string,
    config?: AxiosRequestConfig
  ): Promise<ApiResponse<T>> {
    try {
      const response = await this.client.get<ApiResponse<T>>(url, config);
      return response.data;
    } catch (error: any) {
      return this.handleError(error);
    }
  }

  // POST请求
  async post<T = any>(
    url: string,
    data?: any,
    config?: AxiosRequestConfig
  ): Promise<ApiResponse<T>> {
    try {
      const response = await this.client.post<ApiResponse<T>>(url, data, config);
      return response.data;
    } catch (error: any) {
      return this.handleError(error);
    }
  }

  // PUT请求
  async put<T = any>(
    url: string,
    data?: any,
    config?: AxiosRequestConfig
  ): Promise<ApiResponse<T>> {
    try {
      const response = await this.client.put<ApiResponse<T>>(url, data, config);
      return response.data;
    } catch (error: any) {
      return this.handleError(error);
    }
  }

  // DELETE请求
  async delete<T = any>(
    url: string,
    config?: AxiosRequestConfig
  ): Promise<ApiResponse<T>> {
    try {
      const response = await this.client.delete<ApiResponse<T>>(url, config);
      return response.data;
    } catch (error: any) {
      return this.handleError(error);
    }
  }

  // 文件上传
  async upload<T = any>(
    url: string,
    formData: FormData,
    onProgress?: (progress: number) => void
  ): Promise<ApiResponse<T>> {
    try {
      const response = await this.client.post<ApiResponse<T>>(url, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          if (onProgress && progressEvent.total) {
            const progress = Math.round(
              (progressEvent.loaded * 100) / progressEvent.total
            );
            onProgress(progress);
          }
        },
      });
      return response.data;
    } catch (error: any) {
      return this.handleError(error);
    }
  }

  // 错误处理
  private handleError(error: any): ApiResponse {
    const message = error.response?.data?.message || error.message || '请求失败';
    const status = error.response?.status || 500;
    
    console.error('API Error:', {
      message,
      status,
      url: error.config?.url,
      method: error.config?.method,
    });

    return {
      success: false,
      error: message,
      data: null,
    };
  }

  // 设置认证token
  setAuthToken(token: string): void {
    localStorage.setItem('auth_token', token);
    this.client.defaults.headers.Authorization = `Bearer ${token}`;
  }

  // 清除认证token
  clearAuthToken(): void {
    localStorage.removeItem('auth_token');
    delete this.client.defaults.headers.Authorization;
  }

  // 获取当前token
  getAuthToken(): string | null {
    return localStorage.getItem('auth_token');
  }

  // 检查是否已认证
  isAuthenticated(): boolean {
    return !!this.getAuthToken();
  }
}

// 导出单例实例
export const api = new ApiClient();

// 导出axios实例供特殊用途
export { apiClient };

// 导出常用配置
export const API_ENDPOINTS = {
  // 认证相关
  AUTH: {
    REGISTER: '/api/auth/register',
    LOGIN: '/api/auth/login',
    REFRESH: '/api/auth/refresh',
    LOGOUT: '/api/auth/logout',
  },
  // 用户相关
  USER: {
    PROFILE: '/api/user/profile',
    UPDATE: '/api/user/update',
    DELETE: '/api/user/delete',
    STATS: '/api/user/stats',
    DELETE_IMAGE: '/api/user/delete_image',
  },
  // 上传相关
  UPLOAD: {
    IMAGE: '/api/upload',
    BATCH: '/api/upload/batch',
  },
  // 衣橱相关
  WARDROBE: {
    LIST: '/api/wardrobe',
    ITEM: (id: string) => `/api/wardrobe/${id}`,
    SEARCH: '/api/wardrobe/search',
    CATEGORIES: '/api/wardrobe/categories',
    FAVORITES: '/api/wardrobe/favorites',
    BULK_UPDATE: '/api/wardrobe/bulk-update',
  },
  // 搭配相关
  MATCH: {
    GENERATE: '/api/match',
    HISTORY: '/api/match/history',
    SAVE: '/api/match/save',
  },
  // 反馈相关
  FEEDBACK: {
    SUBMIT: '/api/feedback',
    HISTORY: '/api/feedback/history',
    STATS: '/api/feedback/stats',
  },
} as const;