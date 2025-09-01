import { ReactNode, ComponentType } from 'react';

// 用户相关类型
export interface User {
  id: string;
  email: string;
  created_at: string;
  consent_given: boolean;
}

export interface AuthResponse {
  token: string;
  user: User;
}

export interface LoginRequest {
  email: string;
  password: string;
}

export interface RegisterRequest {
  email: string;
  password: string;
}

// 服装相关类型
export interface WardrobeItem {
  id: string;
  item_id: string;
  filename: string;
  url: string;
  garment_type: string;
  garment_type_cn: string;
  dominant_color: string;
  style_keywords: string[];
  tags: string[];
  created_at: string;
  upload_time: string;
  colors?: Array<{
    rgb: number[];
    hex: string;
    name: string;
    name_cn: string;
    percentage: number;
  }>;
  embeddings?: number[];
}

export interface UploadResponse {
  item_id: string;
  url: string;
  garment_type: string;
  colors: string[];
  embeddings: number[];
}

// 搭配建议相关类型
export interface OutfitSuggestion {
  id: string;
  title_cn: string;
  tips_cn: string[];
  items: WardrobeItem[];
  occasion: string;
  similarity_score: number;
  collage_url?: string;
  created_at: string;
}

export interface MatchRequest {
  item_id: string;
  occasion?: string;
  target_count?: number;
}

export interface MatchResponse {
  suggestions: OutfitSuggestion[];
}

// 反馈相关类型
export interface FeedbackRequest {
  suggestion_id: string;
  liked: boolean;
  notes?: string;
}

export interface FeedbackResponse {
  success: boolean;
  message: string;
}

// API响应基础类型
export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  message?: string;
  error?: string;
}

// 上传相关类型
export interface UploadProgress {
  loaded: number;
  total: number;
  percentage: number;
}

// 搜索和过滤类型
export interface WardrobeFilters {
  category?: string;
  color?: string;
  tags?: string[];
  query?: string;
}

// 统计数据类型
export interface WardrobeStats {
  total_items: number;
  categories: Record<string, number>;
  colors: Record<string, number>;
  recent_uploads: number;
}

// 应用状态类型
export interface AppState {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
}

// 组件Props类型
export interface BaseComponentProps {
  className?: string;
  children?: ReactNode;
}

// 表单状态类型
export interface FormState<T> {
  data: T;
  errors: Partial<Record<keyof T, string>>;
  isSubmitting: boolean;
  isValid: boolean;
}

// 模态框类型
export interface ModalProps extends BaseComponentProps {
  isOpen: boolean;
  onClose: () => void;
  title?: string;
}

// 通知类型
export interface Notification {
  id: string;
  type: 'success' | 'error' | 'warning' | 'info';
  title: string;
  message?: string;
  duration?: number;
  action?: {
    label: string;
    onClick: () => void;
  };
}

// 路由类型
export interface RouteConfig {
  path: string;
  component: ComponentType;
  title: string;
  requiresAuth?: boolean;
  showInNav?: boolean;
}

// 主题类型
export interface Theme {
  colors: {
    primary: string;
    secondary: string;
    accent: string;
    background: string;
    surface: string;
    text: string;
    textSecondary: string;
    border: string;
    error: string;
    warning: string;
    success: string;
    info: string;
  };
  spacing: {
    xs: string;
    sm: string;
    md: string;
    lg: string;
    xl: string;
  };
  borderRadius: {
    sm: string;
    md: string;
    lg: string;
    xl: string;
  };
  shadows: {
    sm: string;
    md: string;
    lg: string;
  };
}

// 设备类型
export type DeviceType = 'mobile' | 'tablet' | 'desktop';

// 语言类型
export type Language = 'zh-CN' | 'en-US';

// 服装类别枚举
export enum GarmentCategory {
  SHIRT = 'shirt',
  PANTS = 'pants',
  JACKET = 'jacket',
  DRESS = 'dress',
  SKIRT = 'skirt',
  SHOES = 'shoes',
  ACCESSORY = 'accessory',
}

// 颜色枚举
export enum Color {
  RED = 'red',
  BLUE = 'blue',
  GREEN = 'green',
  YELLOW = 'yellow',
  BLACK = 'black',
  WHITE = 'white',
  GRAY = 'gray',
  BROWN = 'brown',
  PINK = 'pink',
  PURPLE = 'purple',
  ORANGE = 'orange',
}

// 场合枚举
export enum Occasion {
  CASUAL = 'casual',
  FORMAL = 'formal',
  BUSINESS = 'business',
  PARTY = 'party',
  SPORT = 'sport',
  DATE = 'date',
  TRAVEL = 'travel',
}

// 风格枚举
export enum Style {
  MINIMALIST = 'minimalist',
  VINTAGE = 'vintage',
  BOHEMIAN = 'bohemian',
  CLASSIC = 'classic',
  TRENDY = 'trendy',
  ELEGANT = 'elegant',
  CASUAL = 'casual',
  EDGY = 'edgy',
}