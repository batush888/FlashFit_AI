import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { User } from '@/types';
import { authService } from '@/api';

interface AuthState {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
}

interface AuthActions {
  setUser: (user: User | null) => void;
  setIsAuthenticated: (isAuthenticated: boolean) => void;
  setLoading: (isLoading: boolean) => void;
  setError: (error: string | null) => void;
  login: (email: string, password: string) => Promise<boolean>;
  register: (userData: {
    username: string;
    email: string;
    password: string;
    fullName?: string;
  }) => Promise<boolean>;
  logout: () => void;
  refreshToken: () => Promise<boolean>;
  updateProfile: (userData: Partial<User>) => Promise<boolean>;
  clearError: () => void;
}

type AuthStore = AuthState & AuthActions;

export const useAuthStore = create<AuthStore>()(
  persist(
    (set, get) => ({
      // 初始状态
      user: null,
      isAuthenticated: false,
      isLoading: false,
      error: null,

      // Actions
      setUser: (user) => set({ user }),
      
      setIsAuthenticated: (isAuthenticated) => set({ isAuthenticated }),
      
      setLoading: (isLoading) => set({ isLoading }),
      
      setError: (error) => set({ error }),
      
      clearError: () => set({ error: null }),

      // 登录
      login: async (email, password) => {
        set({ isLoading: true, error: null });
        
        try {
          const response = await authService.login({ email, password });
          
          if (response.success && response.data) {
            const { user, token } = response.data;
            
            // 存储token
            localStorage.setItem('auth_token', token);
            localStorage.setItem('auth_user', JSON.stringify(user));
            
            set({
              user,
              isAuthenticated: true,
              isLoading: false,
              error: null,
            });
            
            return true;
          } else {
            set({
              isLoading: false,
              error: response.message || '登录失败',
            });
            return false;
          }
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : '登录失败';
          set({
            isLoading: false,
            error: errorMessage,
          });
          return false;
        }
      },

      // 注册
      register: async (userData) => {
        set({ isLoading: true, error: null });
        
        try {
          const response = await authService.register(userData);
          
          if (response.success && response.data) {
            const { user, token } = response.data;
            
            // 存储token
            localStorage.setItem('auth_token', token);
            localStorage.setItem('auth_user', JSON.stringify(user));
            
            set({
              user,
              isAuthenticated: true,
              isLoading: false,
              error: null,
            });
            
            return true;
          } else {
            set({
              isLoading: false,
              error: response.message || '注册失败',
            });
            return false;
          }
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : '注册失败';
          set({
            isLoading: false,
            error: errorMessage,
          });
          return false;
        }
      },

      // 登出
      logout: () => {
        // 清除存储的数据
        localStorage.removeItem('auth_token');
        localStorage.removeItem('auth_user');
        
        set({
          user: null,
          isAuthenticated: false,
          isLoading: false,
          error: null,
        });
      },

      // 刷新token
      refreshToken: async () => {
        try {
          const response = await authService.refreshToken();
          
          if (response.success && response.data) {
            const { token } = response.data;
            localStorage.setItem('auth_token', token);
            return true;
          }
          
          return false;
        } catch (error) {
          console.error('Token刷新失败:', error);
          get().logout();
          return false;
        }
      },

      // 更新用户资料
      updateProfile: async (userData) => {
        set({ isLoading: true, error: null });
        
        try {
          const currentUser = get().user;
          if (!currentUser) {
            throw new Error('用户未登录');
          }

          // 这里应该调用更新用户资料的API
          // const response = await userService.updateProfile(userData);
          
          // 临时实现：直接更新本地状态
          const updatedUser = { ...currentUser, ...userData };
          localStorage.setItem('auth_user', JSON.stringify(updatedUser));
          
          set({
            user: updatedUser,
            isLoading: false,
            error: null,
          });
          
          return true;
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : '更新失败';
          set({
            isLoading: false,
            error: errorMessage,
          });
          return false;
        }
      },
    }),
    {
      name: 'auth-storage',
      partialize: (state) => ({
        user: state.user,
        isAuthenticated: state.isAuthenticated,
      }),
    }
  )
);

// 辅助函数：检查用户是否有特定权限
export const useAuthPermissions = () => {
  const { user } = useAuthStore();
  
  return {
    canUpload: () => !!user,
    canManageWardrobe: () => !!user,
    canViewSuggestions: () => !!user,
    canUpdateProfile: () => !!user,
    isAdmin: () => user?.role === 'admin',
    isPremium: () => user?.isPremium || false,
  };
};

// 辅助函数：获取用户显示名称
export const useUserDisplayName = () => {
  const { user } = useAuthStore();
  
  if (!user) return '游客';
  
  return user.fullName || user.username || user.email.split('@')[0];
};

// 辅助函数：检查认证状态
export const useAuthStatus = () => {
  const { isAuthenticated, isLoading, user } = useAuthStore();
  
  return {
    isAuthenticated,
    isLoading,
    isLoggedIn: isAuthenticated && !!user,
    isGuest: !isAuthenticated,
  };
};