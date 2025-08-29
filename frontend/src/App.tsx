import React, { Suspense, useEffect } from 'react';
import { Routes, Route, Navigate, useLocation } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { authService } from '@/api';
import { useAuthStore } from '@/stores/authStore';
import { useThemeStore } from '@/stores/themeStore';
import { useNotificationStore } from '@/stores/notificationStore';

// Layout组件
import Layout from '@/components/Layout/Layout';
import AuthLayout from '@/components/Layout/AuthLayout';

// 页面组件 - 懒加载
const Home = React.lazy(() => import('@/pages/Home'));
const Login = React.lazy(() => import('@/pages/Auth/Login'));
const Register = React.lazy(() => import('@/pages/Auth/Register'));
const Dashboard = React.lazy(() => import('@/pages/Dashboard'));
const Wardrobe = React.lazy(() => import('@/pages/Wardrobe'));
const Upload = React.lazy(() => import('@/pages/Upload'));
const Suggestions = React.lazy(() => import('@/pages/Suggestions'));
const Profile = React.lazy(() => import('@/pages/Profile'));
const Settings = React.lazy(() => import('@/pages/Settings'));
const History = React.lazy(() => import('@/pages/History'));
const Social = React.lazy(() => import('@/pages/Social'));
const NotFound = React.lazy(() => import('@/pages/NotFound'));

// 加载组件
const LoadingSpinner: React.FC = () => (
  <div className="min-h-screen flex items-center justify-center">
    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
  </div>
);

// 错误边界组件
interface ErrorBoundaryState {
  hasError: boolean;
  error?: Error;
}

class ErrorBoundary extends React.Component<
  React.PropsWithChildren<{}>,
  ErrorBoundaryState
> {
  constructor(props: React.PropsWithChildren<{}>) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('应用错误:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen flex items-center justify-center bg-gray-50">
          <div className="max-w-md w-full bg-white rounded-lg shadow-lg p-6 text-center">
            <div className="text-red-500 text-6xl mb-4">⚠️</div>
            <h1 className="text-2xl font-bold text-gray-900 mb-2">出错了</h1>
            <p className="text-gray-600 mb-4">
              应用遇到了一个错误，请刷新页面重试。
            </p>
            <button
              onClick={() => window.location.reload()}
              className="btn btn-primary"
            >
              刷新页面
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

// 受保护的路由组件
interface ProtectedRouteProps {
  children: React.ReactNode;
}

const ProtectedRoute: React.FC<ProtectedRouteProps> = ({ children }) => {
  const { isAuthenticated } = useAuthStore();
  const location = useLocation();

  if (!isAuthenticated) {
    return <Navigate to="/login" state={{ from: location }} replace />;
  }

  return <>{children}</>;
};

// 公共路由组件（仅未认证用户可访问）
const PublicRoute: React.FC<ProtectedRouteProps> = ({ children }) => {
  const { isAuthenticated } = useAuthStore();

  if (isAuthenticated) {
    return <Navigate to="/dashboard" replace />;
  }

  return <>{children}</>;
};

// 主应用组件
const App: React.FC = () => {
  const { setUser, setIsAuthenticated, logout } = useAuthStore();
  const { theme, initializeTheme } = useThemeStore();
  const { addNotification } = useNotificationStore();

  // 检查用户认证状态
  const { data: currentUser, isLoading, error } = useQuery({
    queryKey: ['currentUser'],
    queryFn: authService.getCurrentUser,
    retry: false,
    staleTime: 5 * 60 * 1000, // 5分钟
    enabled: !!authService.getStoredToken(),
  });

  // 初始化应用
  useEffect(() => {
    // 初始化主题
    initializeTheme();

    // 设置用户认证状态
    if (currentUser?.success && currentUser.data) {
      setUser(currentUser.data);
      setIsAuthenticated(true);
    } else if (error) {
      // 认证失败，清除本地存储
      logout();
      addNotification({
        type: 'error',
        title: '认证失败',
        message: '请重新登录',
      });
    }
  }, [currentUser, error, setUser, setIsAuthenticated, logout, initializeTheme, addNotification]);

  // 应用主题类
  useEffect(() => {
    document.documentElement.className = theme;
  }, [theme]);

  // 全局错误处理
  useEffect(() => {
    const handleUnhandledRejection = (event: PromiseRejectionEvent) => {
      console.error('未处理的Promise拒绝:', event.reason);
      addNotification({
        type: 'error',
        title: '系统错误',
        message: '发生了一个意外错误，请稍后重试',
      });
    };

    const handleError = (event: ErrorEvent) => {
      console.error('全局错误:', event.error);
      addNotification({
        type: 'error',
        title: '系统错误',
        message: '发生了一个意外错误，请稍后重试',
      });
    };

    window.addEventListener('unhandledrejection', handleUnhandledRejection);
    window.addEventListener('error', handleError);

    return () => {
      window.removeEventListener('unhandledrejection', handleUnhandledRejection);
      window.removeEventListener('error', handleError);
    };
  }, [addNotification]);

  // 显示加载状态
  if (isLoading) {
    return <LoadingSpinner />;
  }

  return (
    <ErrorBoundary>
      <div className="App min-h-screen bg-gray-50">
        <Suspense fallback={<LoadingSpinner />}>
          <Routes>
            {/* 公共路由 */}
            <Route path="/" element={<Home />} />
            
            {/* 认证路由 */}
            <Route
              path="/login"
              element={
                <PublicRoute>
                  <AuthLayout>
                    <Login />
                  </AuthLayout>
                </PublicRoute>
              }
            />
            <Route
              path="/register"
              element={
                <PublicRoute>
                  <AuthLayout>
                    <Register />
                  </AuthLayout>
                </PublicRoute>
              }
            />

            {/* 受保护的路由 */}
            <Route
              path="/dashboard"
              element={
                <ProtectedRoute>
                  <Layout>
                    <Dashboard />
                  </Layout>
                </ProtectedRoute>
              }
            />
            <Route
              path="/wardrobe"
              element={
                <ProtectedRoute>
                  <Layout>
                    <Wardrobe />
                  </Layout>
                </ProtectedRoute>
              }
            />
            <Route
              path="/upload"
              element={
                <ProtectedRoute>
                  <Layout>
                    <Upload />
                  </Layout>
                </ProtectedRoute>
              }
            />
            <Route
              path="/suggestions"
              element={
                <ProtectedRoute>
                  <Layout>
                    <Suggestions />
                  </Layout>
                </ProtectedRoute>
              }
            />
            <Route
              path="/history"
              element={
                <ProtectedRoute>
                  <Layout>
                    <History />
                  </Layout>
                </ProtectedRoute>
              }
            />
            <Route
              path="/profile"
              element={
                <ProtectedRoute>
                  <Layout>
                    <Profile />
                  </Layout>
                </ProtectedRoute>
              }
            />
            <Route
              path="/settings"
              element={
                <ProtectedRoute>
                  <Layout>
                    <Settings />
                  </Layout>
                </ProtectedRoute>
              }
            />
            <Route
              path="/social"
              element={
                <ProtectedRoute>
                  <Layout>
                    <Social />
                  </Layout>
                </ProtectedRoute>
              }
            />

            {/* 404页面 */}
            <Route path="*" element={<NotFound />} />
          </Routes>
        </Suspense>
      </div>
    </ErrorBoundary>
  );
};

export default App;