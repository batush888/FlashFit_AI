import { Component, ReactNode } from 'react';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error?: Error;
}

class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: any) {
    console.error('ErrorBoundary caught an error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <div className="min-h-screen bg-gray-50 flex items-center justify-center">
          <div className="max-w-md w-full text-center">
            <div className="text-6xl text-red-400 mb-4">⚠️</div>
            <h1 className="text-2xl font-bold text-gray-900 mb-2">出现错误</h1>
            <p className="text-gray-600 mb-6">
              抱歉，应用程序遇到了一个错误。请刷新页面重试。
            </p>
            <div className="space-x-4">
              <button 
                onClick={() => window.location.reload()}
                className="inline-block bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700"
              >
                刷新页面
              </button>
              <button 
                onClick={() => this.setState({ hasError: false })}
                className="inline-block border border-gray-300 text-gray-700 px-6 py-2 rounded-lg hover:bg-gray-50"
              >
                重试
              </button>
            </div>
            {import.meta.env.DEV && this.state.error && (
              <div className="mt-6 text-left">
                <details className="bg-red-50 border border-red-200 rounded p-4">
                  <summary className="cursor-pointer text-red-800 font-medium">错误详情</summary>
                  <pre className="mt-2 text-xs text-red-700 overflow-auto">
                    {this.state.error.stack}
                  </pre>
                </details>
              </div>
            )}
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;