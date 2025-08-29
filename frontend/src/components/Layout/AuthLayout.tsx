interface AuthLayoutProps {
  children: any;
}

const AuthLayout = ({ children }: AuthLayoutProps) => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-md w-full space-y-8">
        {/* Logo */}
        <div className="text-center">
          <div className="mx-auto h-12 w-12 bg-blue-600 rounded-lg flex items-center justify-center">
            <span className="text-white font-bold text-xl">F</span>
          </div>
          <h2 className="mt-6 text-3xl font-extrabold text-gray-900">
            FlashFit AI
          </h2>
          <p className="mt-2 text-sm text-gray-600">
            智能穿搭助手
          </p>
        </div>
        
        {/* Content */}
        <div className="bg-white rounded-lg shadow-xl p-8">
          {children}
        </div>
        
        {/* Footer */}
        <div className="text-center text-sm text-gray-500">
          <p>&copy; 2024 FlashFit AI. 保留所有权利。</p>
        </div>
      </div>
    </div>
  );
};

export default AuthLayout;