const NotFound = () => {
  return (
    <div className="min-h-screen bg-gray-50 flex items-center justify-center">
      <div className="max-w-md w-full text-center">
        <div className="text-6xl text-gray-400 mb-4">404</div>
        <h1 className="text-2xl font-bold text-gray-900 mb-2">页面未找到</h1>
        <p className="text-gray-600 mb-6">
          抱歉，您访问的页面不存在。
        </p>
        <div className="space-x-4">
          <a 
            href="/" 
            className="inline-block bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700"
          >
            返回首页
          </a>
          <button 
            onClick={() => window.history.back()}
            className="inline-block border border-gray-300 text-gray-700 px-6 py-2 rounded-lg hover:bg-gray-50"
          >
            返回上页
          </button>
        </div>
      </div>
    </div>
  );
};

export default NotFound;