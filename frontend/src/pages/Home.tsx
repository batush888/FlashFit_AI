const Home = () => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="container mx-auto px-4 py-16">
        <div className="text-center">
          <h1 className="text-5xl font-bold text-gray-900 mb-6">
            FlashFit AI
          </h1>
          <p className="text-xl text-gray-600 mb-8 max-w-2xl mx-auto">
            智能穿搭助手，让AI帮你搭配出完美的造型
          </p>
          <div className="space-x-4">
            <a 
              href="/login" 
              className="inline-block bg-blue-600 text-white px-8 py-3 rounded-lg font-semibold hover:bg-blue-700 transition-colors"
            >
              开始使用
            </a>
            <a 
              href="/register" 
              className="inline-block border border-blue-600 text-blue-600 px-8 py-3 rounded-lg font-semibold hover:bg-blue-50 transition-colors"
            >
              注册账号
            </a>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Home;