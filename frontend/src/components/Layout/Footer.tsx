const Footer = () => {
  return (
    <footer className="bg-white border-t border-gray-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="py-4">
          <div className="flex flex-col sm:flex-row justify-between items-center">
            <div className="text-sm text-gray-500">
              &copy; 2024 FlashFit AI. 保留所有权利。
            </div>
            <div className="flex space-x-6 mt-2 sm:mt-0">
              <a href="#" className="text-sm text-gray-500 hover:text-gray-700">
                隐私政策
              </a>
              <a href="#" className="text-sm text-gray-500 hover:text-gray-700">
                服务条款
              </a>
              <a href="#" className="text-sm text-gray-500 hover:text-gray-700">
                帮助中心
              </a>
            </div>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;