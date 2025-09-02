interface SidebarProps {
  isOpen: boolean;
  onClose: () => void;
}

const Sidebar = ({ isOpen, onClose }: SidebarProps) => {
  const menuItems = [
    { name: '仪表板', href: '/dashboard', icon: '📊' },
    { name: '我的衣橱', href: '/wardrobe', icon: '👔' },
    { name: '上传服装', href: '/upload', icon: '📤' },
    { name: '穿搭建议', href: '/suggestions', icon: '✨' },
    { name: '搭配历史', href: '/history', icon: '📚' },
    { name: '社交分享', href: '/social', icon: '🌟' },
    { name: '系统监控', href: '/monitoring', icon: '📈' },
    { name: '个人资料', href: '/profile', icon: '👤' },
    { name: '设置', href: '/settings', icon: '⚙️' },
  ];

  return (
    <>
      {/* Mobile overlay */}
      {isOpen && (
        <div 
          className="fixed inset-0 z-40 lg:hidden" 
          onClick={onClose}
        >
          <div className="fixed inset-0 bg-gray-600 bg-opacity-75"></div>
        </div>
      )}

      {/* Sidebar */}
      <div className={`
        fixed inset-y-0 left-0 z-50 w-64 bg-white shadow-lg transform transition-transform duration-300 ease-in-out lg:translate-x-0 lg:static lg:inset-0
        ${isOpen ? 'translate-x-0' : '-translate-x-full'}
      `}>
        <div className="flex flex-col h-full">
          {/* Sidebar header */}
          <div className="flex items-center justify-between h-16 px-4 border-b border-gray-200 lg:hidden">
            <span className="text-lg font-semibold text-gray-900">菜单</span>
            <button
              onClick={onClose}
              className="p-2 rounded-md text-gray-400 hover:text-gray-500 hover:bg-gray-100"
            >
              ✕
            </button>
          </div>

          {/* Navigation */}
          <nav className="flex-1 px-4 py-4 space-y-2">
            {menuItems.map((item) => (
              <a
                key={item.name}
                href={item.href}
                className="flex items-center px-3 py-2 text-sm font-medium text-gray-700 rounded-md hover:bg-gray-100 hover:text-gray-900"
                onClick={onClose}
              >
                <span className="mr-3 text-lg">{item.icon}</span>
                {item.name}
              </a>
            ))}
          </nav>

          {/* Sidebar footer */}
          <div className="p-4 border-t border-gray-200">
            <button className="flex items-center w-full px-3 py-2 text-sm font-medium text-red-600 rounded-md hover:bg-red-50">
              <span className="mr-3 text-lg">🚪</span>
              退出登录
            </button>
          </div>
        </div>
      </div>
    </>
  );
};

export default Sidebar;