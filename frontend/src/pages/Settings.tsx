const Settings = () => {
  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow p-6">
        <h1 className="text-2xl font-bold text-gray-900 mb-4">设置</h1>
        <p className="text-gray-600">管理您的应用设置</p>
      </div>
      
      <div className="bg-white rounded-lg shadow p-6">
        <div className="space-y-6">
          <div>
            <h3 className="text-lg font-medium text-gray-900 mb-4">主题设置</h3>
            <div className="space-y-2">
              <label className="flex items-center">
                <input type="radio" name="theme" value="light" className="mr-2" />
                浅色主题
              </label>
              <label className="flex items-center">
                <input type="radio" name="theme" value="dark" className="mr-2" />
                深色主题
              </label>
              <label className="flex items-center">
                <input type="radio" name="theme" value="system" className="mr-2" defaultChecked />
                跟随系统
              </label>
            </div>
          </div>
          
          <div>
            <h3 className="text-lg font-medium text-gray-900 mb-4">通知设置</h3>
            <div className="space-y-2">
              <label className="flex items-center">
                <input type="checkbox" className="mr-2" defaultChecked />
                接收穿搭建议通知
              </label>
              <label className="flex items-center">
                <input type="checkbox" className="mr-2" defaultChecked />
                接收新功能通知
              </label>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Settings;