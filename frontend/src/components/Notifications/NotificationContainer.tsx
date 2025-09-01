import { useNotificationStore } from '@/stores/notificationStore';

const NotificationContainer = () => {
  const { notifications, removeNotification } = useNotificationStore();

  if (notifications.length === 0) {
    return null;
  }

  return (
    <div className="fixed top-0 left-0 right-0 z-50 p-4">
      <div className="max-w-4xl mx-auto space-y-2">
        {notifications.map((notification) => (
          <div
            key={notification.id}
            className={`
              w-full bg-white rounded-lg shadow-lg border-l-4 overflow-hidden transform transition-all duration-300 ease-in-out
              ${notification.type === 'success' ? 'border-green-500 bg-green-50' : ''}
              ${notification.type === 'error' ? 'border-red-500 bg-red-50' : ''}
              ${notification.type === 'warning' ? 'border-yellow-500 bg-yellow-50' : ''}
              ${notification.type === 'info' ? 'border-blue-500 bg-blue-50' : ''}
            `}
          >
            <div className="px-4 py-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <div className="flex-shrink-0">
                    {notification.type === 'success' && <span className="text-green-600 text-lg">✅</span>}
                    {notification.type === 'error' && <span className="text-red-600 text-lg">❌</span>}
                    {notification.type === 'warning' && <span className="text-yellow-600 text-lg">⚠️</span>}
                    {notification.type === 'info' && <span className="text-blue-600 text-lg">ℹ️</span>}
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center space-x-2">
                      <p className="text-sm font-semibold text-gray-900">
                        {notification.title}
                      </p>
                      <span className="text-gray-400">•</span>
                      <p className="text-sm text-gray-600">
                        {notification.message}
                      </p>
                    </div>
                    {notification.actions && (
                      <div className="mt-2 flex space-x-2">
                        {notification.actions.map((action, index) => (
                          <button
                            key={index}
                            onClick={action.action}
                            className="text-xs bg-white text-gray-700 px-2 py-1 rounded border hover:bg-gray-50 transition-colors"
                          >
                            {action.label}
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
                <button
                  onClick={() => removeNotification(notification.id)}
                  className="flex-shrink-0 ml-4 bg-transparent rounded-md inline-flex text-gray-400 hover:text-gray-600 focus:outline-none transition-colors"
                >
                  <span className="sr-only">关闭</span>
                  <span className="text-xl font-light">×</span>
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default NotificationContainer;