import { useNotificationStore } from '@/stores/notificationStore';

const NotificationContainer = () => {
  const { notifications, removeNotification } = useNotificationStore();

  if (notifications.length === 0) {
    return null;
  }

  return (
    <div className="fixed top-4 right-4 z-50 space-y-2">
      {notifications.map((notification) => (
        <div
          key={notification.id}
          className={`
            max-w-sm w-full bg-white rounded-lg shadow-lg border overflow-hidden
            ${notification.type === 'success' ? 'border-l-4 border-green-500' : ''}
            ${notification.type === 'error' ? 'border-l-4 border-red-500' : ''}
            ${notification.type === 'warning' ? 'border-l-4 border-yellow-500' : ''}
            ${notification.type === 'info' ? 'border-l-4 border-blue-500' : ''}
          `}
        >
          <div className="p-4">
            <div className="flex items-start">
              <div className="flex-shrink-0">
                {notification.type === 'success' && <span className="text-green-500 text-xl">✅</span>}
                {notification.type === 'error' && <span className="text-red-500 text-xl">❌</span>}
                {notification.type === 'warning' && <span className="text-yellow-500 text-xl">⚠️</span>}
                {notification.type === 'info' && <span className="text-blue-500 text-xl">ℹ️</span>}
              </div>
              <div className="ml-3 w-0 flex-1">
                <p className="text-sm font-medium text-gray-900">
                  {notification.title}
                </p>
                <p className="mt-1 text-sm text-gray-500">
                  {notification.message}
                </p>
                {notification.actions && (
                  <div className="mt-3 flex space-x-2">
                    {notification.actions.map((action, index) => (
                      <button
                        key={index}
                        onClick={action.action}
                        className="text-sm bg-gray-100 text-gray-700 px-3 py-1 rounded hover:bg-gray-200"
                      >
                        {action.label}
                      </button>
                    ))}
                  </div>
                )}
              </div>
              <div className="ml-4 flex-shrink-0 flex">
                <button
                  onClick={() => removeNotification(notification.id)}
                  className="bg-white rounded-md inline-flex text-gray-400 hover:text-gray-500 focus:outline-none"
                >
                  <span className="sr-only">关闭</span>
                  <span className="text-lg">×</span>
                </button>
              </div>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
};

export default NotificationContainer;