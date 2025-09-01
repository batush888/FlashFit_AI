import { nanoid } from 'nanoid';
import { useState, useEffect, useCallback } from 'react';

interface Notification {
  id: string;
  type: 'success' | 'error' | 'warning' | 'info';
  title: string;
  message: string;
  duration?: number;
  persistent?: boolean;
  actions?: Array<{
    label: string;
    action: () => void;
  }>;
  createdAt: number;
}

interface NotificationState {
  notifications: Notification[];
}

interface NotificationActions {
  addNotification: (notification: Omit<Notification, 'id' | 'createdAt'>) => void;
  removeNotification: (id: string) => void;
  clearNotifications: () => void;
  clearNotificationsByType: (type: Notification['type']) => void;
}

type NotificationStore = NotificationState & NotificationActions;

// Global notification store for cross-component notifications
class NotificationStoreImpl {
  private listeners: Array<(notifications: Notification[]) => void> = [];
  private notifications: Notification[] = [];

  subscribe(listener: (notifications: Notification[]) => void) {
    this.listeners.push(listener);
    return () => {
      const index = this.listeners.indexOf(listener);
      if (index > -1) {
        this.listeners.splice(index, 1);
      }
    };
  }

  getNotifications(): Notification[] {
    return this.notifications;
  }

  addNotification = (notification: Omit<Notification, 'id' | 'createdAt'>) => {
    const id = nanoid();
    const newNotification: Notification = {
      ...notification,
      id,
      createdAt: Date.now(),
      duration: notification.duration ?? (notification.persistent ? undefined : 5000),
    };

    this.notifications = [...this.notifications, newNotification];
    this.notifyListeners();

    // Auto-remove non-persistent notifications
    if (!newNotification.persistent && newNotification.duration) {
      setTimeout(() => {
        this.removeNotification(newNotification.id);
      }, newNotification.duration);
    }
  };

  removeNotification = (id: string) => {
    this.notifications = this.notifications.filter(n => n.id !== id);
    this.notifyListeners();
  };

  clearNotifications = () => {
    this.notifications = [];
    this.notifyListeners();
  };

  clearNotificationsByType = (type: Notification['type']) => {
    this.notifications = this.notifications.filter(n => n.type !== type);
    this.notifyListeners();
  };

  private notifyListeners() {
    this.listeners.forEach(listener => listener(this.notifications));
  }
}

const notificationStoreImpl = new NotificationStoreImpl();

// React Hook with proper state management
export const useNotificationStore = (): NotificationStore => {
  const [notifications, setNotifications] = useState<Notification[]>(
    notificationStoreImpl.getNotifications()
  );

  useEffect(() => {
    const unsubscribe = notificationStoreImpl.subscribe(setNotifications);
    return unsubscribe;
  }, []);

  const addNotification = useCallback((notification: Omit<Notification, 'id' | 'createdAt'>) => {
    notificationStoreImpl.addNotification(notification);
  }, []);

  const removeNotification = useCallback((id: string) => {
    notificationStoreImpl.removeNotification(id);
  }, []);

  const clearNotifications = useCallback(() => {
    notificationStoreImpl.clearNotifications();
  }, []);

  const clearNotificationsByType = useCallback((type: Notification['type']) => {
    notificationStoreImpl.clearNotificationsByType(type);
  }, []);

  return {
    notifications,
    addNotification,
    removeNotification,
    clearNotifications,
    clearNotificationsByType,
  };
};

// 辅助函数：快速创建不同类型的通知
export const createNotification = {
  success: (title: string, message: string, options?: Partial<Notification>) => ({
    type: 'success' as const,
    title,
    message,
    ...options,
  }),
  
  error: (title: string, message: string, options?: Partial<Notification>) => ({
    type: 'error' as const,
    title,
    message,
    persistent: true, // 错误通知默认持久化
    ...options,
  }),
  
  warning: (title: string, message: string, options?: Partial<Notification>) => ({
    type: 'warning' as const,
    title,
    message,
    ...options,
  }),
  
  info: (title: string, message: string, options?: Partial<Notification>) => ({
    type: 'info' as const,
    title,
    message,
    ...options,
  }),
};

// Global notification functions for use outside React components
export const notify = {
  success: (title: string, message: string, options?: Partial<Notification>) => {
    notificationStoreImpl.addNotification(createNotification.success(title, message, options));
  },
  
  error: (title: string, message: string, options?: Partial<Notification>) => {
    notificationStoreImpl.addNotification(createNotification.error(title, message, options));
  },
  
  warning: (title: string, message: string, options?: Partial<Notification>) => {
    notificationStoreImpl.addNotification(createNotification.warning(title, message, options));
  },
  
  info: (title: string, message: string, options?: Partial<Notification>) => {
    notificationStoreImpl.addNotification(createNotification.info(title, message, options));
  },
};

// Export the store instance for direct access if needed
export { notificationStoreImpl };