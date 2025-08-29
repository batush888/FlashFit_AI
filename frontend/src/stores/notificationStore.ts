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

// 简化的状态管理实现
class NotificationStoreImpl {
  private state: NotificationState = {
    notifications: [],
  };

  private listeners: Array<() => void> = [];

  private notify() {
    this.listeners.forEach(listener => listener());
  }

  subscribe(listener: () => void) {
    this.listeners.push(listener);
    return () => {
      const index = this.listeners.indexOf(listener);
      if (index > -1) {
        this.listeners.splice(index, 1);
      }
    };
  }

  getState(): NotificationState {
    return this.state;
  }

  addNotification = (notification: Omit<Notification, 'id' | 'createdAt'>) => {
    const id = Math.random().toString(36).substr(2, 9);
    const newNotification: Notification = {
      ...notification,
      id,
      createdAt: Date.now(),
      duration: notification.duration ?? 5000,
    };

    this.state = {
      ...this.state,
      notifications: [...this.state.notifications, newNotification],
    };

    this.notify();

    // 自动移除通知（除非是持久化的）
    if (!newNotification.persistent && newNotification.duration && newNotification.duration > 0) {
      setTimeout(() => {
        this.removeNotification(id);
      }, newNotification.duration);
    }
  };

  removeNotification = (id: string) => {
    this.state = {
      ...this.state,
      notifications: this.state.notifications.filter(n => n.id !== id),
    };
    this.notify();
  };

  clearNotifications = () => {
    this.state = {
      ...this.state,
      notifications: [],
    };
    this.notify();
  };

  clearNotificationsByType = (type: Notification['type']) => {
    this.state = {
      ...this.state,
      notifications: this.state.notifications.filter(n => n.type !== type),
    };
    this.notify();
  };
}

const notificationStoreImpl = new NotificationStoreImpl();

// React Hook
export const useNotificationStore = (): NotificationStore => {
  // 简化实现，不使用React hooks
  const state = notificationStoreImpl.getState();
  
  return {
    ...state,
    addNotification: notificationStoreImpl.addNotification,
    removeNotification: notificationStoreImpl.removeNotification,
    clearNotifications: notificationStoreImpl.clearNotifications,
    clearNotificationsByType: notificationStoreImpl.clearNotificationsByType,
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

// 全局通知函数
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