import { useState, useEffect } from 'react';

interface ThemeState {
  theme: 'light' | 'dark' | 'system';
  isDark: boolean;
}

interface ThemeActions {
  setTheme: (theme: 'light' | 'dark' | 'system') => void;
  toggleTheme: () => void;
  initializeTheme: () => void;
}

type ThemeStore = ThemeState & ThemeActions;

// 简化的状态管理实现
class ThemeStoreImpl {
  private state: ThemeState = {
    theme: 'system',
    isDark: false,
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

  getState(): ThemeState {
    return this.state;
  }

  setTheme = (theme: 'light' | 'dark' | 'system') => {
    this.state = {
      ...this.state,
      theme,
      isDark: this.calculateIsDark(theme),
    };
    localStorage.setItem('theme', theme);
    this.applyTheme();
    this.notify();
  };

  toggleTheme = () => {
    const newTheme = this.state.theme === 'light' ? 'dark' : 'light';
    this.setTheme(newTheme);
  };

  initializeTheme = () => {
    const savedTheme = localStorage.getItem('theme') as 'light' | 'dark' | 'system' | null;
    const theme = savedTheme || 'system';
    
    this.state = {
      theme,
      isDark: this.calculateIsDark(theme),
    };
    
    this.applyTheme();
    this.notify();
  };

  private calculateIsDark(theme: 'light' | 'dark' | 'system'): boolean {
    if (theme === 'system') {
      return window.matchMedia('(prefers-color-scheme: dark)').matches;
    }
    return theme === 'dark';
  }

  private applyTheme() {
    const root = document.documentElement;
    if (this.state.isDark) {
      root.classList.add('dark');
    } else {
      root.classList.remove('dark');
    }
  }
}

const themeStoreImpl = new ThemeStoreImpl();

// React Hook
export const useThemeStore = (): ThemeStore => {
  const [, forceUpdate] = useState({});
  
  useEffect(() => {
    const unsubscribe = themeStoreImpl.subscribe(() => {
      forceUpdate({});
    });
    return unsubscribe;
  }, []);

  const state = themeStoreImpl.getState();
  
  return {
    ...state,
    setTheme: themeStoreImpl.setTheme,
    toggleTheme: themeStoreImpl.toggleTheme,
    initializeTheme: themeStoreImpl.initializeTheme,
  };
};

// 监听系统主题变化
if (typeof window !== 'undefined') {
  const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
  mediaQuery.addEventListener('change', () => {
    const currentTheme = themeStoreImpl.getState().theme;
    if (currentTheme === 'system') {
      themeStoreImpl.initializeTheme();
    }
  });
}