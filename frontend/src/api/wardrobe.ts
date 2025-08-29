import { api, API_ENDPOINTS } from './client';
import {
  WardrobeItem,
  WardrobeFilters,
  WardrobeStats,
  ApiResponse,
} from '../types';

/**
 * 衣橱管理相关API服务
 */
export class WardrobeService {
  /**
   * 获取用户衣橱列表
   */
  async getWardrobe(): Promise<ApiResponse<{
    items: WardrobeItem[];
    stats: WardrobeStats;
  }>> {
    return await api.get(API_ENDPOINTS.WARDROBE.LIST);
  }

  /**
   * 获取单个衣橱物品
   */
  async getItem(itemId: string): Promise<ApiResponse<WardrobeItem>> {
    return await api.get(API_ENDPOINTS.WARDROBE.ITEM(itemId));
  }

  /**
   * 更新衣橱物品
   */
  async updateItem(
    itemId: string,
    updates: Partial<Pick<WardrobeItem, 'tags' | 'garment_type'>>
  ): Promise<ApiResponse<WardrobeItem>> {
    return await api.put(API_ENDPOINTS.WARDROBE.ITEM(itemId), updates);
  }

  /**
   * 删除衣橱物品
   */
  async deleteItem(itemId: string): Promise<ApiResponse<void>> {
    return await api.delete(API_ENDPOINTS.WARDROBE.ITEM(itemId));
  }

  /**
   * 搜索衣橱物品
   */
  async searchItems(filters: WardrobeFilters): Promise<ApiResponse<WardrobeItem[]>> {
    const params = new URLSearchParams();
    
    if (filters.query) {
      params.append('query', filters.query);
    }
    if (filters.category) {
      params.append('category', filters.category);
    }
    if (filters.color) {
      params.append('color', filters.color);
    }
    if (filters.tags && filters.tags.length > 0) {
      params.append('tags', filters.tags.join(','));
    }

    const url = `${API_ENDPOINTS.WARDROBE.SEARCH}?${params.toString()}`;
    return await api.get<WardrobeItem[]>(url);
  }

  /**
   * 获取服装分类统计
   */
  async getCategories(): Promise<ApiResponse<Record<string, number>>> {
    return await api.get<Record<string, number>>(API_ENDPOINTS.WARDROBE.CATEGORIES);
  }

  /**
   * 获取收藏的物品
   */
  async getFavorites(): Promise<ApiResponse<WardrobeItem[]>> {
    return await api.get<WardrobeItem[]>(API_ENDPOINTS.WARDROBE.FAVORITES);
  }

  /**
   * 批量更新物品标签
   */
  async bulkUpdateTags(
    itemIds: string[],
    tags: string[]
  ): Promise<ApiResponse<{ updated_count: number }>> {
    return await api.post(API_ENDPOINTS.WARDROBE.BULK_UPDATE, {
      item_ids: itemIds,
      tags,
    });
  }

  /**
   * 按分类获取物品
   */
  async getItemsByCategory(category: string): Promise<ApiResponse<WardrobeItem[]>> {
    return await this.searchItems({ category });
  }

  /**
   * 按颜色获取物品
   */
  async getItemsByColor(color: string): Promise<ApiResponse<WardrobeItem[]>> {
    return await this.searchItems({ color });
  }

  /**
   * 按标签获取物品
   */
  async getItemsByTags(tags: string[]): Promise<ApiResponse<WardrobeItem[]>> {
    return await this.searchItems({ tags });
  }

  /**
   * 获取最近上传的物品
   */
  async getRecentItems(limit: number = 10): Promise<ApiResponse<WardrobeItem[]>> {
    const response = await this.getWardrobe();
    if (response.success && response.data) {
      // 按创建时间排序，取最新的几个
      const sortedItems = response.data.items
        .sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime())
        .slice(0, limit);
      
      return {
        success: true,
        data: sortedItems,
      };
    }
    return {
      success: false,
      error: response.error || '获取最近物品失败',
      data: [],
    };
  }

  /**
   * 获取推荐标签
   */
  async getSuggestedTags(itemId?: string): Promise<ApiResponse<string[]>> {
    // 基于现有物品的标签生成推荐
    const response = await this.getWardrobe();
    if (response.success && response.data) {
      const allTags = response.data.items
        .flatMap(item => item.tags)
        .filter(tag => tag.trim() !== '');
      
      // 统计标签频率
      const tagCounts = allTags.reduce((acc, tag) => {
        acc[tag] = (acc[tag] || 0) + 1;
        return acc;
      }, {} as Record<string, number>);
      
      // 按频率排序，返回前10个
      const suggestedTags = Object.entries(tagCounts)
        .sort(([, a], [, b]) => b - a)
        .slice(0, 10)
        .map(([tag]) => tag);
      
      return {
        success: true,
        data: suggestedTags,
      };
    }
    
    return {
      success: false,
      error: '无法获取推荐标签',
      data: [],
    };
  }

  /**
   * 获取颜色统计
   */
  async getColorStats(): Promise<ApiResponse<Record<string, number>>> {
    const response = await this.getWardrobe();
    if (response.success && response.data) {
      return {
        success: true,
        data: response.data.stats.colors,
      };
    }
    return {
      success: false,
      error: response.error || '获取颜色统计失败',
      data: {},
    };
  }

  /**
   * 检查物品是否存在
   */
  async itemExists(itemId: string): Promise<boolean> {
    try {
      const response = await this.getItem(itemId);
      return response.success;
    } catch (error) {
      return false;
    }
  }

  /**
   * 验证物品数据
   */
  validateItem(item: Partial<WardrobeItem>): {
    isValid: boolean;
    errors: string[];
  } {
    const errors: string[] = [];

    if (item.tags) {
      // 验证标签
      if (item.tags.length > 20) {
        errors.push('标签数量不能超过20个');
      }
      
      const invalidTags = item.tags.filter(tag => 
        !tag || tag.trim().length === 0 || tag.length > 50
      );
      
      if (invalidTags.length > 0) {
        errors.push('标签不能为空且长度不能超过50个字符');
      }
    }

    if (item.garment_type) {
      const validTypes = [
        'shirt', 'pants', 'jacket', 'dress', 'skirt', 
        'shoes', 'accessory', 'top', 'bottom', 'outerwear'
      ];
      
      if (!validTypes.includes(item.garment_type)) {
        errors.push('无效的服装类型');
      }
    }

    return {
      isValid: errors.length === 0,
      errors,
    };
  }

  /**
   * 格式化物品数据用于显示
   */
  formatItemForDisplay(item: WardrobeItem): {
    displayName: string;
    categoryLabel: string;
    colorLabel: string;
    tagsText: string;
    uploadDate: string;
  } {
    const categoryLabels: Record<string, string> = {
      shirt: '上衣',
      pants: '裤子',
      jacket: '外套',
      dress: '连衣裙',
      skirt: '裙子',
      shoes: '鞋子',
      accessory: '配饰',
      top: '上装',
      bottom: '下装',
      outerwear: '外套',
    };

    const colorLabels: Record<string, string> = {
      red: '红色',
      blue: '蓝色',
      green: '绿色',
      yellow: '黄色',
      black: '黑色',
      white: '白色',
      gray: '灰色',
      brown: '棕色',
      pink: '粉色',
      purple: '紫色',
      orange: '橙色',
    };

    return {
      displayName: item.filename.replace(/\.[^/.]+$/, ''), // 移除文件扩展名
      categoryLabel: categoryLabels[item.garment_type] || item.garment_type,
      colorLabel: colorLabels[item.dominant_color] || item.dominant_color,
      tagsText: item.tags.join(', '),
      uploadDate: new Date(item.created_at).toLocaleDateString('zh-CN'),
    };
  }
}

// 导出单例实例
export const wardrobeService = new WardrobeService();

// 导出便捷方法
export const {
  getWardrobe,
  getItem,
  updateItem,
  deleteItem,
  searchItems,
  getCategories,
  getFavorites,
  bulkUpdateTags,
  getItemsByCategory,
  getItemsByColor,
  getItemsByTags,
  getRecentItems,
  getSuggestedTags,
  getColorStats,
  itemExists,
  validateItem,
  formatItemForDisplay,
} = wardrobeService;