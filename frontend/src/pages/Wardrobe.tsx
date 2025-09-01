import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { wardrobeService } from '../api/wardrobe';
import { useNotificationStore } from '../stores/notificationStore';
import { WardrobeItem, WardrobeStats } from '../types';

const Wardrobe = () => {
  const [clothingItems, setClothingItems] = useState<WardrobeItem[]>([]);
  const [stats, setStats] = useState<WardrobeStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [categories, setCategories] = useState([
    { id: 'all', name: '全部', count: 0 },
    { id: 'tops', name: '上装', count: 0 },
    { id: 'bottoms', name: '下装', count: 0 },
    { id: 'dresses', name: '连衣裙', count: 0 },
    { id: 'outerwear', name: '外套', count: 0 },
    { id: 'shoes', name: '鞋子', count: 0 },
    { id: 'accessories', name: '配饰', count: 0 },
    { id: 'others', name: '其他', count: 0 }
  ]);
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
  const [deletingItemId, setDeletingItemId] = useState<string | null>(null);
  const [selectedItems, setSelectedItems] = useState<Set<string>>(new Set());
  const [isSelectionMode, setIsSelectionMode] = useState(false);
  const [bulkDeleting, setBulkDeleting] = useState(false);
  const { addNotification } = useNotificationStore();

  useEffect(() => {
    loadWardrobeData();
  }, []);

  const loadWardrobeData = async () => {
    try {
      setLoading(true);
      const response = await wardrobeService.getWardrobe();
      
      if (response.success && response.data) {
        setClothingItems(response.data.items);
        setStats(response.data.stats);
        
        const categoryLabels: Record<string, string> = {
          'tops': '上装',
          'bottoms': '下装', 
          'dresses': '连衣裙',
          'outerwear': '外套',
          'shoes': '鞋子',
          'accessories': '配饰',
          'others': '其他'
        };
        
        const updatedCategories = [
          { id: 'all', name: '全部', count: response.data.stats.total_items },
          ...Object.entries(response.data.stats.categories).map(([key, count]) => ({
            id: key,
            name: categoryLabels[key] || key,
            count: count as number
          }))
        ];
        
        setCategories(updatedCategories);
      }
    } catch (error) {
      console.error('Failed to load wardrobe data:', error);
      addNotification({
        title: '加载失败',
        message: '无法加载衣橱数据，请稍后重试',
        type: 'error'
      });
    } finally {
      setLoading(false);
    }
  };

  const toggleItemSelection = (itemId: string) => {
    const newSelection = new Set(selectedItems);
    if (newSelection.has(itemId)) {
      newSelection.delete(itemId);
    } else {
      newSelection.add(itemId);
    }
    setSelectedItems(newSelection);
  };

  const selectAllItems = () => {
    const allItemIds = new Set(filteredItems.map(item => item.item_id));
    setSelectedItems(allItemIds);
  };

  const clearSelection = () => {
    setSelectedItems(new Set());
  };

  const handleDeleteItem = async (itemId: string, itemName: string) => {
    if (!confirm(`确定要删除 "${itemName}" 吗？此操作无法撤销。`)) {
      return;
    }

    try {
      setDeletingItemId(itemId);
      const response = await wardrobeService.deleteItem(itemId);
      
      if (response.success) {
        setClothingItems(prev => prev.filter(item => item.item_id !== itemId));
        
        // Update stats
        setStats(prev => ({
          ...prev!,
          total_items: prev!.total_items - 1
        }));
        
        addNotification({
          title: '删除成功',
          message: `已删除 "${itemName}"`,
          type: 'success'
        });
      } else {
        throw new Error(response.message || '删除失败');
      }
    } catch (error) {
      console.error('Delete failed:', error);
      addNotification({
        title: '删除失败',
        message: '删除服装时出现错误，请稍后重试',
        type: 'error'
      });
    } finally {
      setDeletingItemId(null);
    }
  };

  const handleBulkDelete = async () => {
    if (selectedItems.size === 0) return;
    
    const itemNames = Array.from(selectedItems).map(id => {
      const item = clothingItems.find(item => item.item_id === id);
      return item ? item.filename.replace(/\.[^/.]+$/, '') : id;
    });
    
    if (!confirm(`确定要删除选中的 ${selectedItems.size} 件服装吗？\n\n${itemNames.join('\n')}\n\n此操作无法撤销。`)) {
      return;
    }

    try {
      setBulkDeleting(true);
      const deletePromises = Array.from(selectedItems).map(itemId =>
        wardrobeService.deleteItem(itemId).then(response => ({
          itemId,
          success: response.success,
          error: response.message
        }))
      );
      
      const results = await Promise.allSettled(deletePromises);
      let successCount = 0;
      let failedItems: string[] = [];
      
      results.forEach((result, index) => {
        const itemId = Array.from(selectedItems)[index];
        const item = clothingItems.find(item => item.item_id === itemId);
        const itemName = item ? item.filename.replace(/\.[^/.]+$/, '') : itemId;
        
        if (result.status === 'fulfilled' && result.value.success) {
          successCount++;
          setClothingItems(prev => prev.filter(item => item.item_id !== itemId));
        } else {
          failedItems.push(itemName);
        }
      });
      
      // Update stats
      setStats(prev => ({
        ...prev!,
        total_items: prev!.total_items - successCount
      }));
      
      // Clear selection
      setSelectedItems(new Set());
      setIsSelectionMode(false);
      
      // Show notification
      if (successCount === selectedItems.size) {
        addNotification({
          title: '批量删除成功',
          message: `已成功删除 ${successCount} 件服装`,
          type: 'success'
        });
      } else if (successCount > 0) {
        addNotification({
          title: '部分删除成功',
          message: `成功删除 ${successCount} 件，失败 ${failedItems.length} 件`,
          type: 'warning'
        });
      } else {
        addNotification({
          title: '批量删除失败',
          message: '所有删除操作都失败了，请稍后重试',
          type: 'error'
        });
      }
    } catch (error) {
      console.error('Bulk delete failed:', error);
      addNotification({
        title: '批量删除失败',
        message: '批量删除时出现错误，请稍后重试',
        type: 'error'
      });
    } finally {
      setBulkDeleting(false);
    }
  };

  const filteredItems = clothingItems.filter(item => {
    if (selectedCategory === 'all') return true;
    return item.garment_type === selectedCategory;
  });

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 p-6">
      {/* Header */}
      <div className="max-w-7xl mx-auto mb-8">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-3xl font-bold text-gray-900 mb-2">我的衣橱</h1>
            <p className="text-gray-600">
              {stats ? `共 ${stats.total_items} 件服装` : '加载中...'}
            </p>
          </div>
          
          <div className="flex items-center gap-4">
            {/* Search */}
            <div className="relative">
              <input
                type="text"
                placeholder="搜索服装..."
                className="pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
              <svg className="absolute left-3 top-2.5 h-5 w-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
            </div>
            
            {/* View Mode Toggle */}
            <button
              onClick={() => setViewMode(viewMode === 'grid' ? 'list' : 'grid')}
              className={`p-2 rounded-lg transition-colors ${
                viewMode === 'grid' 
                  ? 'bg-blue-100 text-blue-600' 
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
            >
              {viewMode === 'grid' ? (
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 10h16M4 14h16M4 18h16" />
                </svg>
              ) : (
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" />
                </svg>
              )}
            </button>
            
            {/* Selection Mode Toggle */}
            <button
              onClick={() => {
                setIsSelectionMode(!isSelectionMode);
                if (isSelectionMode) {
                  setSelectedItems(new Set());
                }
              }}
              className={`px-4 py-2 rounded-lg transition-colors ${
                isSelectionMode
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
            >
              {isSelectionMode ? '取消选择' : '批量选择'}
            </button>
          </div>
        </div>
        
        {/* Selection Controls */}
        {isSelectionMode && (
          <div className="flex items-center gap-4 mb-4">
            <button
              onClick={selectAllItems}
              className="text-blue-600 hover:text-blue-700 text-sm font-medium"
            >
              全选
            </button>
            <button
              onClick={clearSelection}
              className="text-gray-600 hover:text-gray-700 text-sm font-medium"
            >
              清空
            </button>
            {selectedItems.size > 0 && (
              <button
                onClick={handleBulkDelete}
                disabled={bulkDeleting}
                className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 disabled:opacity-50 text-sm font-medium flex items-center gap-2"
              >
                {bulkDeleting ? (
                  <>
                    <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    删除中...
                  </>
                ) : (
                  `删除选中 (${selectedItems.size})`
                )}
              </button>
            )}
          </div>
        )}
        
        {/* Add Button */}
        <div className="mb-6">
          <Link
            to="/upload"
            className="inline-flex items-center px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-xl hover:from-blue-700 hover:to-purple-700 transition-all duration-300 shadow-lg hover:shadow-xl transform hover:-translate-y-0.5"
          >
            <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
            </svg>
            添加服装
          </Link>
        </div>

        {/* Category Filter */}
        <div className="flex flex-wrap gap-2 mt-6">
          {categories.map((category) => (
            <button
              key={category.id}
              onClick={() => setSelectedCategory(category.id)}
              className={`inline-flex items-center px-4 py-2 rounded-full text-sm font-medium transition-all ${
                selectedCategory === category.id
                  ? 'bg-blue-100 text-blue-700 ring-2 ring-blue-500 ring-opacity-20'
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
            >
              {category.name}
              <span className="ml-2 px-2 py-0.5 bg-white/50 rounded-full text-xs">
                {category.count}
              </span>
            </button>
          ))}
        </div>
      </div>

      {/* Items Grid/List */}
      {loading ? (
        <div className="bg-white rounded-2xl shadow-lg p-8 border border-gray-100">
          <div className="flex items-center justify-center py-12">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
            <span className="ml-3 text-gray-600">加载中...</span>
          </div>
        </div>
      ) : filteredItems.length > 0 ? (
        <div className={`${
          viewMode === 'grid'
            ? 'grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6'
            : 'space-y-4'
        }`}>
          {filteredItems.map((item, index) => (
            <div
              key={item.item_id}
              className={`bg-white rounded-2xl shadow-lg hover:shadow-xl transition-all duration-300 border border-gray-100 group animate-slide-up ${
                viewMode === 'list' ? 'flex items-center p-4' : 'overflow-hidden'
              }`}
              style={{ animationDelay: `${index * 100}ms` }}
            >
              {viewMode === 'grid' ? (
                <>
                  <div className="aspect-[4/5] bg-gray-100 relative overflow-hidden">
                    <img 
                      src={`http://localhost:8080${item.url}`}
                      alt={item.filename}
                      className="w-full h-full object-cover"
                      onError={(e) => {
                        const target = e.target as HTMLImageElement;
                        target.style.display = 'none';
                        target.nextElementSibling?.classList.remove('hidden');
                      }}
                    />
                    <div className="hidden w-full h-full bg-gradient-to-br from-gray-200 to-gray-300 flex items-center justify-center">
                      <svg className="w-12 h-12 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                      </svg>
                    </div>
                    <div className="absolute inset-0 bg-gradient-to-t from-black/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity"></div>
                    
                    {/* Selection Checkbox for Grid View */}
                    {isSelectionMode && (
                      <div className="absolute top-3 left-3 z-10">
                        <input
                          type="checkbox"
                          checked={selectedItems.has(item.item_id)}
                          onChange={() => toggleItemSelection(item.item_id)}
                          className="w-5 h-5 text-blue-600 bg-white border-2 border-gray-300 rounded focus:ring-blue-500 focus:ring-2"
                        />
                      </div>
                    )}
                    
                    {!isSelectionMode && (
                      <div className="absolute top-3 right-3 opacity-0 group-hover:opacity-100 transition-opacity flex gap-2">
                        <button className="p-2 bg-white/90 rounded-full shadow-lg hover:bg-white transition-colors">
                          <svg className="w-4 h-4 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" />
                          </svg>
                        </button>
                        <button 
                          onClick={() => handleDeleteItem(item.item_id, item.filename.replace(/\.[^/.]+$/, ''))}
                          disabled={deletingItemId === item.item_id}
                          className="p-2 bg-red-500/90 rounded-full shadow-lg hover:bg-red-600 transition-colors disabled:opacity-50"
                        >
                          {deletingItemId === item.item_id ? (
                            <svg className="w-4 h-4 text-white animate-spin" fill="none" viewBox="0 0 24 24">
                              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                            </svg>
                          ) : (
                            <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                            </svg>
                          )}
                        </button>
                      </div>
                    )}
                  </div>
                  <div className="p-4">
                    <h3 className="font-bold text-gray-900 mb-1 group-hover:text-blue-600 transition-colors">
                      {item.filename.replace(/\.[^/.]+$/, '')}
                    </h3>
                    <p className="text-sm text-gray-500 mb-2">{item.colors?.[0]?.name_cn || item.dominant_color} • {item.garment_type_cn || item.garment_type}</p>
                    <div className="flex flex-wrap gap-1 mb-3">
                      {item.tags.map((tag, tagIndex) => (
                        <span
                          key={tagIndex}
                          className="px-2 py-1 bg-blue-50 text-blue-600 text-xs rounded-full"
                        >
                          {tag}
                        </span>
                      ))}
                    </div>
                    <span className="text-xs text-gray-400">{new Date(item.created_at).toLocaleDateString('zh-CN')}</span>
                  </div>
                </>
              ) : (
                <>
                  {/* Selection Checkbox for List View */}
                  {isSelectionMode && (
                    <div className="flex items-center mr-4">
                      <input
                        type="checkbox"
                        checked={selectedItems.has(item.item_id)}
                        onChange={() => toggleItemSelection(item.item_id)}
                        className="w-5 h-5 text-blue-600 bg-white border-2 border-gray-300 rounded focus:ring-blue-500 focus:ring-2"
                      />
                    </div>
                  )}
                  
                  <div className="w-20 h-20 bg-gray-100 rounded-xl flex-shrink-0 mr-4 relative overflow-hidden">
                    <img 
                      src={`http://localhost:8080${item.url}`}
                      alt={item.filename}
                      className="w-full h-full object-cover rounded-xl"
                      onError={(e) => {
                        const target = e.target as HTMLImageElement;
                        target.style.display = 'none';
                        target.nextElementSibling?.classList.remove('hidden');
                      }}
                    />
                    <div className="hidden w-full h-full bg-gradient-to-br from-gray-200 to-gray-300 flex items-center justify-center rounded-xl">
                      <svg className="w-8 h-8 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                      </svg>
                    </div>
                  </div>
                  <div className="flex-1 min-w-0">
                    <h3 className="font-bold text-gray-900 mb-1 truncate group-hover:text-blue-600 transition-colors">
                      {item.filename.replace(/\.[^/.]+$/, '')}
                    </h3>
                    <p className="text-sm text-gray-500 mb-2">{item.colors?.[0]?.name_cn || item.dominant_color} • {item.garment_type_cn || item.garment_type}</p>
                    <div className="flex flex-wrap gap-1 mb-2">
                      {item.tags.map((tag, tagIndex) => (
                        <span
                          key={tagIndex}
                          className="px-2 py-1 bg-blue-50 text-blue-600 text-xs rounded-full"
                        >
                          {tag}
                        </span>
                      ))}
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-gray-400">{new Date(item.created_at).toLocaleDateString('zh-CN')}</span>
                      <div className="flex items-center gap-2">
                        <button className="text-blue-600 hover:text-blue-700 transition-colors">
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                          </svg>
                        </button>
                        {!isSelectionMode && (
                          <button 
                            onClick={() => handleDeleteItem(item.item_id, item.filename.replace(/\.[^/.]+$/, ''))}
                            disabled={deletingItemId === item.item_id}
                            className="text-red-500 hover:text-red-600 transition-colors disabled:opacity-50"
                          >
                            {deletingItemId === item.item_id ? (
                              <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                              </svg>
                            ) : (
                              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                              </svg>
                            )}
                          </button>
                        )}
                      </div>
                    </div>
                  </div>
                </>
              )}
            </div>
          ))}
        </div>
      ) : (
        <div className="bg-white rounded-2xl shadow-lg p-12 border border-gray-100 text-center">
          <div className="w-24 h-24 mx-auto mb-6 bg-gray-100 rounded-full flex items-center justify-center">
            <svg className="w-12 h-12 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" />
            </svg>
          </div>
          <h3 className="text-xl font-semibold text-gray-900 mb-2">衣橱空空如也</h3>
          <p className="text-gray-500 mb-6">开始添加您的第一件服装吧！</p>
          <Link
            to="/upload"
            className="inline-flex items-center px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-xl hover:from-blue-700 hover:to-purple-700 transition-all duration-300 shadow-lg hover:shadow-xl transform hover:-translate-y-0.5"
          >
            <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
            </svg>
            添加服装
          </Link>
        </div>
      )}
    </div>
  );
};

export default Wardrobe;