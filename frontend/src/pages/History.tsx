import React, { useState, useEffect } from 'react';
import { historyAPI, OutfitHistoryItem, OutfitStatistics } from '../api/history';
import { Heart, Clock, Star, TrendingUp, Calendar, Archive, Filter } from 'lucide-react';

interface HistoryPageProps {}

const History: React.FC<HistoryPageProps> = () => {
  const [outfits, setOutfits] = useState<OutfitHistoryItem[]>([]);
  const [favorites, setFavorites] = useState<OutfitHistoryItem[]>([]);
  const [statistics, setStatistics] = useState<OutfitStatistics | null>(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<'all' | 'favorites' | 'worn' | 'stats'>('all');
  const [filterOccasion, setFilterOccasion] = useState<string>('all');
  const [page, setPage] = useState(0);
  const [hasMore, setHasMore] = useState(true);

  // 加载搭配历史
  const loadOutfitHistory = async (pageNum: number = 0, reset: boolean = false) => {
    try {
      const response = await historyAPI.getOutfitHistory(20, pageNum * 20);
      if (response.success && response.data) {
        if (reset) {
          setOutfits(response.data.outfits || []);
        } else {
          setOutfits(prev => [...prev, ...(response.data?.outfits || [])]);
        }
        setHasMore(response.data.has_more || false);
      }
    } catch (error) {
      console.error('加载搭配历史失败:', error);
    }
  };

  // 加载收藏搭配
  const loadFavorites = async () => {
    try {
      const response = await historyAPI.getFavoriteOutfits();
      if (response.success && response.data) {
        setFavorites(response.data.favorites || []);
      }
    } catch (error) {
      console.error('加载收藏搭配失败:', error);
    }
  };

  // 加载统计信息
  const loadStatistics = async () => {
    try {
      const response = await historyAPI.getOutfitStatistics();
      if (response.success && response.data) {
        setStatistics(response.data);
      }
    } catch (error) {
      console.error('加载统计信息失败:', error);
    }
  };

  // 切换收藏状态
  const toggleFavorite = async (historyId: string) => {
    try {
      const response = await historyAPI.toggleFavoriteOutfit(historyId);
      if (response.success) {
        // 更新本地状态
        setOutfits(prev => prev.map(outfit => 
          outfit.id === historyId 
            ? { ...outfit, is_favorite: response.data.is_favorite }
            : outfit
        ));
        
        // 如果在收藏页面，重新加载收藏列表
        if (activeTab === 'favorites') {
          loadFavorites();
        }
      }
    } catch (error) {
      console.error('切换收藏状态失败:', error);
    }
  };

  // 标记为已穿着
  const markAsWorn = async (historyId: string) => {
    try {
      const response = await historyAPI.markOutfitWorn(historyId);
      if (response.success) {
        // 更新本地状态
        setOutfits(prev => prev.map(outfit => 
          outfit.id === historyId 
            ? { 
                ...outfit, 
                is_worn: true, 
                wear_count: response.data.wear_count,
                last_worn: new Date().toISOString()
              }
            : outfit
        ));
        
        // 重新加载统计信息
        loadStatistics();
      }
    } catch (error) {
      console.error('标记穿着状态失败:', error);
    }
  };

  // 初始化数据
  useEffect(() => {
    const initData = async () => {
      setLoading(true);
      await Promise.all([
        loadOutfitHistory(0, true),
        loadFavorites(),
        loadStatistics()
      ]);
      setLoading(false);
    };
    
    initData();
  }, []);

  // 加载更多
  const loadMore = () => {
    if (hasMore && !loading) {
      const nextPage = page + 1;
      setPage(nextPage);
      loadOutfitHistory(nextPage);
    }
  };

  // 过滤搭配
  const filteredOutfits = outfits.filter(outfit => {
    if (filterOccasion === 'all') return true;
    return outfit.occasion === filterOccasion;
  });

  // 获取已穿着的搭配
  const wornOutfits = outfits.filter(outfit => outfit.is_worn);

  // 渲染搭配卡片
  const renderOutfitCard = (outfit: OutfitHistoryItem) => (
    <div key={outfit.id} className="bg-white rounded-lg shadow-md p-4 hover:shadow-lg transition-shadow">
      {/* 搭配标题和操作按钮 */}
      <div className="flex justify-between items-start mb-3">
        <div>
          <h3 className="font-semibold text-gray-800">{outfit.title}</h3>
          <p className="text-sm text-gray-500">{outfit.occasion}</p>
        </div>
        <div className="flex space-x-2">
          <button
            onClick={() => toggleFavorite(outfit.id)}
            className={`p-2 rounded-full transition-colors ${
              outfit.is_favorite 
                ? 'bg-red-100 text-red-600 hover:bg-red-200' 
                : 'bg-gray-100 text-gray-400 hover:bg-gray-200'
            }`}
          >
            <Heart className={`w-4 h-4 ${outfit.is_favorite ? 'fill-current' : ''}`} />
          </button>
          {!outfit.is_worn && (
            <button
              onClick={() => markAsWorn(outfit.id)}
              className="p-2 rounded-full bg-blue-100 text-blue-600 hover:bg-blue-200 transition-colors"
              title="标记为已穿着"
            >
              <Clock className="w-4 h-4" />
            </button>
          )}
        </div>
      </div>

      {/* 搭配物品 */}
      <div className="grid grid-cols-3 gap-2 mb-3">
        {outfit.items.slice(0, 3).map((item, index) => (
          <div key={index} className="aspect-square bg-gray-100 rounded-lg overflow-hidden">
            <img 
              src={item.image_url || '/api/placeholder/100/100'} 
              alt={item.name}
              className="w-full h-full object-cover"
            />
          </div>
        ))}
      </div>

      {/* 风格标签 */}
      <div className="flex flex-wrap gap-1 mb-3">
        {outfit.style_tags.map((tag, index) => (
          <span 
            key={index}
            className="px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded-full"
          >
            {tag}
          </span>
        ))}
      </div>

      {/* 搭配信息 */}
      <div className="flex justify-between items-center text-sm text-gray-500">
        <div className="flex items-center space-x-4">
          <span className="flex items-center">
            <Star className="w-3 h-3 mr-1" />
            {(outfit.similarity_score * 100).toFixed(0)}%
          </span>
          {outfit.is_worn && (
            <span className="flex items-center text-green-600">
              <Clock className="w-3 h-3 mr-1" />
              已穿 {outfit.wear_count} 次
            </span>
          )}
        </div>
        <span>{new Date(outfit.created_at).toLocaleDateString()}</span>
      </div>
    </div>
  );

  // 渲染统计卡片
  const renderStatisticsCard = () => {
    if (!statistics) return null;

    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <div className="bg-white rounded-lg shadow-md p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">总搭配数</p>
              <p className="text-2xl font-bold text-gray-900">{statistics.total_outfits}</p>
            </div>
            <Archive className="w-8 h-8 text-blue-600" />
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-md p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">收藏搭配</p>
              <p className="text-2xl font-bold text-gray-900">{statistics.favorite_count}</p>
            </div>
            <Heart className="w-8 h-8 text-red-600" />
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-md p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">已穿搭配</p>
              <p className="text-2xl font-bold text-gray-900">{statistics.worn_count}</p>
            </div>
            <Clock className="w-8 h-8 text-green-600" />
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-md p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">穿着率</p>
              <p className="text-2xl font-bold text-gray-900">{statistics.wear_rate}%</p>
            </div>
            <TrendingUp className="w-8 h-8 text-purple-600" />
          </div>
        </div>
      </div>
    );
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">加载搭配历史中...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* 页面标题 */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">搭配历史</h1>
          <p className="text-gray-600">管理您的搭配记录、收藏和穿着历史</p>
        </div>

        {/* 统计信息 */}
        {activeTab === 'stats' && renderStatisticsCard()}

        {/* 标签页导航 */}
        <div className="bg-white rounded-lg shadow-sm mb-6">
          <div className="border-b border-gray-200">
            <nav className="flex space-x-8 px-6">
              {[
                { key: 'all', label: '全部搭配', count: outfits.length },
                { key: 'favorites', label: '收藏搭配', count: favorites.length },
                { key: 'worn', label: '已穿搭配', count: wornOutfits.length },
                { key: 'stats', label: '统计信息', count: null }
              ].map(tab => (
                <button
                  key={tab.key}
                  onClick={() => setActiveTab(tab.key as any)}
                  className={`py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
                    activeTab === tab.key
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  {tab.label}
                  {tab.count !== null && (
                    <span className="ml-2 bg-gray-100 text-gray-900 py-0.5 px-2.5 rounded-full text-xs">
                      {tab.count}
                    </span>
                  )}
                </button>
              ))}
            </nav>
          </div>

          {/* 过滤器 */}
          {(activeTab === 'all' || activeTab === 'favorites' || activeTab === 'worn') && (
            <div className="px-6 py-4 border-b border-gray-200">
              <div className="flex items-center space-x-4">
                <Filter className="w-5 h-5 text-gray-400" />
                <select
                  value={filterOccasion}
                  onChange={(e) => setFilterOccasion(e.target.value)}
                  className="border border-gray-300 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="all">全部场合</option>
                  <option value="casual">休闲</option>
                  <option value="formal">正式</option>
                  <option value="business">商务</option>
                  <option value="party">聚会</option>
                  <option value="sport">运动</option>
                </select>
              </div>
            </div>
          )}
        </div>

        {/* 内容区域 */}
        <div className="space-y-6">
          {activeTab === 'stats' && statistics && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              {/* 场合分布 */}
              <div className="bg-white rounded-lg shadow-md p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">场合分布</h3>
                <div className="space-y-3">
                  {Object.entries(statistics.occasion_distribution).map(([occasion, count]) => (
                    <div key={occasion} className="flex justify-between items-center">
                      <span className="text-gray-600">{occasion}</span>
                      <div className="flex items-center space-x-2">
                        <div className="w-20 bg-gray-200 rounded-full h-2">
                          <div 
                            className="bg-blue-600 h-2 rounded-full" 
                            style={{ width: `${(count / statistics.total_outfits) * 100}%` }}
                          ></div>
                        </div>
                        <span className="text-sm font-medium text-gray-900">{count}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* 风格分布 */}
              <div className="bg-white rounded-lg shadow-md p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">风格分布</h3>
                <div className="space-y-3">
                  {Object.entries(statistics.style_distribution).slice(0, 5).map(([style, count]) => (
                    <div key={style} className="flex justify-between items-center">
                      <span className="text-gray-600">{style}</span>
                      <div className="flex items-center space-x-2">
                        <div className="w-20 bg-gray-200 rounded-full h-2">
                          <div 
                            className="bg-purple-600 h-2 rounded-full" 
                            style={{ width: `${(count / statistics.total_outfits) * 100}%` }}
                          ></div>
                        </div>
                        <span className="text-sm font-medium text-gray-900">{count}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* 搭配列表 */}
          {activeTab !== 'stats' && (
            <>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {activeTab === 'all' && filteredOutfits.map(renderOutfitCard)}
                {activeTab === 'favorites' && favorites.filter(outfit => 
                  filterOccasion === 'all' || outfit.occasion === filterOccasion
                ).map(renderOutfitCard)}
                {activeTab === 'worn' && wornOutfits.filter(outfit => 
                  filterOccasion === 'all' || outfit.occasion === filterOccasion
                ).map(renderOutfitCard)}
              </div>

              {/* 加载更多按钮 */}
              {activeTab === 'all' && hasMore && (
                <div className="text-center">
                  <button
                    onClick={loadMore}
                    className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                  >
                    加载更多
                  </button>
                </div>
              )}

              {/* 空状态 */}
              {(
                (activeTab === 'all' && filteredOutfits.length === 0) ||
                (activeTab === 'favorites' && favorites.length === 0) ||
                (activeTab === 'worn' && wornOutfits.length === 0)
              ) && (
                <div className="text-center py-12">
                  <Calendar className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                  <h3 className="text-lg font-medium text-gray-900 mb-2">
                    {activeTab === 'favorites' ? '暂无收藏搭配' : 
                     activeTab === 'worn' ? '暂无穿着记录' : '暂无搭配历史'}
                  </h3>
                  <p className="text-gray-500">
                    {activeTab === 'favorites' ? '收藏您喜欢的搭配，方便随时查看' : 
                     activeTab === 'worn' ? '标记穿着过的搭配，记录您的时尚历程' : '开始创建您的第一个搭配吧'}
                  </p>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default History;