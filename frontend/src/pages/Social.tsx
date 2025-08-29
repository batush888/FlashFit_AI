import React, { useState, useEffect } from 'react';
import { Share2, Users, TrendingUp, Heart, Star, Plus, Filter } from 'lucide-react';
import SocialFeed from '../components/Social/SocialFeed';
import ShareOutfitModal from '../components/Social/ShareOutfitModal';
import { getUserSharedOutfits, getSocialStats } from '../api/social';
import type { SharedOutfit, SocialStats } from '../api/social';

const Social: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'feed' | 'my-shares'>('feed');
  const [showShareModal, setShowShareModal] = useState(false);
  const [selectedOutfit, setSelectedOutfit] = useState<any>(null);
  const [myShares, setMyShares] = useState<SharedOutfit[]>([]);
  const [stats, setStats] = useState<SocialStats | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    loadUserStats();
    if (activeTab === 'my-shares') {
      loadMyShares();
    }
  }, [activeTab]);

  const loadUserStats = async () => {
    try {
      const response = await getSocialStats();
      if (response.success && response.data) {
        setStats(response.data);
      }
    } catch (error) {
      console.error('Failed to load user stats:', error);
    }
  };

  const loadMyShares = async () => {
    setLoading(true);
    try {
      const response = await getUserSharedOutfits(20, 0);
      if (response.success && response.data) {
        setMyShares(response.data.shares);
      }
    } catch (error) {
      console.error('Failed to load user shares:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleShareOutfit = (outfitData: any) => {
    setSelectedOutfit(outfitData);
    setShowShareModal(true);
  };

  const handleShareSuccess = (shareData: any) => {
    // Refresh my shares if on that tab
    if (activeTab === 'my-shares') {
      loadMyShares();
    }
    // Refresh stats
    loadUserStats();
    setShowShareModal(false);
  };

  const formatTimeAgo = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffInSeconds = Math.floor((now.getTime() - date.getTime()) / 1000);
    
    if (diffInSeconds < 60) return '刚刚';
    if (diffInSeconds < 3600) return `${Math.floor(diffInSeconds / 60)}分钟前`;
    if (diffInSeconds < 86400) return `${Math.floor(diffInSeconds / 3600)}小时前`;
    if (diffInSeconds < 2592000) return `${Math.floor(diffInSeconds / 86400)}天前`;
    return date.toLocaleDateString('zh-CN');
  };

  // Mock outfit data for demo purposes
  const mockOutfitData = {
    title: '春日休闲搭配',
    occasion: '日常',
    items: [
      { name: '白色T恤', category: '上装' },
      { name: '牛仔裤', category: '下装' },
      { name: '小白鞋', category: '鞋子' }
    ],
    collage_url: '/api/placeholder/400/500',
    style_tags: ['休闲', '简约', '舒适'],
    tips: ['适合春秋季节', '百搭不出错']
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-6xl mx-auto px-4 py-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-gray-900 flex items-center gap-3">
                <Users className="w-7 h-7 text-blue-600" />
                社区广场
              </h1>
              <p className="text-gray-600 mt-1">发现和分享时尚搭配灵感</p>
            </div>
            
            <button
              onClick={() => handleShareOutfit(mockOutfitData)}
              className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors font-medium"
            >
              <Plus className="w-4 h-4" />
              分享搭配
            </button>
          </div>

          {/* User Stats */}
          {stats && (
            <div className="mt-6 grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="bg-blue-50 rounded-lg p-4 text-center">
                <div className="text-2xl font-bold text-blue-600">{stats.shares_count}</div>
                <div className="text-sm text-blue-700">我的分享</div>
              </div>
              <div className="bg-purple-50 rounded-lg p-4 text-center">
                <div className="text-2xl font-bold text-purple-600">{stats.liked_shares_count}</div>
                <div className="text-sm text-purple-700">获得点赞</div>
              </div>
              <div className="bg-green-50 rounded-lg p-4 text-center">
                <div className="text-2xl font-bold text-green-600">{stats.followers_count}</div>
                <div className="text-sm text-green-700">粉丝数</div>
              </div>
              <div className="bg-orange-50 rounded-lg p-4 text-center">
                <div className="text-2xl font-bold text-orange-600">{(stats.engagement_rate * 100).toFixed(1)}%</div>
                <div className="text-sm text-orange-700">互动率</div>
              </div>
            </div>
          )}

          {/* Navigation Tabs */}
          <div className="mt-6 flex space-x-1 bg-gray-100 rounded-lg p-1">
            <button
              onClick={() => setActiveTab('feed')}
              className={`flex-1 flex items-center justify-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                activeTab === 'feed'
                  ? 'bg-white text-blue-600 shadow-sm'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              <TrendingUp className="w-4 h-4" />
              社区动态
            </button>
            <button
              onClick={() => setActiveTab('my-shares')}
              className={`flex-1 flex items-center justify-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                activeTab === 'my-shares'
                  ? 'bg-white text-blue-600 shadow-sm'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              <Share2 className="w-4 h-4" />
              我的分享
            </button>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="max-w-6xl mx-auto px-4 py-6">
        {activeTab === 'feed' ? (
          <SocialFeed className="max-w-2xl mx-auto" />
        ) : (
          /* My Shares Tab */
          <div className="max-w-2xl mx-auto">
            {loading ? (
              <div className="space-y-6">
                {[...Array(3)].map((_, i) => (
                  <div key={i} className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 animate-pulse">
                    <div className="h-64 bg-gray-200 rounded-lg mb-4" />
                    <div className="h-4 bg-gray-200 rounded w-3/4 mb-2" />
                    <div className="h-3 bg-gray-200 rounded w-1/2" />
                  </div>
                ))}
              </div>
            ) : myShares.length > 0 ? (
              <div className="space-y-6">
                {myShares.map((share) => (
                  <div key={share.share_id} className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
                    {/* Share Header */}
                    <div className="p-4 border-b border-gray-100">
                      <div className="flex items-center justify-between">
                        <div>
                          <h3 className="font-semibold text-gray-900">{share.title}</h3>
                          <p className="text-sm text-gray-500">{formatTimeAgo(share.created_at)}</p>
                        </div>
                        <div className="flex items-center gap-4 text-sm text-gray-500">
                          <span className="flex items-center gap-1">
                            <Heart className="w-4 h-4" />
                            {share.like_count}
                          </span>
                          <span className="flex items-center gap-1">
                            <Star className="w-4 h-4" />
                            {share.average_rating ? share.average_rating.toFixed(1) : '暂无'}
                          </span>
                        </div>
                      </div>
                    </div>

                    {/* Share Content */}
                    <div className="p-4">
                      <div className="flex gap-4">
                        <img
                          src={share.collage_url || '/api/placeholder/200/250'}
                          alt={share.title}
                          className="w-32 h-40 object-cover rounded-lg"
                        />
                        <div className="flex-1">
                          <p className="text-gray-600 mb-3">{share.description}</p>
                          
                          {/* Tags */}
                          {share.share_tags && share.share_tags.length > 0 && (
                            <div className="flex flex-wrap gap-2 mb-3">
                              {share.share_tags.map((tag: string, index: number) => (
                                <span
                                  key={index}
                                  className="px-2 py-1 bg-gray-100 text-gray-600 text-xs rounded-full"
                                >
                                  #{tag}
                                </span>
                              ))}
                            </div>
                          )}

                          {/* Privacy Level */}
                          <div className="flex items-center gap-2">
                            <span className="text-xs text-gray-500">可见性:</span>
                            <span className={`px-2 py-1 text-xs rounded-full ${
                              share.privacy_level === 'public'
                                ? 'bg-green-100 text-green-700'
                                : share.privacy_level === 'friends'
                                ? 'bg-blue-100 text-blue-700'
                                : 'bg-gray-100 text-gray-700'
                            }`}>
                              {share.privacy_level === 'public' ? '公开' : 
                               share.privacy_level === 'friends' ? '好友' : '私密'}
                            </span>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Share Stats */}
                    <div className="px-4 py-3 bg-gray-50 border-t border-gray-100">
                      <div className="flex items-center justify-between text-sm text-gray-600">
                        <div className="flex items-center gap-4">
                          <span>{share.view_count} 次浏览</span>
                          <span>{share.like_count} 个点赞</span>
                          <span>{share.comment_count} 条评论</span>
                        </div>
                        <div className="flex items-center gap-2">
                          {share.is_featured && (
                            <span className="px-2 py-1 bg-yellow-100 text-yellow-700 text-xs rounded-full">
                              精选
                            </span>
                          )}
                          <span className={`px-2 py-1 text-xs rounded-full ${
                            share.is_active
                              ? 'bg-green-100 text-green-700'
                              : 'bg-red-100 text-red-700'
                          }`}>
                            {share.is_active ? '活跃' : '已隐藏'}
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              /* Empty State */
              <div className="text-center py-12">
                <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
                  <Share2 className="w-8 h-8 text-gray-400" />
                </div>
                <h3 className="text-lg font-medium text-gray-900 mb-2">还没有分享内容</h3>
                <p className="text-gray-500 mb-4">分享你的第一个搭配，让更多人看到你的时尚品味！</p>
                <button
                  onClick={() => handleShareOutfit(mockOutfitData)}
                  className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors font-medium"
                >
                  立即分享
                </button>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Share Modal */}
      {showShareModal && selectedOutfit && (
        <ShareOutfitModal
          isOpen={showShareModal}
          onClose={() => setShowShareModal(false)}
          outfitData={selectedOutfit}
          onShareSuccess={handleShareSuccess}
        />
      )}
    </div>
  );
};

export default Social;