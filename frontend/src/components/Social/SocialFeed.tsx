import React, { useState, useEffect } from 'react';
import { Heart, Star, Eye, MessageCircle, Share2, MoreHorizontal, Filter, TrendingUp } from 'lucide-react';
import { getPopularSharedOutfits, likeSharedOutfit, rateSharedOutfit, getSocialStats } from '../../api/social';
import type { SharedOutfit, SocialStats } from '../../api/social';

interface SocialFeedProps {
  className?: string;
}

const SocialFeed: React.FC<SocialFeedProps> = ({ className = '' }) => {
  const [outfits, setOutfits] = useState<SharedOutfit[]>([]);
  const [stats, setStats] = useState<SocialStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<'popular' | 'recent' | 'trending'>('popular');
  const [likedOutfits, setLikedOutfits] = useState<Set<string>>(new Set());
  const [ratingOutfit, setRatingOutfit] = useState<string | null>(null);

  useEffect(() => {
    loadFeed();
    loadStats();
  }, [filter]);

  const loadFeed = async () => {
    setLoading(true);
    try {
      const response = await getPopularSharedOutfits(20, 0);
      if (response.success && response.data) {
        setOutfits(response.data.outfits);
      }
    } catch (error) {
      console.error('Failed to load social feed:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadStats = async () => {
    try {
      const response = await getSocialStats();
      if (response.success && response.data) {
        setStats(response.data);
      }
    } catch (error) {
      console.error('Failed to load social stats:', error);
    }
  };

  const handleLike = async (shareId: string) => {
    try {
      const response = await likeSharedOutfit(shareId);
      if (response.success && response.data) {
        setLikedOutfits(prev => {
          const newSet = new Set(prev);
          if (response.data?.action === 'liked') {
            newSet.add(shareId);
          } else {
            newSet.delete(shareId);
          }
          return newSet;
        });
        
        // Update outfit likes count
        setOutfits(prev => prev.map(outfit => 
          outfit.share_id === shareId 
            ? { ...outfit, like_count: response.data?.like_count || outfit.like_count }
            : outfit
        ));
      }
    } catch (error) {
      console.error('Failed to like outfit:', error);
    }
  };

  const handleRate = async (shareId: string, rating: number) => {
    setRatingOutfit(shareId);
    try {
      const response = await rateSharedOutfit(shareId, rating);
      if (response.success && response.data) {
        // Update outfit rating
        setOutfits(prev => prev.map(outfit => 
          outfit.share_id === shareId 
            ? { 
                ...outfit, 
                average_rating: response.data?.average_rating || outfit.average_rating,
                rating_count: response.data?.rating_count || outfit.rating_count
              }
            : outfit
        ));
      }
    } catch (error) {
      console.error('Failed to rate outfit:', error);
    } finally {
      setRatingOutfit(null);
    }
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

  const renderStarRating = (rating: number, interactive: boolean = false, onRate?: (rating: number) => void) => {
    return (
      <div className="flex items-center gap-1">
        {[1, 2, 3, 4, 5].map((star) => (
          <button
            key={star}
            onClick={() => interactive && onRate?.(star)}
            disabled={!interactive}
            className={`${
              interactive ? 'hover:scale-110 cursor-pointer' : 'cursor-default'
            } transition-transform`}
          >
            <Star
              className={`w-4 h-4 ${
                star <= rating
                  ? 'fill-yellow-400 text-yellow-400'
                  : 'text-gray-300'
              }`}
            />
          </button>
        ))}
      </div>
    );
  };

  if (loading) {
    return (
      <div className={`space-y-6 ${className}`}>
        {[...Array(3)].map((_, i) => (
          <div key={i} className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 animate-pulse">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 bg-gray-200 rounded-full" />
              <div className="flex-1">
                <div className="h-4 bg-gray-200 rounded w-24 mb-2" />
                <div className="h-3 bg-gray-200 rounded w-16" />
              </div>
            </div>
            <div className="h-64 bg-gray-200 rounded-lg mb-4" />
            <div className="h-4 bg-gray-200 rounded w-3/4 mb-2" />
            <div className="h-3 bg-gray-200 rounded w-1/2" />
          </div>
        ))}
      </div>
    );
  }

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Header with Stats */}
      {stats && (
        <div className="bg-gradient-to-r from-blue-50 to-purple-50 rounded-xl p-6 border border-gray-200">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold text-gray-900 flex items-center gap-2">
              <TrendingUp className="w-5 h-5 text-blue-600" />
              社区动态
            </h2>
            <div className="flex gap-2">
              {[
                { key: 'popular', label: '热门', icon: TrendingUp },
                { key: 'recent', label: '最新', icon: MessageCircle },
                { key: 'trending', label: '趋势', icon: Star }
              ].map(({ key, label, icon: Icon }) => (
                <button
                  key={key}
                  onClick={() => setFilter(key as any)}
                  className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors flex items-center gap-1 ${
                    filter === key
                      ? 'bg-blue-600 text-white'
                      : 'bg-white text-gray-600 hover:bg-gray-50'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  {label}
                </button>
              ))}
            </div>
          </div>
          
          <div className="grid grid-cols-4 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">{stats.shares_count}</div>
              <div className="text-sm text-gray-600">分享搭配</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-purple-600">{stats.total_likes}</div>
              <div className="text-sm text-gray-600">总点赞</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">{stats.total_views}</div>
              <div className="text-sm text-gray-600">总浏览</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-orange-600">{stats.average_rating.toFixed(1)}</div>
              <div className="text-sm text-gray-600">平均评分</div>
            </div>
          </div>
        </div>
      )}

      {/* Feed */}
      <div className="space-y-6">
        {outfits.map((outfit) => (
          <div key={outfit.share_id} className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden hover:shadow-md transition-shadow">
            {/* User Header */}
            <div className="p-4 flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center text-white font-semibold">
                  {outfit.user_id.charAt(0).toUpperCase()}
                </div>
                <div>
                  <div className="font-medium text-gray-900">用户 {outfit.user_id.slice(-4)}</div>
                  <div className="text-sm text-gray-500">{formatTimeAgo(outfit.created_at)}</div>
                </div>
              </div>
              <button className="p-2 hover:bg-gray-100 rounded-lg transition-colors">
                <MoreHorizontal className="w-5 h-5 text-gray-400" />
              </button>
            </div>

            {/* Outfit Image */}
            <div className="relative">
              <img
                src={outfit.collage_url || '/api/placeholder/400/500'}
                alt={outfit.title}
                className="w-full h-96 object-cover"
              />
              {outfit.privacy_level !== 'public' && (
                <div className="absolute top-3 right-3 px-2 py-1 bg-black bg-opacity-50 text-white text-xs rounded-full">
                  {outfit.privacy_level === 'friends' ? '好友可见' : '私密'}
                </div>
              )}
            </div>

            {/* Content */}
            <div className="p-4">
              <h3 className="font-semibold text-gray-900 mb-2">{outfit.title}</h3>
              {outfit.description && (
                <p className="text-gray-600 mb-3">{outfit.description}</p>
              )}
              
              {/* Tags */}
              {outfit.share_tags && outfit.share_tags.length > 0 && (
                <div className="flex flex-wrap gap-2 mb-4">
                  {outfit.share_tags.map((tag: string, index: number) => (
                    <span
                      key={index}
                      className="px-2 py-1 bg-gray-100 text-gray-600 text-xs rounded-full"
                    >
                      #{tag}
                    </span>
                  ))}
                </div>
              )}

              {/* Stats */}
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-4 text-sm text-gray-500">
                  <span className="flex items-center gap-1">
                    <Eye className="w-4 h-4" />
                    {outfit.view_count}
                  </span>
                  <span className="flex items-center gap-1">
                    <Heart className="w-4 h-4" />
                    {outfit.like_count}
                  </span>
                  <span className="flex items-center gap-1">
                    <Star className="w-4 h-4" />
                    {outfit.average_rating ? outfit.average_rating.toFixed(1) : '暂无'}
                    {outfit.rating_count > 0 && (
                      <span className="text-xs">({outfit.rating_count})</span>
                    )}
                  </span>
                </div>
              </div>

              {/* Actions */}
              <div className="flex items-center justify-between pt-3 border-t border-gray-100">
                <div className="flex items-center gap-4">
                  <button
                    onClick={() => handleLike(outfit.share_id)}
                    className={`flex items-center gap-2 px-3 py-2 rounded-lg transition-colors ${
                      likedOutfits.has(outfit.share_id)
                        ? 'bg-red-50 text-red-600'
                        : 'hover:bg-gray-50 text-gray-600'
                    }`}
                  >
                    <Heart className={`w-4 h-4 ${
                      likedOutfits.has(outfit.share_id) ? 'fill-current' : ''
                    }`} />
                    <span className="text-sm font-medium">点赞</span>
                  </button>
                  
                  <div className="flex items-center gap-2">
                    <span className="text-sm text-gray-600">评分:</span>
                    {renderStarRating(
                      0, // Current user rating (would need to track this)
                      ratingOutfit !== outfit.share_id,
                      (rating) => handleRate(outfit.share_id, rating)
                    )}
                    {ratingOutfit === outfit.share_id && (
                      <div className="w-4 h-4 border-2 border-blue-600 border-t-transparent rounded-full animate-spin" />
                    )}
                  </div>
                </div>
                
                <button className="flex items-center gap-2 px-3 py-2 hover:bg-gray-50 rounded-lg transition-colors text-gray-600">
                  <Share2 className="w-4 h-4" />
                  <span className="text-sm font-medium">分享</span>
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Load More */}
      {outfits.length > 0 && (
        <div className="text-center">
          <button
            onClick={loadFeed}
            className="px-6 py-3 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-lg transition-colors font-medium"
          >
            加载更多
          </button>
        </div>
      )}

      {/* Empty State */}
      {!loading && outfits.length === 0 && (
        <div className="text-center py-12">
          <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
            <Share2 className="w-8 h-8 text-gray-400" />
          </div>
          <h3 className="text-lg font-medium text-gray-900 mb-2">暂无分享内容</h3>
          <p className="text-gray-500">成为第一个分享搭配的用户吧！</p>
        </div>
      )}
    </div>
  );
};

export default SocialFeed;