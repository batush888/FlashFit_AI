import React, { useState } from 'react';
import { X, Share2, Facebook, Twitter, Instagram, Link, Copy, Star, Heart, Eye, 小红书, TikTok} from 'lucide-react';
import { shareOutfit, generateSocialShareUrl, copyShareLink, validateShareOptions } from '../../api/social';
import type { ShareOptions, SharedOutfit } from '../../api/social';

interface ShareOutfitModalProps {
  isOpen: boolean;
  onClose: () => void;
  outfitData: any;
  onShareSuccess?: (shareData: any) => void;
}

const ShareOutfitModal: React.FC<ShareOutfitModalProps> = ({
  isOpen,
  onClose,
  outfitData,
  onShareSuccess
}) => {
  const [shareOptions, setShareOptions] = useState<ShareOptions>({
    description: '',
    privacy_level: 'public',
    allow_comments: true,
    tags: []
  });
  const [isSharing, setIsSharing] = useState(false);
  const [shareResult, setShareResult] = useState<any>(null);
  const [errors, setErrors] = useState<string[]>([]);
  const [newTag, setNewTag] = useState('');
  const [copySuccess, setCopySuccess] = useState(false);

  if (!isOpen) return null;

  const handleShare = async () => {
    setIsSharing(true);
    setErrors([]);

    // 验证分享选项
    const validation = validateShareOptions(shareOptions);
    if (!validation.isValid) {
      setErrors(validation.errors);
      setIsSharing(false);
      return;
    }

    try {
      const response = await shareOutfit(outfitData, shareOptions);
      if (response.success) {
        setShareResult(response.data);
        onShareSuccess?.(response.data);
      } else {
        setErrors([response.message || '分享失败']);
      }
    } catch (error) {
      setErrors(['分享时发生错误，请稍后重试']);
    } finally {
      setIsSharing(false);
    }
  };

  const handleAddTag = () => {
    if (newTag.trim() && !shareOptions.tags.includes(newTag.trim())) {
      setShareOptions(prev => ({
        ...prev,
        tags: [...prev.tags, newTag.trim()]
      }));
      setNewTag('');
    }
  };

  const handleRemoveTag = (tagToRemove: string) => {
    setShareOptions(prev => ({
      ...prev,
      tags: prev.tags.filter(tag => tag !== tagToRemove)
    }));
  };

  const handleCopyLink = async (shareUrl: string) => {
    const success = await copyShareLink(shareUrl);
    if (success) {
      setCopySuccess(true);
      setTimeout(() => setCopySuccess(false), 2000);
    }
  };

  const handleSocialShare = (platform: string, shareUrl: string, title: string) => {
    const url = generateSocialShareUrl(platform, shareUrl, title);
    window.open(url, '_blank', 'width=600,height=400');
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-xl max-w-2xl w-full max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b">
          <h2 className="text-xl font-semibold text-gray-900 flex items-center gap-2">
            <Share2 className="w-5 h-5" />
            分享搭配
          </h2>
          <button
            onClick={onClose}
            className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {!shareResult ? (
          /* Share Form */
          <div className="p-6 space-y-6">
            {/* Outfit Preview */}
            <div className="bg-gray-50 rounded-lg p-4">
              <h3 className="font-medium text-gray-900 mb-2">搭配预览</h3>
              <div className="flex items-center gap-4">
                {outfitData.collage_url && (
                  <img
                    src={outfitData.collage_url}
                    alt="搭配预览"
                    className="w-20 h-20 object-cover rounded-lg"
                  />
                )}
                <div>
                  <p className="font-medium">{outfitData.title || '我的搭配'}</p>
                  <p className="text-sm text-gray-600">{outfitData.occasion || '日常'}</p>
                  <p className="text-xs text-gray-500">
                    {outfitData.items?.length || 0} 件单品
                  </p>
                </div>
              </div>
            </div>

            {/* Description */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                分享描述
              </label>
              <textarea
                value={shareOptions.description}
                onChange={(e) => setShareOptions(prev => ({ ...prev, description: e.target.value }))}
                placeholder="分享一些关于这个搭配的想法..."
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
                rows={3}
                maxLength={500}
              />
              <p className="text-xs text-gray-500 mt-1">
                {(shareOptions.description || '').length}/500
              </p>
            </div>

            {/* Privacy Level */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                隐私设置
              </label>
              <div className="grid grid-cols-3 gap-3">
                {[
                  { value: 'public', label: '公开', desc: '所有人可见' },
                  { value: 'friends', label: '好友', desc: '仅好友可见' },
                  { value: 'private', label: '私密', desc: '仅自己可见' }
                ].map((option) => (
                  <label key={option.value} className="cursor-pointer">
                    <input
                      type="radio"
                      name="privacy"
                      value={option.value}
                      checked={shareOptions.privacy_level === option.value}
                      onChange={(e) => setShareOptions(prev => ({ 
                        ...prev, 
                        privacy_level: e.target.value as 'public' | 'friends' | 'private'
                      }))}
                      className="sr-only"
                    />
                    <div className={`p-3 border rounded-lg text-center transition-colors ${
                      shareOptions.privacy_level === option.value
                        ? 'border-blue-500 bg-blue-50 text-blue-700'
                        : 'border-gray-300 hover:border-gray-400'
                    }`}>
                      <div className="font-medium text-sm">{option.label}</div>
                      <div className="text-xs text-gray-500">{option.desc}</div>
                    </div>
                  </label>
                ))}
              </div>
            </div>

            {/* Comments Setting */}
            <div className="flex items-center justify-between">
              <div>
                <label className="text-sm font-medium text-gray-700">允许评论</label>
                <p className="text-xs text-gray-500">其他用户可以对你的分享进行评论</p>
              </div>
              <label className="relative inline-flex items-center cursor-pointer">
                <input
                  type="checkbox"
                  checked={shareOptions.allow_comments}
                  onChange={(e) => setShareOptions(prev => ({ 
                    ...prev, 
                    allow_comments: e.target.checked 
                  }))}
                  className="sr-only"
                />
                <div className={`w-11 h-6 rounded-full transition-colors ${
                  shareOptions.allow_comments ? 'bg-blue-600' : 'bg-gray-300'
                }`}>
                  <div className={`w-5 h-5 bg-white rounded-full shadow transform transition-transform ${
                    shareOptions.allow_comments ? 'translate-x-5' : 'translate-x-0'
                  } mt-0.5 ml-0.5`} />
                </div>
              </label>
            </div>

            {/* Tags */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                标签
              </label>
              <div className="flex flex-wrap gap-2 mb-3">
                {shareOptions.tags.map((tag, index) => (
                  <span
                    key={index}
                    className="inline-flex items-center gap-1 px-3 py-1 bg-blue-100 text-blue-800 text-sm rounded-full"
                  >
                    {tag}
                    <button
                      onClick={() => handleRemoveTag(tag)}
                      className="hover:bg-blue-200 rounded-full p-0.5"
                    >
                      <X className="w-3 h-3" />
                    </button>
                  </span>
                ))}
              </div>
              <div className="flex gap-2">
                <input
                  type="text"
                  value={newTag}
                  onChange={(e) => setNewTag(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && handleAddTag()}
                  placeholder="添加标签..."
                  className="flex-1 px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm"
                />
                <button
                  onClick={handleAddTag}
                  disabled={!newTag.trim()}
                  className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed text-sm"
                >
                  添加
                </button>
              </div>
            </div>

            {/* Errors */}
            {errors.length > 0 && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-3">
                <ul className="text-sm text-red-600 space-y-1">
                  {errors.map((error, index) => (
                    <li key={index}>• {error}</li>
                  ))}
                </ul>
              </div>
            )}

            {/* Actions */}
            <div className="flex gap-3 pt-4">
              <button
                onClick={onClose}
                className="flex-1 px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition-colors"
              >
                取消
              </button>
              <button
                onClick={handleShare}
                disabled={isSharing}
                className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center justify-center gap-2"
              >
                {isSharing ? (
                  <>
                    <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                    分享中...
                  </>
                ) : (
                  <>
                    <Share2 className="w-4 h-4" />
                    分享搭配
                  </>
                )}
              </button>
            </div>
          </div>
        ) : (
          /* Share Success */
          <div className="p-6 space-y-6">
            <div className="text-center">
              <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <Share2 className="w-8 h-8 text-green-600" />
              </div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">分享成功！</h3>
              <p className="text-gray-600">你的搭配已成功分享到社区</p>
            </div>

            {/* Share Link */}
            <div className="bg-gray-50 rounded-lg p-4">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                分享链接
              </label>
              <div className="flex gap-2">
                <input
                  type="text"
                  value={`https://flashfit.ai${shareResult.share_url}`}
                  readOnly
                  className="flex-1 px-3 py-2 bg-white border border-gray-300 rounded-lg text-sm"
                />
                <button
                  onClick={() => handleCopyLink(shareResult.share_url)}
                  className={`px-4 py-2 rounded-lg transition-colors flex items-center gap-2 ${
                    copySuccess
                      ? 'bg-green-600 text-white'
                      : 'bg-gray-600 text-white hover:bg-gray-700'
                  }`}
                >
                  {copySuccess ? (
                    <>
                      <span className="text-sm">已复制</span>
                    </>
                  ) : (
                    <>
                      <Copy className="w-4 h-4" />
                      <span className="text-sm">复制</span>
                    </>
                  )}
                </button>
              </div>
            </div>

            {/* Social Media Sharing */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-3">
                分享到社交媒体
              </label>
              <div className="grid grid-cols-2 gap-3">
                <button
                  onClick={() => handleSocialShare('facebook', shareResult.share_url, shareResult.shared_outfit.title)}
                  className="flex items-center gap-3 p-3 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
                >
                  <Facebook className="w-5 h-5 text-blue-600" />
                  <span className="text-sm font-medium">Facebook</span>
                </button>
                <button
                  onClick={() => handleSocialShare('twitter', shareResult.share_url, shareResult.shared_outfit.title)}
                  className="flex items-center gap-3 p-3 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
                >
                  <Twitter className="w-5 h-5 text-blue-400" />
                  <span className="text-sm font-medium">Twitter</span>
                </button>
                <button
                  onClick={() => handleSocialShare('pinterest', shareResult.share_url, shareResult.shared_outfit.title)}
                  className="flex items-center gap-3 p-3 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
                >
                  <div className="w-5 h-5 bg-red-600 rounded" />
                  <span className="text-sm font-medium">Pinterest</span>
                </button>
                <button
                  onClick={() => handleSocialShare('whatsapp', shareResult.share_url, shareResult.shared_outfit.title)}
                  className="flex items-center gap-3 p-3 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
                >
                  <div className="w-5 h-5 bg-green-500 rounded" />
                  <span className="text-sm font-medium">WhatsApp</span>
                </button>
                <button
                  onClick={() => handleSocialShare('xiaohongshu', shareResult.share_url, shareResult.shared_outfit.title)}
                  className="flex items-center gap-3 p-3 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
                >
                  <div className="w-5 h-5 text-red-600" />
                  <span className="text-sm font-medium">小红书</span>
                </button>
                <button
                  onClick={() => handleSocialShare('tiktok', shareResult.share_url, shareResult.shared_outfit.title)}
                  className="flex items-center gap-3 p-3 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
                >
                  <div className="w-5 h-5 text-black-600" />
                  <span className="text-sm font-medium">TikTok</span>
                </button>
              </div>
            </div>

            {/* Share Stats Preview */}
            <div className="bg-blue-50 rounded-lg p-4">
              <h4 className="font-medium text-blue-900 mb-2">分享统计</h4>
              <div className="grid grid-cols-3 gap-4 text-center">
                <div>
                  <div className="flex items-center justify-center gap-1 text-blue-600 mb-1">
                    <Eye className="w-4 h-4" />
                    <span className="text-sm font-medium">浏览</span>
                  </div>
                  <div className="text-lg font-semibold text-blue-900">0</div>
                </div>
                <div>
                  <div className="flex items-center justify-center gap-1 text-blue-600 mb-1">
                    <Heart className="w-4 h-4" />
                    <span className="text-sm font-medium">点赞</span>
                  </div>
                  <div className="text-lg font-semibold text-blue-900">0</div>
                </div>
                <div>
                  <div className="flex items-center justify-center gap-1 text-blue-600 mb-1">
                    <Star className="w-4 h-4" />
                    <span className="text-sm font-medium">评分</span>
                  </div>
                  <div className="text-lg font-semibold text-blue-900">-</div>
                </div>
              </div>
            </div>

            {/* Actions */}
            <div className="flex gap-3">
              <button
                onClick={onClose}
                className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
              >
                完成
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ShareOutfitModal;