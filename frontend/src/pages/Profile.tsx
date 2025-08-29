import React, { useState, useEffect } from 'react';
import { useAuthStore } from '../stores/authStore';

interface UserProfile {
  id: string;
  email: string;
  displayName: string;
  avatar?: string;
  bio?: string;
  location?: string;
  joinDate: string;
  preferences: {
    favoriteStyles: string[];
    favoriteColors: string[];
    favoriteBrands: string[];
    sizes: {
      tops: string;
      bottoms: string;
      shoes: string;
      dresses: string;
    };
    occasions: string[];
  };
  privacy: {
    profileVisibility: 'public' | 'friends' | 'private';
    showWardrobe: boolean;
    showOutfits: boolean;
    allowRecommendations: boolean;
  };
  statistics: {
    totalItems: number;
    outfitsCreated: number;
    recommendationsReceived: number;
    favoriteOutfits: number;
  };
}

const Profile = () => {
  const { user } = useAuthStore();
  const [activeTab, setActiveTab] = useState('profile');
  const [isEditing, setIsEditing] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [profile, setProfile] = useState<UserProfile>({
    id: user?.id || '',
    email: user?.email || '',
    displayName: user?.email?.split('@')[0] || '',
    bio: '',
    location: '',
    joinDate: user?.created_at || new Date().toISOString(),
    preferences: {
      favoriteStyles: ['休闲', '商务'],
      favoriteColors: ['黑色', '白色', '蓝色'],
      favoriteBrands: [],
      sizes: {
        tops: 'M',
        bottoms: 'M',
        shoes: '42',
        dresses: 'M'
      },
      occasions: ['工作', '休闲']
    },
    privacy: {
      profileVisibility: 'public',
      showWardrobe: true,
      showOutfits: true,
      allowRecommendations: true
    },
    statistics: {
      totalItems: 0,
      outfitsCreated: 0,
      recommendationsReceived: 0,
      favoriteOutfits: 0
    }
  });

  const tabs = [
    { id: 'profile', name: '个人信息', icon: '👤' },
    { id: 'preferences', name: '偏好设置', icon: '⚙️' },
    { id: 'privacy', name: '隐私设置', icon: '🔒' },
    { id: 'statistics', name: '统计信息', icon: '📊' }
  ];

  const styleOptions = [
    '休闲', '商务', '正式', '运动', '街头', '复古', '简约', '波希米亚', '朋克', '优雅'
  ];

  const colorOptions = [
    '黑色', '白色', '灰色', '蓝色', '红色', '绿色', '黄色', '紫色', '粉色', '棕色', '橙色', '青色'
  ];

  const occasionOptions = [
    '工作', '休闲', '正式场合', '约会', '聚会', '运动', '旅行', '购物', '居家', '户外活动'
  ];

  const handleSave = async () => {
    setIsSaving(true);
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 1000));
    setIsSaving(false);
    setIsEditing(false);
  };

  const handlePreferenceToggle = (category: 'favoriteStyles' | 'favoriteColors' | 'favoriteBrands' | 'occasions', item: string) => {
    setProfile(prev => ({
      ...prev,
      preferences: {
        ...prev.preferences,
        [category]: prev.preferences[category].includes(item)
          ? prev.preferences[category].filter(i => i !== item)
          : [...prev.preferences[category], item]
      }
    }));
  };

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Header Section */}
      <div className="bg-gradient-to-r from-indigo-600 to-purple-600 rounded-2xl shadow-xl p-8 text-white">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between">
          <div className="flex items-center space-x-6">
            {/* Avatar */}
            <div className="relative">
              <div className="w-24 h-24 bg-white/20 rounded-2xl flex items-center justify-center backdrop-blur-sm border border-white/30">
                {profile.avatar ? (
                  <img src={profile.avatar} alt="Avatar" className="w-full h-full rounded-2xl object-cover" />
                ) : (
                  <svg className="w-12 h-12 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                  </svg>
                )}
              </div>
              <button className="absolute -bottom-2 -right-2 w-8 h-8 bg-white text-indigo-600 rounded-full flex items-center justify-center shadow-lg hover:bg-gray-50 transition-colors">
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" />
                </svg>
              </button>
            </div>
            
            {/* User Info */}
            <div>
              <h1 className="text-3xl font-bold mb-1">{profile.displayName}</h1>
              <p className="text-white/90 mb-2">{profile.email}</p>
              <div className="flex items-center text-sm text-white/80">
                <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3a2 2 0 012-2h4a2 2 0 012 2v4m-6 0h6m-6 0l-2 13h10l-2-13" />
                </svg>
                加入于 {new Date(profile.joinDate).toLocaleDateString('zh-CN')}
              </div>
            </div>
          </div>
          
          {/* Action Buttons */}
          <div className="mt-6 md:mt-0 flex space-x-3">
            {isEditing ? (
              <>
                <button
                  onClick={() => setIsEditing(false)}
                  className="px-6 py-3 bg-white/20 text-white rounded-xl hover:bg-white/30 transition-colors font-medium backdrop-blur-sm border border-white/30"
                >
                  取消
                </button>
                <button
                  onClick={handleSave}
                  disabled={isSaving}
                  className="px-6 py-3 bg-white text-indigo-600 rounded-xl hover:bg-gray-50 transition-colors font-medium shadow-lg disabled:opacity-50"
                >
                  {isSaving ? '保存中...' : '保存更改'}
                </button>
              </>
            ) : (
              <button
                onClick={() => setIsEditing(true)}
                className="px-6 py-3 bg-white/20 text-white rounded-xl hover:bg-white/30 transition-colors font-medium backdrop-blur-sm border border-white/30"
              >
                编辑资料
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="bg-white rounded-2xl shadow-lg border border-gray-100">
        <div className="flex overflow-x-auto">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex-1 min-w-0 px-6 py-4 text-sm font-medium transition-colors duration-200 ${
                activeTab === tab.id
                  ? 'text-indigo-600 border-b-2 border-indigo-600 bg-indigo-50'
                  : 'text-gray-500 hover:text-gray-700 hover:bg-gray-50'
              }`}
            >
              <div className="flex items-center justify-center space-x-2">
                <span className="text-lg">{tab.icon}</span>
                <span className="hidden sm:inline">{tab.name}</span>
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Tab Content */}
      <div className="bg-white rounded-2xl shadow-lg border border-gray-100 p-8">
        {activeTab === 'profile' && (
          <div className="space-y-8">
            <div>
              <h2 className="text-2xl font-bold text-gray-900 mb-6">个人信息</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">显示名称</label>
                  <input
                    type="text"
                    value={profile.displayName}
                    onChange={(e) => setProfile(prev => ({ ...prev, displayName: e.target.value }))}
                    disabled={!isEditing}
                    className="w-full px-4 py-3 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent disabled:bg-gray-50 disabled:text-gray-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">邮箱地址</label>
                  <input
                    type="email"
                    value={profile.email}
                    disabled
                    className="w-full px-4 py-3 border border-gray-200 rounded-xl bg-gray-50 text-gray-500"
                  />
                </div>
                <div className="md:col-span-2">
                  <label className="block text-sm font-medium text-gray-700 mb-2">个人简介</label>
                  <textarea
                    value={profile.bio}
                    onChange={(e) => setProfile(prev => ({ ...prev, bio: e.target.value }))}
                    disabled={!isEditing}
                    rows={3}
                    className="w-full px-4 py-3 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent disabled:bg-gray-50 disabled:text-gray-500"
                    placeholder="介绍一下您的风格偏好和时尚理念..."
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">所在地区</label>
                  <input
                    type="text"
                    value={profile.location}
                    onChange={(e) => setProfile(prev => ({ ...prev, location: e.target.value }))}
                    disabled={!isEditing}
                    className="w-full px-4 py-3 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent disabled:bg-gray-50 disabled:text-gray-500"
                    placeholder="例如：北京市朝阳区"
                  />
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'preferences' && (
          <div className="space-y-8">
            <div>
              <h2 className="text-2xl font-bold text-gray-900 mb-6">偏好设置</h2>
              
              {/* Style Preferences */}
              <div className="mb-8">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">喜欢的风格</h3>
                <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
                  {styleOptions.map((style) => (
                    <button
                      key={style}
                      onClick={() => handlePreferenceToggle('favoriteStyles', style)}
                      disabled={!isEditing}
                      className={`px-4 py-2 rounded-xl text-sm font-medium transition-colors ${
                        profile.preferences.favoriteStyles.includes(style)
                          ? 'bg-indigo-100 text-indigo-700 border-2 border-indigo-200'
                          : 'bg-gray-50 text-gray-600 border-2 border-gray-200 hover:bg-gray-100'
                      } disabled:opacity-50 disabled:cursor-not-allowed`}
                    >
                      {style}
                    </button>
                  ))}
                </div>
              </div>

              {/* Color Preferences */}
              <div className="mb-8">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">喜欢的颜色</h3>
                <div className="grid grid-cols-3 md:grid-cols-6 gap-3">
                  {colorOptions.map((color) => (
                    <button
                      key={color}
                      onClick={() => handlePreferenceToggle('favoriteColors', color)}
                      disabled={!isEditing}
                      className={`px-4 py-2 rounded-xl text-sm font-medium transition-colors ${
                        profile.preferences.favoriteColors.includes(color)
                          ? 'bg-indigo-100 text-indigo-700 border-2 border-indigo-200'
                          : 'bg-gray-50 text-gray-600 border-2 border-gray-200 hover:bg-gray-100'
                      } disabled:opacity-50 disabled:cursor-not-allowed`}
                    >
                      {color}
                    </button>
                  ))}
                </div>
              </div>

              {/* Size Information */}
              <div className="mb-8">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">尺码信息</h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">上装</label>
                    <select
                      value={profile.preferences.sizes.tops}
                      onChange={(e) => setProfile(prev => ({
                        ...prev,
                        preferences: {
                          ...prev.preferences,
                          sizes: { ...prev.preferences.sizes, tops: e.target.value }
                        }
                      }))}
                      disabled={!isEditing}
                      className="w-full px-3 py-2 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500 disabled:bg-gray-50"
                    >
                      <option value="XS">XS</option>
                      <option value="S">S</option>
                      <option value="M">M</option>
                      <option value="L">L</option>
                      <option value="XL">XL</option>
                      <option value="XXL">XXL</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">下装</label>
                    <select
                      value={profile.preferences.sizes.bottoms}
                      onChange={(e) => setProfile(prev => ({
                        ...prev,
                        preferences: {
                          ...prev.preferences,
                          sizes: { ...prev.preferences.sizes, bottoms: e.target.value }
                        }
                      }))}
                      disabled={!isEditing}
                      className="w-full px-3 py-2 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500 disabled:bg-gray-50"
                    >
                      <option value="XS">XS</option>
                      <option value="S">S</option>
                      <option value="M">M</option>
                      <option value="L">L</option>
                      <option value="XL">XL</option>
                      <option value="XXL">XXL</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">鞋子</label>
                    <input
                      type="text"
                      value={profile.preferences.sizes.shoes}
                      onChange={(e) => setProfile(prev => ({
                        ...prev,
                        preferences: {
                          ...prev.preferences,
                          sizes: { ...prev.preferences.sizes, shoes: e.target.value }
                        }
                      }))}
                      disabled={!isEditing}
                      className="w-full px-3 py-2 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500 disabled:bg-gray-50"
                      placeholder="例如：42"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">连衣裙</label>
                    <select
                      value={profile.preferences.sizes.dresses}
                      onChange={(e) => setProfile(prev => ({
                        ...prev,
                        preferences: {
                          ...prev.preferences,
                          sizes: { ...prev.preferences.sizes, dresses: e.target.value }
                        }
                      }))}
                      disabled={!isEditing}
                      className="w-full px-3 py-2 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500 disabled:bg-gray-50"
                    >
                      <option value="XS">XS</option>
                      <option value="S">S</option>
                      <option value="M">M</option>
                      <option value="L">L</option>
                      <option value="XL">XL</option>
                      <option value="XXL">XXL</option>
                    </select>
                  </div>
                </div>
              </div>

              {/* Occasion Preferences */}
              <div>
                <h3 className="text-lg font-semibold text-gray-900 mb-4">常用场合</h3>
                <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
                  {occasionOptions.map((occasion) => (
                    <button
                      key={occasion}
                      onClick={() => handlePreferenceToggle('occasions', occasion)}
                      disabled={!isEditing}
                      className={`px-4 py-2 rounded-xl text-sm font-medium transition-colors ${
                        profile.preferences.occasions.includes(occasion)
                          ? 'bg-indigo-100 text-indigo-700 border-2 border-indigo-200'
                          : 'bg-gray-50 text-gray-600 border-2 border-gray-200 hover:bg-gray-100'
                      } disabled:opacity-50 disabled:cursor-not-allowed`}
                    >
                      {occasion}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'privacy' && (
          <div className="space-y-8">
            <div>
              <h2 className="text-2xl font-bold text-gray-900 mb-6">隐私设置</h2>
              
              <div className="space-y-6">
                <div className="flex items-center justify-between p-4 bg-gray-50 rounded-xl">
                  <div>
                    <h3 className="text-lg font-medium text-gray-900">个人资料可见性</h3>
                    <p className="text-sm text-gray-600">控制谁可以查看您的个人资料</p>
                  </div>
                  <select
                    value={profile.privacy.profileVisibility}
                    onChange={(e) => setProfile(prev => ({
                      ...prev,
                      privacy: {
                        ...prev.privacy,
                        profileVisibility: e.target.value as 'public' | 'friends' | 'private'
                      }
                    }))}
                    disabled={!isEditing}
                    className="px-4 py-2 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500 disabled:bg-gray-100"
                  >
                    <option value="public">公开</option>
                    <option value="friends">仅好友</option>
                    <option value="private">私密</option>
                  </select>
                </div>

                <div className="flex items-center justify-between p-4 bg-gray-50 rounded-xl">
                  <div>
                    <h3 className="text-lg font-medium text-gray-900">显示衣橱</h3>
                    <p className="text-sm text-gray-600">允许其他用户查看您的衣橱</p>
                  </div>
                  <button
                    onClick={() => setProfile(prev => ({
                      ...prev,
                      privacy: { ...prev.privacy, showWardrobe: !prev.privacy.showWardrobe }
                    }))}
                    disabled={!isEditing}
                    className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                      profile.privacy.showWardrobe ? 'bg-indigo-600' : 'bg-gray-200'
                    } disabled:opacity-50`}
                  >
                    <span
                      className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                        profile.privacy.showWardrobe ? 'translate-x-6' : 'translate-x-1'
                      }`}
                    />
                  </button>
                </div>

                <div className="flex items-center justify-between p-4 bg-gray-50 rounded-xl">
                  <div>
                    <h3 className="text-lg font-medium text-gray-900">显示搭配</h3>
                    <p className="text-sm text-gray-600">允许其他用户查看您的搭配</p>
                  </div>
                  <button
                    onClick={() => setProfile(prev => ({
                      ...prev,
                      privacy: { ...prev.privacy, showOutfits: !prev.privacy.showOutfits }
                    }))}
                    disabled={!isEditing}
                    className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                      profile.privacy.showOutfits ? 'bg-indigo-600' : 'bg-gray-200'
                    } disabled:opacity-50`}
                  >
                    <span
                      className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                        profile.privacy.showOutfits ? 'translate-x-6' : 'translate-x-1'
                      }`}
                    />
                  </button>
                </div>

                <div className="flex items-center justify-between p-4 bg-gray-50 rounded-xl">
                  <div>
                    <h3 className="text-lg font-medium text-gray-900">个性化推荐</h3>
                    <p className="text-sm text-gray-600">允许系统根据您的偏好提供推荐</p>
                  </div>
                  <button
                    onClick={() => setProfile(prev => ({
                      ...prev,
                      privacy: { ...prev.privacy, allowRecommendations: !prev.privacy.allowRecommendations }
                    }))}
                    disabled={!isEditing}
                    className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                      profile.privacy.allowRecommendations ? 'bg-indigo-600' : 'bg-gray-200'
                    } disabled:opacity-50`}
                  >
                    <span
                      className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                        profile.privacy.allowRecommendations ? 'translate-x-6' : 'translate-x-1'
                      }`}
                    />
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'statistics' && (
          <div className="space-y-8">
            <div>
              <h2 className="text-2xl font-bold text-gray-900 mb-6">统计信息</h2>
              
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
                <div className="bg-gradient-to-br from-blue-50 to-indigo-50 p-6 rounded-2xl border border-blue-100">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-blue-600">衣橱单品</p>
                      <p className="text-3xl font-bold text-blue-900">{profile.statistics.totalItems}</p>
                    </div>
                    <div className="w-12 h-12 bg-blue-100 rounded-xl flex items-center justify-center">
                      <svg className="w-6 h-6 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                      </svg>
                    </div>
                  </div>
                </div>

                <div className="bg-gradient-to-br from-green-50 to-emerald-50 p-6 rounded-2xl border border-green-100">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-green-600">创建搭配</p>
                      <p className="text-3xl font-bold text-green-900">{profile.statistics.outfitsCreated}</p>
                    </div>
                    <div className="w-12 h-12 bg-green-100 rounded-xl flex items-center justify-center">
                      <svg className="w-6 h-6 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 100 4m0-4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 100 4m0-4v2m0-6V4" />
                      </svg>
                    </div>
                  </div>
                </div>

                <div className="bg-gradient-to-br from-purple-50 to-violet-50 p-6 rounded-2xl border border-purple-100">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-purple-600">AI推荐</p>
                      <p className="text-3xl font-bold text-purple-900">{profile.statistics.recommendationsReceived}</p>
                    </div>
                    <div className="w-12 h-12 bg-purple-100 rounded-xl flex items-center justify-center">
                      <svg className="w-6 h-6 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                      </svg>
                    </div>
                  </div>
                </div>

                <div className="bg-gradient-to-br from-pink-50 to-rose-50 p-6 rounded-2xl border border-pink-100">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-pink-600">收藏搭配</p>
                      <p className="text-3xl font-bold text-pink-900">{profile.statistics.favoriteOutfits}</p>
                    </div>
                    <div className="w-12 h-12 bg-pink-100 rounded-xl flex items-center justify-center">
                      <svg className="w-6 h-6 text-pink-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" />
                      </svg>
                    </div>
                  </div>
                </div>
              </div>

              {/* Activity Chart Placeholder */}
              <div className="bg-gray-50 rounded-2xl p-8 text-center">
                <div className="w-16 h-16 bg-gray-200 rounded-full flex items-center justify-center mx-auto mb-4">
                  <svg className="w-8 h-8 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                  </svg>
                </div>
                <h3 className="text-lg font-semibold text-gray-900 mb-2">活动统计图表</h3>
                <p className="text-gray-600">详细的使用统计和趋势分析即将推出</p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Profile;