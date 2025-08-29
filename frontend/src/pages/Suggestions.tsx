import React, { useState } from 'react';
import { Link } from 'react-router-dom';

interface OutfitSuggestion {
  id: string;
  name: string;
  occasion: string;
  weather: string;
  style: string;
  items: {
    id: string;
    name: string;
    category: string;
    image: string;
  }[];
  confidence: number;
  tags: string[];
}

const Suggestions = () => {
  const [selectedOccasion, setSelectedOccasion] = useState('all');
  const [selectedWeather, setSelectedWeather] = useState('all');
  const [isGenerating, setIsGenerating] = useState(false);

  const occasions = [
    { id: 'all', name: '全部场合', icon: '🌟' },
    { id: 'work', name: '工作', icon: '💼' },
    { id: 'casual', name: '休闲', icon: '👕' },
    { id: 'formal', name: '正式', icon: '👔' },
    { id: 'party', name: '聚会', icon: '🎉' },
    { id: 'sport', name: '运动', icon: '🏃' }
  ];

  const weatherOptions = [
    { id: 'all', name: '全部天气', icon: '🌈' },
    { id: 'sunny', name: '晴天', icon: '☀️' },
    { id: 'cloudy', name: '多云', icon: '☁️' },
    { id: 'rainy', name: '雨天', icon: '🌧️' },
    { id: 'cold', name: '寒冷', icon: '❄️' }
  ];

  // Mock data for demonstration
  const outfitSuggestions: OutfitSuggestion[] = [
    {
      id: '1',
      name: '商务休闲风',
      occasion: 'work',
      weather: 'sunny',
      style: 'business-casual',
      confidence: 95,
      tags: ['专业', '舒适', '现代'],
      items: [
        { id: '1', name: '白色衬衫', category: 'tops', image: '/api/placeholder/150/200' },
        { id: '2', name: '深蓝色西装裤', category: 'bottoms', image: '/api/placeholder/150/200' },
        { id: '3', name: '棕色皮鞋', category: 'shoes', image: '/api/placeholder/150/200' }
      ]
    },
    {
      id: '2',
      name: '周末休闲',
      occasion: 'casual',
      weather: 'cloudy',
      style: 'casual',
      confidence: 88,
      tags: ['轻松', '舒适', '时尚'],
      items: [
        { id: '4', name: '条纹T恤', category: 'tops', image: '/api/placeholder/150/200' },
        { id: '5', name: '牛仔裤', category: 'bottoms', image: '/api/placeholder/150/200' },
        { id: '6', name: '白色运动鞋', category: 'shoes', image: '/api/placeholder/150/200' }
      ]
    },
    {
      id: '3',
      name: '优雅晚宴',
      occasion: 'formal',
      weather: 'sunny',
      style: 'elegant',
      confidence: 92,
      tags: ['优雅', '正式', '经典'],
      items: [
        { id: '7', name: '黑色连衣裙', category: 'dresses', image: '/api/placeholder/150/200' },
        { id: '8', name: '高跟鞋', category: 'shoes', image: '/api/placeholder/150/200' },
        { id: '9', name: '珍珠项链', category: 'accessories', image: '/api/placeholder/150/200' }
      ]
    }
  ];

  const filteredSuggestions = outfitSuggestions.filter(suggestion => {
    const matchesOccasion = selectedOccasion === 'all' || suggestion.occasion === selectedOccasion;
    const matchesWeather = selectedWeather === 'all' || suggestion.weather === selectedWeather;
    return matchesOccasion && matchesWeather;
  });

  const generateNewSuggestions = () => {
    setIsGenerating(true);
    // Simulate AI generation
    setTimeout(() => {
      setIsGenerating(false);
    }, 2000);
  };

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Header Section */}
      <div className="bg-gradient-to-r from-indigo-600 to-purple-600 rounded-2xl shadow-xl p-8 text-white">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold mb-2">AI 穿搭建议</h1>
            <p className="text-white/90">基于您的衣橱和偏好，为您推荐完美搭配</p>
          </div>
          <div className="hidden md:flex items-center space-x-4">
            <div className="text-center">
              <div className="text-2xl font-bold">{filteredSuggestions.length}</div>
              <div className="text-sm text-white/80">个建议</div>
            </div>
            <div className="w-px h-12 bg-white/20"></div>
            <div className="text-center">
              <div className="text-2xl font-bold">95%</div>
              <div className="text-sm text-white/80">匹配度</div>
            </div>
          </div>
        </div>
      </div>

      {/* Filters and Actions */}
      <div className="bg-white rounded-2xl shadow-lg p-6 border border-gray-100">
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between space-y-4 lg:space-y-0">
          <div className="flex flex-col sm:flex-row sm:items-center space-y-4 sm:space-y-0 sm:space-x-6">
            {/* Occasion Filter */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">场合</label>
              <select
                value={selectedOccasion}
                onChange={(e) => setSelectedOccasion(e.target.value)}
                className="px-4 py-2 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-gray-50"
              >
                {occasions.map(occasion => (
                  <option key={occasion.id} value={occasion.id}>
                    {occasion.icon} {occasion.name}
                  </option>
                ))}
              </select>
            </div>

            {/* Weather Filter */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">天气</label>
              <select
                value={selectedWeather}
                onChange={(e) => setSelectedWeather(e.target.value)}
                className="px-4 py-2 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-gray-50"
              >
                {weatherOptions.map(weather => (
                  <option key={weather.id} value={weather.id}>
                    {weather.icon} {weather.name}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Generate Button */}
          <button
            onClick={generateNewSuggestions}
            disabled={isGenerating}
            className="inline-flex items-center px-6 py-3 bg-gradient-to-r from-indigo-600 to-purple-600 text-white font-medium rounded-xl hover:from-indigo-700 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all transform hover:scale-105 shadow-lg"
          >
            {isGenerating ? (
              <>
                <svg className="animate-spin w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                </svg>
                生成中...
              </>
            ) : (
              <>
                <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
                生成新建议
              </>
            )}
          </button>
        </div>
      </div>

      {/* Suggestions Grid */}
      {filteredSuggestions.length > 0 ? (
        <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-8">
          {filteredSuggestions.map((suggestion, index) => (
            <div
              key={suggestion.id}
              className="bg-white rounded-2xl shadow-lg hover:shadow-xl transition-all duration-300 border border-gray-100 group animate-slide-up"
              style={{ animationDelay: `${index * 150}ms` }}
            >
              {/* Header */}
              <div className="p-6 pb-4">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="text-xl font-bold text-gray-900 group-hover:text-indigo-600 transition-colors">
                    {suggestion.name}
                  </h3>
                  <div className="flex items-center space-x-1">
                    <svg className="w-4 h-4 text-yellow-400" fill="currentColor" viewBox="0 0 20 20">
                      <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                    </svg>
                    <span className="text-sm font-medium text-gray-600">{suggestion.confidence}%</span>
                  </div>
                </div>
                
                {/* Tags */}
                <div className="flex flex-wrap gap-2 mb-4">
                  {suggestion.tags.map((tag, tagIndex) => (
                    <span
                      key={tagIndex}
                      className="px-3 py-1 bg-indigo-50 text-indigo-600 text-sm rounded-full font-medium"
                    >
                      {tag}
                    </span>
                  ))}
                </div>
              </div>

              {/* Outfit Items */}
              <div className="px-6 pb-6">
                <div className="grid grid-cols-3 gap-3">
                  {suggestion.items.map((item, itemIndex) => (
                    <div
                      key={item.id}
                      className="group/item relative"
                    >
                      <div className="aspect-[3/4] bg-gray-100 rounded-xl overflow-hidden">
                        <div className="w-full h-full bg-gradient-to-br from-gray-200 to-gray-300 flex items-center justify-center">
                          <svg className="w-8 h-8 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                          </svg>
                        </div>
                        <div className="absolute inset-0 bg-black/0 group-hover/item:bg-black/10 transition-colors"></div>
                      </div>
                      <p className="text-xs text-gray-600 mt-2 text-center truncate">{item.name}</p>
                    </div>
                  ))}
                </div>
              </div>

              {/* Actions */}
              <div className="px-6 pb-6">
                <div className="flex space-x-3">
                  <button className="flex-1 px-4 py-2 bg-indigo-50 text-indigo-600 rounded-xl hover:bg-indigo-100 transition-colors font-medium">
                    查看详情
                  </button>
                  <button className="px-4 py-2 border border-gray-200 text-gray-600 rounded-xl hover:bg-gray-50 transition-colors">
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" />
                    </svg>
                  </button>
                  <button className="px-4 py-2 border border-gray-200 text-gray-600 rounded-xl hover:bg-gray-50 transition-colors">
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.684 13.342C8.886 12.938 9 12.482 9 12c0-.482-.114-.938-.316-1.342m0 2.684a3 3 0 110-2.684m0 2.684l6.632 3.316m-6.632-6l6.632-3.316m0 0a3 3 0 105.367-2.684 3 3 0 00-5.367 2.684zm0 9.316a3 3 0 105.367 2.684 3 3 0 00-5.367-2.684z" />
                    </svg>
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="bg-white rounded-2xl shadow-lg p-12 text-center border border-gray-100">
          <div className="w-24 h-24 bg-gradient-to-br from-indigo-100 to-purple-100 rounded-full flex items-center justify-center mx-auto mb-6">
            <svg className="w-12 h-12 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
          </div>
          <h3 className="text-xl font-bold text-gray-900 mb-2">暂无匹配的建议</h3>
          <p className="text-gray-500 mb-6">尝试调整筛选条件或上传更多服装来获得个性化建议</p>
          <div className="flex flex-col sm:flex-row gap-3 justify-center">
            <Link
              to="/upload"
              className="inline-flex items-center px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white font-medium rounded-xl hover:from-blue-700 hover:to-purple-700 transition-all transform hover:scale-105 shadow-lg"
            >
              <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
              </svg>
              上传服装
            </Link>
            <button
              onClick={generateNewSuggestions}
              className="inline-flex items-center px-6 py-3 border border-gray-300 text-gray-700 font-medium rounded-xl hover:bg-gray-50 transition-colors"
            >
              <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
              重新生成
            </button>
          </div>
        </div>
      )}

      {/* Tips Section */}
      <div className="bg-gradient-to-r from-indigo-50 to-purple-50 rounded-2xl p-6 border border-indigo-100">
        <h3 className="text-lg font-bold text-gray-900 mb-4 flex items-center">
          <svg className="w-5 h-5 mr-2 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
          </svg>
          AI 建议小贴士
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-gray-600">
          <div className="flex items-start">
            <div className="w-2 h-2 bg-indigo-500 rounded-full mt-2 mr-3 flex-shrink-0"></div>
            <div>AI会根据颜色搭配、风格协调性和场合适宜性来推荐搭配</div>
          </div>
          <div className="flex items-start">
            <div className="w-2 h-2 bg-indigo-500 rounded-full mt-2 mr-3 flex-shrink-0"></div>
            <div>上传的服装越多，AI的推荐就越准确和个性化</div>
          </div>
          <div className="flex items-start">
            <div className="w-2 h-2 bg-indigo-500 rounded-full mt-2 mr-3 flex-shrink-0"></div>
            <div>可以保存喜欢的搭配，系统会学习您的偏好</div>
          </div>
          <div className="flex items-start">
            <div className="w-2 h-2 bg-indigo-500 rounded-full mt-2 mr-3 flex-shrink-0"></div>
            <div>定期更新衣橱信息，获得最新的时尚搭配建议</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Suggestions;