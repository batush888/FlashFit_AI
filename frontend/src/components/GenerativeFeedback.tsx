import React, { useState, useEffect } from 'react';
import {
  generativeFeedbackService,
  FeedbackStats,
  UserPreferences,
  TrainingStatus
} from '../services/generativeFeedbackService';

interface GenerativeFeedbackProps {
  className?: string;
}

const GenerativeFeedback: React.FC<GenerativeFeedbackProps> = ({ className = '' }) => {
  const [feedbackStats, setFeedbackStats] = useState<FeedbackStats | null>(null);
  const [globalStats, setGlobalStats] = useState<FeedbackStats | null>(null);
  const [userPreferences, setUserPreferences] = useState<UserPreferences | null>(null);
  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'stats' | 'preferences' | 'training'>('stats');

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      setLoading(true);
      setError(null);

      const [stats, global, preferences, training] = await Promise.allSettled([
        generativeFeedbackService.getFeedbackStats(),
        generativeFeedbackService.getGlobalFeedbackStats(),
        generativeFeedbackService.getUserPreferences(),
        generativeFeedbackService.getTrainingStatus()
      ]);

      if (stats.status === 'fulfilled') {
        setFeedbackStats(stats.value);
      }
      if (global.status === 'fulfilled') {
        setGlobalStats(global.value);
      }
      if (preferences.status === 'fulfilled') {
        setUserPreferences(preferences.value);
      }
      if (training.status === 'fulfilled') {
        setTrainingStatus(training.value);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load feedback data');
    } finally {
      setLoading(false);
    }
  };

  const renderStatsTab = () => (
    <div className="space-y-6">
      {/* User Stats */}
      {feedbackStats && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">Your Feedback Statistics</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">{feedbackStats.total_feedback}</div>
              <div className="text-sm text-gray-600">Total Feedback</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">
                {feedbackStats.average_rating.toFixed(2)}
              </div>
              <div className="text-sm text-gray-600">Avg Rating</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-purple-600">{feedbackStats.recent_feedback_count}</div>
              <div className="text-sm text-gray-600">Recent Feedback</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-orange-600">{feedbackStats.training_queue_size}</div>
              <div className="text-sm text-gray-600">Training Queue</div>
            </div>
          </div>
        </div>
      )}

      {/* Global Stats */}
      {globalStats && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">Global Statistics</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">{globalStats.total_feedback}</div>
              <div className="text-sm text-gray-600">Total Feedback</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">
                {globalStats.average_rating.toFixed(2)}
              </div>
              <div className="text-sm text-gray-600">Avg Rating</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-purple-600">{globalStats.recent_feedback_count}</div>
              <div className="text-sm text-gray-600">Recent Feedback</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-orange-600">{globalStats.training_queue_size}</div>
              <div className="text-sm text-gray-600">Training Queue</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );

  const renderPreferencesTab = () => (
    <div className="space-y-6">
      {userPreferences && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">Learned Preferences</h3>
          <div className="mb-4">
            <div className="text-sm text-gray-600 mb-2">Confidence: {(userPreferences.confidence * 100).toFixed(1)}%</div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div 
                className="bg-blue-600 h-2 rounded-full" 
                style={{ width: `${userPreferences.confidence * 100}%` }}
              ></div>
            </div>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {Object.entries(userPreferences.preferences).map(([category, subcategories]) => (
              <div key={category} className="border rounded-lg p-4">
                <h4 className="font-medium text-gray-800 mb-3 capitalize">{category}</h4>
                {Object.entries(subcategories).map(([subcat, items]) => (
                  <div key={subcat} className="mb-2">
                    <div className="text-sm font-medium text-gray-700 capitalize">{subcat}:</div>
                    <div className="flex flex-wrap gap-1 mt-1">
                      {items.map((item, index) => (
                        <span 
                          key={index}
                          className="inline-block bg-blue-100 text-blue-800 text-xs px-2 py-1 rounded"
                        >
                          {item}
                        </span>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );

  const renderTrainingTab = () => (
    <div className="space-y-6">
      {trainingStatus && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">Model Training Status</h3>
          
          <div className="mb-6">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-gray-700">Training Progress</span>
              <span className="text-sm text-gray-600">
                {trainingStatus.training_queue_size} / {trainingStatus.min_feedback_for_training}
              </span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-3">
              <div 
                className={`h-3 rounded-full ${
                  trainingStatus.ready_for_training ? 'bg-green-600' : 'bg-blue-600'
                }`}
                style={{ 
                  width: `${Math.min(100, (trainingStatus.training_queue_size / trainingStatus.min_feedback_for_training) * 100)}%` 
                }}
              ></div>
            </div>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">{trainingStatus.training_queue_size}</div>
              <div className="text-sm text-gray-600">Queue Size</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">{trainingStatus.total_feedback}</div>
              <div className="text-sm text-gray-600">Total Feedback</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-purple-600">{trainingStatus.recent_feedback}</div>
              <div className="text-sm text-gray-600">Recent</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-orange-600">{trainingStatus.min_feedback_for_training}</div>
              <div className="text-sm text-gray-600">Min Required</div>
            </div>
          </div>

          <div className="mt-6">
            <div className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${
              trainingStatus.ready_for_training 
                ? 'bg-green-100 text-green-800' 
                : 'bg-yellow-100 text-yellow-800'
            }`}>
              <div className={`w-2 h-2 rounded-full mr-2 ${
                trainingStatus.ready_for_training ? 'bg-green-600' : 'bg-yellow-600'
              }`}></div>
              {trainingStatus.ready_for_training ? 'Ready for Training' : 'Collecting Feedback'}
            </div>
          </div>
        </div>
      )}
    </div>
  );

  if (loading) {
    return (
      <div className={`flex items-center justify-center p-8 ${className}`}>
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
        <span className="ml-2 text-gray-600">Loading feedback data...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className={`bg-red-50 border border-red-200 rounded-lg p-4 ${className}`}>
        <div className="text-red-800 font-medium">Error loading feedback data</div>
        <div className="text-red-600 text-sm mt-1">{error}</div>
        <button 
          onClick={loadData}
          className="mt-3 px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700 text-sm"
        >
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className={`max-w-6xl mx-auto ${className}`}>
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-gray-800 mb-2">Generative AI Feedback</h2>
        <p className="text-gray-600">Monitor your feedback and help improve our AI recommendations</p>
      </div>

      {/* Tab Navigation */}
      <div className="border-b border-gray-200 mb-6">
        <nav className="-mb-px flex space-x-8">
          {[
            { id: 'stats', label: 'Statistics', icon: '📊' },
            { id: 'preferences', label: 'Preferences', icon: '❤️' },
            { id: 'training', label: 'Training', icon: '🤖' }
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`py-2 px-1 border-b-2 font-medium text-sm ${
                activeTab === tab.id
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              <span className="mr-2">{tab.icon}</span>
              {tab.label}
            </button>
          ))}
        </nav>
      </div>

      {/* Tab Content */}
      {activeTab === 'stats' && renderStatsTab()}
      {activeTab === 'preferences' && renderPreferencesTab()}
      {activeTab === 'training' && renderTrainingTab()}

      {/* Refresh Button */}
      <div className="mt-6 text-center">
        <button
          onClick={loadData}
          className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
        >
          Refresh Data
        </button>
      </div>
    </div>
  );
};

export default GenerativeFeedback;