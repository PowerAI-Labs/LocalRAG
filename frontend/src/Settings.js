import React, { useState, useEffect } from 'react';
import { X, Save } from 'lucide-react';

const Settings = ({ isOpen, onClose, onSave, hasContext, onClearContext}) => {
  // Retrieve saved settings from localStorage with default values
  const [settings, setSettings] = useState({
    ollamaAPI: localStorage.getItem('ollamaAPI') || 'http://localhost:11434',
    timeout: parseInt(localStorage.getItem('timeout') || 300),
    maxMessageCount: parseInt(localStorage.getItem('maxMessageCount') || 10),
    embeddingChunkSize: parseInt(localStorage.getItem('embeddingChunkSize') || 10000),
    temperature: parseFloat(localStorage.getItem('temperature') || 0.7),
    topP: parseFloat(localStorage.getItem('topP') || 0.95)
  });

  const [availableModels, setAvailableModels] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (isOpen) {
      fetchAvailableModels();
    }
  }, [isOpen]);

  useEffect(() => {
    console.log("Settings received hasContext prop:", hasContext);
  }, [hasContext]);

  const fetchAvailableModels = async () => {
    try {
      setLoading(true);
      const response = await fetch('http://localhost:8000/models');
      if (response.ok) {
        const data = await response.json();
        setAvailableModels(data.models || []);
      }
    } catch (error) {
      console.error('Error fetching models:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleSave = () => {
    // Validate and save settings
    const validatedSettings = {
      ...settings,
      timeout: Math.max(30, Math.min(settings.timeout, 1200)), // 30s to 20m
      maxMessageCount: Math.max(1, Math.min(settings.maxMessageCount, 50)), // 1 to 50 messages
      embeddingChunkSize: Math.max(1000, Math.min(settings.embeddingChunkSize, 50000)), // 1000 to 50000
      temperature: Math.max(0, Math.min(settings.temperature, 1)),
      topP: Math.max(0, Math.min(settings.topP, 1))
    };

    // Save to localStorage
    Object.entries(validatedSettings).forEach(([key, value]) => {
      localStorage.setItem(key, value);
    });

    // Call onSave with the settings
    onSave(validatedSettings);
    onClose();
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setSettings(prev => ({
      ...prev,
      [name]: name.includes('temperature') || name.includes('topP') 
        ? parseFloat(value) 
        : parseInt(value)
    }));
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-[#1f2024] rounded-lg w-full max-w-md p-6">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-xl font-semibold text-gray-200">Settings</h2>
          <button 
            onClick={onClose}
            className="text-gray-400 hover:text-gray-200 transition-colors"
          >
            <X size={24} />
          </button>
        </div>

        <div className="space-y-6">
          {/* Ollama API Settings */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Ollama API Endpoint
            </label>
            <input
              type="text"
              name="ollamaAPI"
              value={settings.ollamaAPI}
              onChange={(e) => setSettings(prev => ({...prev, ollamaAPI: e.target.value}))}
              className="w-full bg-[#2c2d31] text-gray-200 p-2 rounded-lg border border-gray-700 focus:outline-none focus:border-blue-500"
              placeholder="http://localhost:11434"
            />
            <p className="mt-1 text-sm text-gray-400">
              Default: http://localhost:11434
            </p>
          </div>
                    {/* Available Models */}
                    <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Available Models
            </label>
            <div className="bg-[#2c2d31] rounded-lg border border-gray-700 p-2 max-h-40 overflow-y-auto">
              {loading ? (
                <div className="flex items-center justify-center py-4">
                  <div className="animate-spin rounded-full h-4 w-4 border-2 border-blue-500 border-t-transparent"></div>
                  <span className="ml-2 text-gray-400">Loading models...</span>
                </div>
              ) : availableModels.length > 0 ? (
                <ul className="space-y-1">
                  {availableModels.map((model, index) => (
                    <li key={index} className="text-gray-300 text-sm flex items-center gap-2">
                      <span className="w-2 h-2 rounded-full bg-green-500"></span>
                      {model.name}
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="text-gray-400 text-center py-4">No models found</p>
              )}
            </div>
          </div>
          
          {/* Advanced Settings */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-xs text-gray-400 mb-1">
                Timeout (seconds)
                <input
                  type="number"
                  name="timeout"
                  value={settings.timeout}
                  onChange={handleInputChange}
                  min="30"
                  max="1200"
                  className="w-full bg-[#2c2d31] text-gray-200 p-2 rounded-lg border border-gray-700 focus:outline-none focus:border-blue-500 mt-1"
                />
              </label>
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-1">
                Max Context Messages
                <input
                  type="number"
                  name="maxMessageCount"
                  value={settings.maxMessageCount}
                  onChange={handleInputChange}
                  min="1"
                  max="50"
                  className="w-full bg-[#2c2d31] text-gray-200 p-2 rounded-lg border border-gray-700 focus:outline-none focus:border-blue-500 mt-1"
                />
              </label>
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-1">
                Embedding Chunk Size
                <input
                  type="number"
                  name="embeddingChunkSize"
                  value={settings.embeddingChunkSize}
                  onChange={handleInputChange}
                  min="1000"
                  max="50000"
                  className="w-full bg-[#2c2d31] text-gray-200 p-2 rounded-lg border border-gray-700 focus:outline-none focus:border-blue-500 mt-1"
                />
              </label>
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-1">
                Temperature
                <input
                  type="number"
                  name="temperature"
                  value={settings.temperature}
                  onChange={handleInputChange}
                  min="0"
                  max="1"
                  step="0.1"
                  className="w-full bg-[#2c2d31] text-gray-200 p-2 rounded-lg border border-gray-700 focus:outline-none focus:border-blue-500 mt-1"
                />
              </label>
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-1">
                Top P
                <input
                  type="number"
                  name="topP"
                  value={settings.topP}
                  onChange={handleInputChange}
                  min="0"
                  max="1"
                  step="0.1"
                  className="w-full bg-[#2c2d31] text-gray-200 p-2 rounded-lg border border-gray-700 focus:outline-none focus:border-blue-500 mt-1"
                />
              </label>
            </div>
          </div>
        </div>
             
        <div className="mt-6 flex justify-between items-center">
        <button
          onClick={onClearContext}
          disabled={!hasContext}
          className={`px-4 py-2 rounded-lg transition-colors text-sm ${
            hasContext 
              ? 'bg-white hover:bg-red-100 text-gray-700 shadow-sm' 
              : 'bg-gray-100 text-gray-400 cursor-not-allowed'
          }`}
        >
          {hasContext ? 'Clear Context' : `No Context Available`}
        </button>
          <button
            onClick={handleSave}
            className="flex items-center gap-2 bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors"
          >
            <Save size={20} />
            Save Settings
          </button>
        </div>
      </div>
    </div>
  );
};

export default Settings;