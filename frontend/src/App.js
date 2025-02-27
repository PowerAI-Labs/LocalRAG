import React, { useState, useRef, useEffect } from 'react';
import { Upload, Send, X, Settings as SettingsIcon, Sun, Moon, FileText, File,Table } from 'lucide-react';
import Settings from './Settings';
import { useTheme } from './contexts/ThemeContext';
import { Eye, EyeOff } from 'lucide-react';
import Papa from 'papaparse';
import * as XLSX from 'xlsx';
import { XMLParser } from 'fast-xml-parser';

// Utility function for getting file icon
const getFileIcon = (fileType) => {
  switch (fileType) {
    case 'application/pdf':
      return <FileText className="text-red-500" />;
    case 'text/plain':
      return <File className="text-blue-500" />;
    case 'text/csv':
    case 'application/vnd.ms-excel':
    case 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
      return <Table className="text-green-500" />;
    case 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
      return <FileText className="text-blue-700" />;
    case 'application/xml':
    case 'text/xml':
      return <FileText className="text-purple-500" />;
    default:
      return <File />;
  }
};

// ThinkingAnimation Component
const ThinkingAnimation = () => (
  <div className="flex items-center justify-start gap-2 p-4">
    <div className="flex space-x-2">
      <div className="h-2 w-2 bg-blue-500 rounded-full animate-bounce [animation-delay:-0.3s]"></div>
      <div className="h-2 w-2 bg-blue-500 rounded-full animate-bounce [animation-delay:-0.15s]"></div>
      <div className="h-2 w-2 bg-blue-500 rounded-full animate-bounce"></div>
    </div>
  </div>
);

// File Preview Components
const PDFPreview = ({ file, isPreviewVisible, togglePreview }) => {
  const { isDark } = useTheme();
  
  return (
    <div className="relative h-full">
      <div className="absolute top-2 right-2 z-10">
        <button 
          onClick={togglePreview}
          className={`p-1 rounded-full transition-colors ${
            isDark 
              ? 'hover:bg-gray-700 text-gray-300' 
              : 'hover:bg-gray-100 text-gray-600'
          }`}
          title={isPreviewVisible ? "Hide Preview" : "Show Preview"}
        >
          {isPreviewVisible ? <EyeOff size={20} /> : <Eye size={20} />}
        </button>
      </div>
      
      {isPreviewVisible ? (
        <div className={`w-full h-full ${isDark ? 'bg-[#2c2d31]' : 'bg-white'} rounded-lg overflow-hidden`}>
          <iframe
            src={URL.createObjectURL(file)}
            className="w-full h-full"
            title="PDF Preview"
          />
        </div>
      ) : (
        <div className={`w-full h-full flex items-center justify-center ${
          isDark ? 'bg-[#2c2d31] text-gray-300' : 'bg-white text-gray-700'
        }`}>
          <p>Preview hidden. Click eye icon to show.</p>
        </div>
      )}
    </div>
  );
};

const WordPreview = ({ file, isPreviewVisible, togglePreview }) => {
  const { isDark } = useTheme();
  const [content, setContent] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const readFile = () => {
      const reader = new FileReader();
      
      reader.onload = (e) => {
        try {
          const text = e.target.result;
          const limitedText = typeof text === 'string' 
            ? text.substring(0, 1000) 
            : 'Unable to read file content';
          
          setContent(limitedText);
          setLoading(false);
        } catch (err) {
          console.error('File read error:', err);
          setError('Error reading file');
          setLoading(false);
        }
      };

      reader.onerror = () => {
        setError('Error reading file');
        setLoading(false);
      };

      reader.readAsText(file);
    };

    readFile();
  }, [file]);

  return (
    <div className={`relative h-full p-4 ${isDark ? 'bg-[#2c2d31] text-gray-200' : 'bg-white text-gray-800'}`}>
      <div className="absolute top-2 right-2 z-10">
        <button 
          onClick={togglePreview}
          className={`p-1 rounded-full transition-colors ${
            isDark 
              ? 'hover:bg-gray-700 text-gray-300' 
              : 'hover:bg-gray-100 text-gray-600'
          }`}
          title={isPreviewVisible ? "Hide Preview" : "Show Preview"}
        >
          {isPreviewVisible ? <EyeOff size={20} /> : <Eye size={20} />}
        </button>
      </div>

      {loading ? (
        <div className="flex items-center justify-center py-4">
          <div className="animate-spin rounded-full h-4 w-4 border-2 border-blue-500 border-t-transparent"></div>
          <span className="ml-2 text-gray-400">Loading document...</span>
        </div>
      ) : error ? (
        <p className="text-sm text-red-500">{error}</p>
      ) : isPreviewVisible ? (
        <div className="max-h-full overflow-y-auto">
          <pre className="whitespace-pre-wrap font-sans text-sm">
            {content || 'No content found'}
            {content.length === 1000 && '...'}
          </pre>
        </div>
      ) : (
        <div className="flex items-center justify-center h-full">
          <p className="text-sm opacity-70">Preview hidden. Click eye icon to show.</p>
        </div>
      )}
    </div>
  );
};

const TextPreview = ({ file, isPreviewVisible, togglePreview }) => {
  const { isDark } = useTheme();
  const [content, setContent] = useState('');

  useEffect(() => {
    const reader = new FileReader();
    reader.onload = (e) => setContent(e.target.result);
    reader.readAsText(file);
  }, [file]);

  return (
    <div className={`relative h-full p-4 ${isDark ? 'bg-[#2c2d31] text-gray-200' : 'bg-white text-gray-800'}`}>
      <div className="absolute top-2 right-2 z-10">
        <button 
          onClick={togglePreview}
          className={`p-1 rounded-full transition-colors ${
            isDark 
              ? 'hover:bg-gray-700 text-gray-300' 
              : 'hover:bg-gray-100 text-gray-600'
          }`}
          title={isPreviewVisible ? "Hide Preview" : "Show Preview"}
        >
          {isPreviewVisible ? <EyeOff size={20} /> : <Eye size={20} />}
        </button>
      </div>

      {isPreviewVisible ? (
        <pre className="whitespace-pre-wrap font-mono text-sm h-full overflow-y-auto">
          {content}
        </pre>
      ) : (
        <div className="flex items-center justify-center h-full">
          <p className="text-sm opacity-70">Preview hidden. Click eye icon to show.</p>
        </div>
      )}
    </div>
  );
};

const ImagePreview = ({ file, isPreviewVisible, togglePreview }) => {
  const { isDark } = useTheme();
  
  return (
    <div className={`relative h-full ${isDark ? 'bg-[#2c2d31]' : 'bg-white'}`}>
      <div className="absolute top-2 right-2 z-10">
        <button
          onClick={togglePreview}
          className={`p-1 rounded-full transition-colors ${
            isDark
              ? 'hover:bg-gray-700 text-gray-300'
              : 'hover:bg-gray-100 text-gray-600'
          }`}
          title={isPreviewVisible ? "Hide Preview" : "Show Preview"}
        >
          {isPreviewVisible ? <EyeOff size={20} /> : <Eye size={20} />}
        </button>
      </div>

      {isPreviewVisible ? (
        <img
          src={URL.createObjectURL(file)}
          alt="Preview"
          className="object-contain h-full w-full"
        />
      ) : (
        <div className="flex items-center justify-center h-full">
          <p className="text-sm opacity-70">Preview hidden. Click eye icon to show.</p>
        </div>
      )}
    </div>
  );
}

// CSV/Excel Preview Component
const SpreadsheetPreview = ({ file, isPreviewVisible, togglePreview }) => {
  const { isDark } = useTheme();
  const [sheetData, setSheetData] = useState([]);
  const [sheetNames, setSheetNames] = useState([]);
  const [selectedSheet, setSelectedSheet] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const readFile = () => {
      const reader = new FileReader();
      
      reader.onload = (e) => {
        try {
          const workbook = XLSX.read(e.target.result, { type: 'binary' });
          const sheets = workbook.SheetNames;
          setSheetNames(sheets);
          
          // Default to first sheet
          const firstSheetName = sheets[0];
          setSelectedSheet(firstSheetName);
          
          // Convert sheet to array of objects
          const worksheet = workbook.Sheets[firstSheetName];
          const jsonData = XLSX.utils.sheet_to_json(worksheet, { header: 1 });
          
          // Limit to first 100 rows
          setSheetData(jsonData.slice(0, 101)); // Include header + 100 rows
          setLoading(false);
        } catch (err) {
          console.error('File read error:', err);
          setError('Error reading spreadsheet');
          setLoading(false);
        }
      };

      reader.onerror = () => {
        setError('Error reading file');
        setLoading(false);
      };

      reader.readAsBinaryString(file);
    };

    readFile();
  }, [file]);

  const changeSheet = (sheetName) => {
    const workbook = XLSX.read(file, { type: 'binary' });
    const worksheet = workbook.Sheets[sheetName];
    const jsonData = XLSX.utils.sheet_to_json(worksheet, { header: 1 });
    
    // Limit to first 100 rows
    setSheetData(jsonData.slice(0, 101)); // Include header + 100 rows
    setSelectedSheet(sheetName);
  };

  return (
    <div className={`relative h-full p-4 ${isDark ? 'bg-[#2c2d31] text-gray-200' : 'bg-white text-gray-800'}`}>
      <div className="absolute top-2 right-2 z-10 flex items-center gap-2">
        {/* Row count indicator */}
        <span className="text-sm text-gray-500">
          Showing first {Math.min(sheetData.length - 1, 100)} rows
        </span>

        {/* Sheet selector */}
        {sheetNames.length > 1 && (
          <select 
            value={selectedSheet || ''}
            onChange={(e) => changeSheet(e.target.value)}
            className={`p-1 rounded text-sm ${
              isDark 
                ? 'bg-[#1f2024] text-gray-300 border-gray-700' 
                : 'bg-gray-100 text-gray-700 border-gray-200'
            }`}
          >
            {sheetNames.map(name => (
              <option key={name} value={name}>{name}</option>
            ))}
          </select>
        )}
        
        <button 
          onClick={togglePreview}
          className={`p-1 rounded-full transition-colors ${
            isDark 
              ? 'hover:bg-gray-700 text-gray-300' 
              : 'hover:bg-gray-100 text-gray-600'
          }`}
          title={isPreviewVisible ? "Hide Preview" : "Show Preview"}
        >
          {isPreviewVisible ? <EyeOff size={20} /> : <Eye size={20} />}
        </button>
      </div>

      {loading ? (
        <div className="flex items-center justify-center py-4">
          <div className="animate-spin rounded-full h-4 w-4 border-2 border-blue-500 border-t-transparent"></div>
          <span className="ml-2 text-gray-400">Loading spreadsheet...</span>
        </div>
      ) : error ? (
        <p className="text-sm text-red-500">{error}</p>
      ) : isPreviewVisible ? (
        <div className="max-h-full overflow-auto">
          <table className={`w-full border-collapse text-sm ${
            isDark ? 'text-gray-300' : 'text-gray-700'
          }`}>
            <thead>
              <tr>
                {sheetData[0]?.map((header, index) => (
                  <th 
                    key={index} 
                    className={`p-2 border ${
                      isDark ? 'border-gray-700 bg-[#1f2024]' : 'border-gray-200 bg-gray-100'
                    }`}
                  >
                    {header}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {sheetData.slice(1).map((row, rowIndex) => (
                <tr 
                  key={rowIndex} 
                  className={`${
                    rowIndex % 2 === 0 
                      ? isDark 
                        ? 'bg-[#1a1b1e]' 
                        : 'bg-white' 
                      : isDark 
                        ? 'bg-[#1f2024]' 
                        : 'bg-gray-50'
                  }`}
                >
                  {row.map((cell, cellIndex) => (
                    <td 
                      key={cellIndex} 
                      className={`p-2 border ${
                        isDark ? 'border-gray-700' : 'border-gray-200'
                      }`}
                    >
                      {cell}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="flex items-center justify-center h-full">
          <p className="text-sm opacity-70">Preview hidden. Click eye icon to show.</p>
        </div>
      )}
    </div>
  );
};

// XML Preview Component
const XMLPreview = ({ file, isPreviewVisible, togglePreview }) => {
  const { isDark } = useTheme();
  const [xmlContent, setXmlContent] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [parsedXml, setParsedXml] = useState(null);

  useEffect(() => {
    const reader = new FileReader();
    
    reader.onload = (e) => {
      try {
        const xmlText = e.target.result;
        setXmlContent(xmlText);
        
        // Parse XML
        const parser = new XMLParser({
          ignoreAttributes: false,
          attributeNamePrefix: "@_",
          parseAttributeValue: true
        });
        const jsonObj = parser.parse(xmlText);
        
        setParsedXml(jsonObj);
        setLoading(false);
      } catch (err) {
        console.error('XML parsing error:', err);
        setError('Error parsing XML');
        setLoading(false);
      }
    };

    reader.onerror = () => {
      setError('Error reading file');
      setLoading(false);
    };

    reader.readAsText(file);
  }, [file]);

  // Recursive function to render nested JSON/XML structure
  const renderXmlStructure = (obj, depth = 0) => {
    if (obj === null || obj === undefined) return null;
    
    if (typeof obj !== 'object') {
      return <div className="pl-4">{String(obj)}</div>;
    }

    return Object.entries(obj).map(([key, value]) => {
      // Skip attribute keys
      if (key.startsWith('@_')) return null;

      return (
        <div key={key} className={`pl-${depth * 4}`}>
          <div className="font-semibold">
            {key}:
            {/* Render attributes if exist */}
            {Object.entries(obj)
              .filter(([attrKey]) => attrKey.startsWith('@_' + key))
              .map(([attrKey, attrValue]) => (
                <span 
                  key={attrKey} 
                  className="ml-2 text-sm text-gray-500"
                >
                  {attrKey.replace('@_' + key, '')}="
                  {String(attrValue)}"
                </span>
              ))}
          </div>
          {renderXmlStructure(value, depth + 1)}
        </div>
      );
    });
  };

  return (
    <div className={`relative h-full p-4 ${isDark ? 'bg-[#2c2d31] text-gray-200' : 'bg-white text-gray-800'}`}>
      <div className="absolute top-2 right-2 z-10">
        <button 
          onClick={togglePreview}
          className={`p-1 rounded-full transition-colors ${
            isDark 
              ? 'hover:bg-gray-700 text-gray-300' 
              : 'hover:bg-gray-100 text-gray-600'
          }`}
          title={isPreviewVisible ? "Hide Preview" : "Show Preview"}
        >
          {isPreviewVisible ? <EyeOff size={20} /> : <Eye size={20} />}
        </button>
      </div>

      {loading ? (
        <div className="flex items-center justify-center py-4">
          <div className="animate-spin rounded-full h-4 w-4 border-2 border-blue-500 border-t-transparent"></div>
          <span className="ml-2 text-gray-400">Loading XML...</span>
        </div>
      ) : error ? (
        <p className="text-sm text-red-500">{error}</p>
      ) : isPreviewVisible ? (
        <div>
          <div className="max-h-96 overflow-y-auto">
            {parsedXml && renderXmlStructure(parsedXml)}
          </div>
          <details className="mt-4">
            <summary className="cursor-pointer text-sm">Raw XML</summary>
            <pre className={`text-xs overflow-x-auto p-2 rounded ${
              isDark ? 'bg-[#1f2024]' : 'bg-gray-100'
            }`}>
              {xmlContent}
            </pre>
          </details>
        </div>
      ) : (
        <div className="flex items-center justify-center h-full">
          <p className="text-sm opacity-70">Preview hidden. Click eye icon to show.</p>
        </div>
      )}
    </div>
  );
};

const Message = ({ message, isThinking }) => {
  const { isDark } = useTheme();
  const [isThinkingExpanded, setIsThinkingExpanded] = useState(true);

  // Check if the message has the thinking tag
  const isThinkingMessage = message.content.includes('<think>');
  
  // Split message into thinking and response parts
  let thinkingContent = '';
  let responseContent = message.content;

  if (isThinkingMessage) {
    const thinkMatch = message.content.match(/<think>(.*?)<\/think>/s);
    if (thinkMatch) {
      thinkingContent = thinkMatch[1].trim();
      responseContent = message.content.split('</think>')[1]?.trim() || '';
    }
  }

  // Theme colors
  const bgColors = isDark ? {
    user: 'bg-[#2563eb]',
    assistant: 'bg-[#1f2024]',
    thinking: 'bg-[#1e2939]',
    facet: 'bg-[#2c2d31]',
    highlight: 'bg-[#2d3748]'
  } : {
    user: 'bg-blue-500',
    assistant: 'bg-white',
    thinking: 'bg-gray-100',
    facet: 'bg-gray-50',
    highlight: 'bg-blue-50'
  };

  const textColors = isDark ? {
    normal: 'text-gray-200',
    muted: 'text-gray-400'
  } : {
    normal: 'text-gray-800',
    muted: 'text-gray-500'
  };

  // Format enhanced search content
  const formatEnhancedContent = (content) => {
    if (content.includes("Based on semantic relationships") || 
        content.includes("Based on content analysis") ||
        content.includes("Using fuzzy matching results")) {
      
      const sections = content.split('\n\n');

      return (
        <div className="space-y-4">
          {/* Search Analysis Header */}
          <div className={`p-3 rounded-lg ${bgColors.facet}`}>
            <p className="text-sm font-medium text-blue-400 mb-2">Search Analysis</p>
            <p className="text-sm">{sections[0]}</p>
          </div>

          {/* Related Terms Section */}
          {content.includes("Related terms") && (
            <div className="flex flex-wrap gap-2">
              {content.match(/Related terms.*?: (.*)/)[1].split(', ').map((term, idx) => (
                <span 
                  key={idx}
                  className={`px-2 py-1 rounded-full text-xs ${
                    isDark ? 'bg-blue-900 text-blue-100' : 'bg-blue-100 text-blue-800'
                  }`}
                >
                  {term.trim()}
                </span>
              ))}
            </div>
          )}

          {/* Categories Section */}
          {content.includes("Key categories") && (
            <div className={`p-3 rounded-lg ${bgColors.facet}`}>
              <p className="text-sm font-medium text-blue-400 mb-2">Categories Found</p>
              {content.match(/Key categories found:\n([\s\S]*?)(?:\n\n|$)/)[1]
                .split('\n')
                .map((category, idx) => (
                  <div key={idx} className="text-sm mb-1">
                    {category.trim()}
                  </div>
                ))}
            </div>
          )}

          {/* Main Content */}
          <div className="space-y-3">
            {sections.slice(1).map((section, idx) => {
              if (section.startsWith('Context')) {
                const [contextNum, ...contextContent] = section.split('\n');
                return (
                  <div key={idx} className={`p-3 rounded-lg ${bgColors.highlight}`}>
                    <p className="text-sm font-medium text-blue-400 mb-2">{contextNum}</p>
                    <p className="text-sm whitespace-pre-wrap">{contextContent.join('\n')}</p>
                  </div>
                );
              }
              return (
                <p key={idx} className="text-sm whitespace-pre-wrap">{section}</p>
              );
            })}
          </div>
        </div>
      );
    }

    // Return regular content if not enhanced search
    return <span className="whitespace-pre-wrap">{content}</span>;
  };

  // Separate components for thinking and response
  const ThinkingSection = () => (
    <div className="flex justify-start">
      <div className={`max-w-[80%] ${bgColors.thinking} rounded-lg overflow-hidden shadow-sm`}>
        <div 
          className={`flex items-center gap-2 p-2 cursor-pointer ${isDark ? 'hover:bg-[#2c3b4f]' : 'hover:bg-gray-200'}`}
          onClick={() => setIsThinkingExpanded(!isThinkingExpanded)}
        >
          <div className="h-6 w-6 rounded-full bg-orange-500 flex items-center justify-center">
            <span className="text-xs text-white">AI</span>
          </div>
          <span className={`text-sm ${isDark ? 'text-blue-300' : 'text-blue-600'}`}>Thinking</span>
          <svg
            className={`w-4 h-4 transition-transform ${isThinkingExpanded ? 'rotate-180' : ''} ${isDark ? 'text-blue-300' : 'text-blue-600'}`}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </div>
        <div 
          className={`transition-all duration-200 ease-in-out ${
            isThinkingExpanded ? 'max-h-96 p-3' : 'max-h-0'
          } overflow-hidden border-t ${isDark ? 'border-gray-700' : 'border-gray-200'}`}
        >
          <div className={`whitespace-pre-wrap ${textColors.normal}`}>{thinkingContent}</div>
        </div>
      </div>
    </div>
  );

  const ResponseSection = () => (
    <div className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}>
      <div className={`max-w-[80%] rounded-lg p-3 ${
        message.type === 'user' 
          ? `${bgColors.user} text-white` 
          : `${bgColors.assistant} ${textColors.normal}`
      }`}>
        <div className="flex items-start gap-2">
          {message.type === 'user' ? (
            <>
              <div className="h-6 w-6 rounded-full bg-blue-500 flex items-center justify-center">
                <span className="text-xs">You</span>
              </div>
              <span>{message.content}</span>
            </>
          ) : (
            <>
              <div className="h-6 w-6 rounded-full bg-orange-500 flex items-center justify-center">
                <span className="text-xs">AI</span>
              </div>
              <div>
                {formatEnhancedContent(message.content)}
                
                {/* Render images */}
                {message.images && message.images.map((img, index) => (
                  <div key={index} className="mt-2">
                    <img 
                      src={`data:image/png;base64,${img}`} 
                      alt={`Generated image ${index + 1}`} 
                      className="max-w-full rounded-lg shadow-md"
                    />
                  </div>
                ))}
                
                {/* Render attachments */}
                {message.attachments && message.attachments.map((attachment, index) => (
                  <div 
                    key={index} 
                    className={`mt-2 p-2 rounded-lg ${
                      isDark ? 'bg-[#2c2d31]' : 'bg-gray-100'
                    }`}
                  >
                    <span className="font-medium">{attachment.name}</span>
                    <p className="text-sm opacity-70">{attachment.description}</p>
                  </div>
                ))}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );

  return (
    <div className="flex flex-col gap-3">
      {thinkingContent && <ThinkingSection />}
      {responseContent && !isThinking && <ResponseSection />}
      {isThinking && <ThinkingAnimation />}
    </div>
  );
};

// File handling types and functions
const FileCard = ({ file, onRemove, onSelect, isSelected, isDark }) => {
  return (
    <div 
      className={`relative p-3 rounded-lg cursor-pointer border transition-all ${
        isSelected 
          ? isDark 
            ? 'bg-[#2c2d31] border-blue-500' 
            : 'bg-blue-50 border-blue-500'
          : isDark
            ? 'bg-[#1f2024] border-gray-700 hover:border-gray-600'
            : 'bg-white border-gray-200 hover:border-gray-300'
      }`}
      onClick={onSelect}
    >
      <div className="flex items-center gap-3">
        {getFileIcon(file.type)}
        <div className="flex-1 min-w-0">
          <p className={`text-sm font-medium truncate ${
            isDark ? 'text-gray-200' : 'text-gray-700'
          }`}>
            {file.name}
          </p>
          <p className={`text-xs ${
            isDark ? 'text-gray-400' : 'text-gray-500'
          }`}>
            {(file.size / 1024 / 1024).toFixed(2)} MB
          </p>
        </div>
        <button
          onClick={(e) => {
            e.stopPropagation();
            onRemove();
          }}
          className={`p-1 rounded-full hover:bg-opacity-80 ${
            isDark ? 'hover:bg-gray-700' : 'hover:bg-gray-100'
          }`}
        >
          <X size={16} className={isDark ? 'text-gray-400' : 'text-gray-500'} />
        </button>
      </div>
    </div>
  );
};

// File Preview Handler
const FilePreview = ({ file }) => {
  const [isPreviewVisible, setIsPreviewVisible] = useState(true);

  const togglePreview = () => {
    setIsPreviewVisible(!isPreviewVisible);
  };

  if (!file) return null;

  switch (file.type) {
    case 'application/pdf':
      return <PDFPreview file={file} isPreviewVisible={isPreviewVisible} togglePreview={togglePreview} />;
    case 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
      return <WordPreview file={file} isPreviewVisible={isPreviewVisible} togglePreview={togglePreview} />;
    case 'text/plain':
      return <TextPreview file={file} isPreviewVisible={isPreviewVisible} togglePreview={togglePreview} />;
    case 'image/jpeg':
    case 'image/jpg':
    case 'image/png':
    case 'image/gif':
    case 'image/bmp':
    case 'image/webp':
    case 'image/svg+xml':
      return <ImagePreview file={file} isPreviewVisible={isPreviewVisible} togglePreview={togglePreview} />;
    case 'text/csv':
    case 'application/vnd.ms-excel':
    case 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
      return <SpreadsheetPreview file={file} isPreviewVisible={isPreviewVisible} togglePreview={togglePreview} />;
    case 'application/xml':
    case 'text/xml':
      return <XMLPreview file={file} isPreviewVisible={isPreviewVisible} togglePreview={togglePreview} />;
    default:
      return null;
  }
};

const App = () => {
  const { isDark, toggleTheme } = useTheme();
  // Core states
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [isThinking, setIsThinking] = useState(false);
  const [input, setInput] = useState('');
  const [isEnhancedSearch, setIsEnhancedSearch] = useState(false);
  const messagesEndRef = useRef(null);

   // Settings states
   const [settings, setSettings] = useState({
    ollamaAPI: localStorage.getItem('ollamaAPI') || 'http://localhost:11434',
    timeout: parseInt(localStorage.getItem('timeout') || 300),
    maxMessageCount: parseInt(localStorage.getItem('maxMessageCount') || 10),
    embeddingChunkSize: parseInt(localStorage.getItem('embeddingChunkSize') || 10000),
    temperature: parseFloat(localStorage.getItem('temperature') || 0.7),
    topP: parseFloat(localStorage.getItem('topP') || 0.95)
  });

  // Batch processing states
  const [isBatchMode, setIsBatchMode] = useState(false);
  const [batchJobs, setBatchJobs] = useState([]);
  const [activeBatchId, setActiveBatchId] = useState(null);
  const [batchStatus, setBatchStatus] = useState(null);

  // File handling states [existing states remain the same]
  const [files, setFiles] = useState([]);
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState('');
  const fileInputRef = useRef(null);
  const [previewKey, setPreviewKey] = useState(0);

   // Settings and model states
   const [isSettingsOpen, setIsSettingsOpen] = useState(false);
    const [ollamaAPI, setOllamaAPI] = useState(settings.ollamaAPI);
   const [selectedModel, setSelectedModel] = useState(localStorage.getItem('selectedModel') || 'deepseek-r1:8b');
   const [availableModels, setAvailableModels] = useState([]);
   const [hasContext, setHasContext] = useState(false);

   // Polling interval for batch status
  const POLLING_INTERVAL = 3000; // 3 seconds

  // Poll batch status
  useEffect(() => {
    let pollingInterval;
    if (activeBatchId) {
      pollingInterval = setInterval(async () => {
        await checkBatchStatus(activeBatchId);
      }, POLLING_INTERVAL);
    }
    return () => clearInterval(pollingInterval);
  }, [activeBatchId]);

  // Check batch status
  const checkBatchStatus = async (batchId) => {
    try {
      const response = await fetch(`http://localhost:8000/api/v1/batch/${batchId}`);
      if (response.ok) {
        const status = await response.json();
        setBatchStatus(status);
        
        // Update UI based on status
        if (status.status === 'completed') {
          setUploadStatus('Batch processing completed successfully!');
          setActiveBatchId(null);
        } else if (status.status === 'failed') {
          setUploadStatus('Batch processing failed. Please check logs.');
          setActiveBatchId(null);
        }
      }
    } catch (error) {
      console.error('Error checking batch status:', error);
    }
  };

  //Check context status function
  const checkContextStatus = async () => {
    try {
      const response = await fetch('http://localhost:8000/context-status');
      if (response.ok) {
        const data = await response.json();
        console.log("Context status API response:", data);
        
        if (data.total_chunks > 0) {
          console.log("Setting hasContext to TRUE");
          setHasContext(true);
        } else {
          console.log("Setting hasContext to FALSE - no chunks");
          setHasContext(false);
        }
      }
    } catch (error) {
      console.error('Error checking context status:', error);
      setHasContext(false);
    }
  };
  
  // Add useEffect to check context status periodically or after specific actions
  useEffect(() => {
    checkContextStatus();
  }, []);

  /// Enhanced file upload handler
  const handleFileUpload = async (event) => {
    const newFiles = Array.from(event.target.files);
    const allowedTypes = [
      'application/pdf', 'text/plain',
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
      'image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/bmp', 'image/webp', 'image/svg+xml',
      'text/csv', 'application/vnd.ms-excel',
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
      'application/xml', 'text/xml'
    ];

    const validFiles = newFiles.filter(file => 
      allowedTypes.includes(file.type) && file.size <= 50 * 1024 * 1024
    );

    if (validFiles.length !== newFiles.length) {
      setUploadStatus('Some files were skipped due to invalid type or size');
    }

    setFiles(prev => [...prev, ...validFiles]);
    if (validFiles.length > 0 && !selectedFile) {
      setSelectedFile(validFiles[0]);
    }

    // Process files based on mode
    if (isBatchMode) {
      await handleBatchUpload(validFiles);
    } else {
      await handleStandardUpload(validFiles);
    }
  };

  // Batch upload handler
  const handleBatchUpload = async (files) => {
    try {
      setUploadStatus('Preparing batch upload...');
      
      // Create FormData with all files
      const formData = new FormData();
      files.forEach(file => {
        formData.append('files', file);
      });
  
      // Upload files using batch upload endpoint
      const response = await fetch('http://localhost:8000/api/v1/batch/upload', {
        method: 'POST',
        body: formData
      });
  
      if (!response.ok) throw new Error('Batch upload failed');
  
      const data = await response.json();
      setActiveBatchId(data.batch_id);
      setBatchJobs(prev => [...prev, data]);
      setUploadStatus(`Batch job ${data.batch_id} started. Processing ${data.files_uploaded} files...`);
  
    } catch (error) {
      console.error('Batch upload error:', error);
      setUploadStatus('Failed to start batch processing');
    }
  };

  // Standard upload handler
  const handleStandardUpload = async (files) => {
    for (const file of files) {
      try {
        setUploadStatus(`Processing ${file.name}...`);
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch('http://localhost:8000/upload', {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) throw new Error('Upload failed');

        setUploadStatus(`${file.name} processed successfully!`);
      } catch (error) {
        setUploadStatus(`Failed to process ${file.name}`);
        console.error('Upload error:', error);
      }
    }
  };

  // Batch job management
  const cancelBatchJob = async (batchId) => {
    try {
      const response = await fetch(`http://localhost:8000/api/v1/batch/${batchId}`, {
        method: 'DELETE'
      });

      if (response.ok) {
        setBatchJobs(prev => prev.filter(job => job.batch_id !== batchId));
        if (activeBatchId === batchId) {
          setActiveBatchId(null);
        }
        setUploadStatus('Batch job cancelled successfully');
      }
    } catch (error) {
      console.error('Error cancelling batch job:', error);
    }
  };

  // Batch Progress Component
  const BatchProgress = ({ status }) => {
    if (!status) return null;

    const progress = (status.processed_files / status.total_files) * 100;

    return (
      <div className={`p-4 rounded-lg ${isDark ? 'bg-[#2c2d31]' : 'bg-white'} mb-4`}>
        <div className="flex justify-between mb-2">
          <span className={`text-sm ${isDark ? 'text-gray-300' : 'text-gray-700'}`}>
            Batch Progress
          </span>
          <span className={`text-sm ${isDark ? 'text-blue-400' : 'text-blue-600'}`}>
            {Math.round(progress)}%
          </span>
        </div>
        <div className="w-full h-2 bg-gray-200 rounded-full overflow-hidden">
          <div 
            className="h-full bg-blue-500 transition-all duration-500"
            style={{ width: `${progress}%` }}
          />
        </div>
        <div className="mt-2 text-xs text-gray-500">
          {status.processed_files} / {status.total_files} files processed
        </div>
      </div>
    );
  };

  // Add Batch Mode Toggle to your UI
  const BatchModeToggle = () => (
    <div className={`flex items-center gap-2 px-3 py-2 rounded-lg border ${
      isDark ? 'border-gray-700' : 'border-gray-200'
    }`}>
      <span className={`text-sm ${isDark ? 'text-gray-300' : 'text-gray-700'}`}>
        Processing Mode:
      </span>
      <button
        onClick={() => setIsBatchMode(!isBatchMode)}
        className={`px-3 py-1 rounded-lg text-sm transition-colors ${
          isBatchMode
            ? 'bg-blue-600 text-white'
            : isDark
              ? 'bg-[#2c2d31] text-gray-400'
              : 'bg-gray-100 text-gray-600'
        }`}
      >
        {isBatchMode ? 'Batch' : 'Standard'}
      </button>
    </div>
  );




// Fetch available models
const fetchAvailableModels = async () => {
  try {
    const response = await fetch('http://localhost:8000/models');
    if (response.ok) {
      const data = await response.json();
      if (data.models && Array.isArray(data.models)) {
        setAvailableModels(data.models);
        if (!selectedModel && data.models.length > 0) {
          setSelectedModel(data.models[0].name);
          localStorage.setItem('selectedModel', data.models[0].name);
        }
      }
    }
  } catch (error) {
    console.error('Error fetching models:', error);
  }
};

// Clear context function
const clearContext = async () => {
  try {
    setLoading(true);
    const response = await fetch('http://localhost:8000/clear-context', {
      method: 'POST'
    });
    
    if (response.ok) {
      setHasContext(false);
      setMessages([]);
      setFiles([]);
      setSelectedFile(null);
      setUploadStatus('');
      await checkContextStatus();
      // Show success message
      setMessages([{
        type: 'system',
        content: 'Context has been cleared. You can start a new conversation.'
      }]);
    } else {
      throw new Error('Failed to clear context');
    }
  } catch (error) {
    console.error('Error clearing context:', error);
    setMessages(prev => [...prev, {
      type: 'system',
      content: 'Failed to clear context. Please try again.'
    }]);
  } finally {
    setLoading(false);
  }
};

const openSettings = async () => {
  await checkContextStatus(); // Refresh context status first
  setIsSettingsOpen(true);
};

// Settings save handler
const handleSettingsSave = (newSettings) => {
  setSettings(newSettings);
  localStorage.setItem('ollamaAPI', settings.ollamaAPI);
  fetchAvailableModels();
};
  const removeFile = (fileToRemove) => {
    setFiles(prev => prev.filter(f => f !== fileToRemove));
    if (selectedFile === fileToRemove) {
      setSelectedFile(files.length > 1 ? files[0] : null);
    }
  };

  // Auto-scroll and model fetching effects
  useEffect(() => {
    fetchAvailableModels();
  }, [ollamaAPI]);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };
  // Query handling and message processing
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;
  
    const requestId = new Date().toISOString();
    const userMessage = input.trim();
    setInput('');
  
    setMessages(prev => {
      const updatedMessages = [...prev, { type: 'user', content: userMessage }];
      const messagesWithThinking = [
        ...updatedMessages, 
        { 
          type: 'assistant', 
          content: '<think>Thinking...</think>Processing your request...' 
        }
      ];
      return messagesWithThinking.length > settings.maxMessageCount 
        ? messagesWithThinking.slice(-settings.maxMessageCount) 
        : messagesWithThinking;
    });
  
    setLoading(true);
    setIsThinking(true);
    scrollToBottom();
  
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => {
        controller.abort();
        console.log(`[${requestId}] Request timed out after ${settings.timeout} seconds`);
      }, settings.timeout * 1000);
  
      // Choose endpoint based on search mode
      const endpoint = isEnhancedSearch ? '/enhanced-search' : '/query';
      
      const requestBody = {
        question: userMessage,
        model: selectedModel,
        context_window: settings.embeddingChunkSize,
        timeout: settings.timeout,
        temperature: settings.temperature,
        top_p: settings.topP,
      };
  
      // Add enhanced search parameters if needed
      if (isEnhancedSearch) {
        requestBody.semantic_search = true;
        requestBody.fuzzy_matching = true;
        requestBody.include_facets = true;
        requestBody.page = 1;
        requestBody.page_size = 5;
      }
  
      const response = await fetch(`http://localhost:8000${endpoint}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
        signal: controller.signal
      });
  
      clearTimeout(timeoutId);
  
      if (!response.ok) throw new Error('Query failed');
      
      const data = await response.json();
      
      // Handle enhanced search response
      const processedResponse = {
        type: 'assistant',
        content: isEnhancedSearch ? data.metadata.llm_response : data.response,
        images: data.images || [],
        attachments: data.attachments || []
      };
  
      // Update messages
      setMessages(prevMessages => {
        const filteredMessages = prevMessages.slice(0, -2);
        const updatedMessages = [
          ...filteredMessages, 
          { type: 'user', content: userMessage },
          processedResponse
        ];
        return updatedMessages.length > settings.maxMessageCount 
          ? updatedMessages.slice(-settings.maxMessageCount) 
          : updatedMessages;
      });
  
    
      // Error handling remains the same
    
    } catch (error) {
      console.error(`[${requestId}] Error:`, error);
      
      // Replace thinking message with error message
      setMessages(prevMessages => {
        // Remove the last two messages (thinking message and previous user message)
        const filteredMessages = prevMessages.slice(0, -2);
        
        const errorMessage = error.name === 'AbortError' 
          ? `Request timed out after ${settings.timeout} seconds.`
          : 'Sorry, there was an error processing your request.';
  
        const updatedMessages = [
          ...filteredMessages, 
          { type: 'user', content: userMessage },
          {
            type: 'system',
            content: errorMessage
          }
        ];
  
        return updatedMessages.length > settings.maxMessageCount 
          ? updatedMessages.slice(-settings.maxMessageCount) 
          : updatedMessages;
      });
    } finally {
      setLoading(false);
      setIsThinking(false);
      scrollToBottom();
    }
  };

  // Main UI Structure
  return (
    <div className="fixed inset-0 overflow-hidden">
      <div className={`h-full flex ${isDark ? 'bg-[#1a1b1e]' : 'bg-gray-50'}`}>
        {/* Theme and Settings Buttons */}
        <div className="fixed top-4 right-4 z-10 flex gap-2">
          <button
            onClick={toggleTheme}
            className={`p-2 rounded-lg transition-colors ${
              isDark 
                ? 'bg-[#2c2d31] hover:bg-[#34353a] text-gray-300' 
                : 'bg-white hover:bg-gray-100 text-gray-700 shadow-sm'
            }`}
          >
            {isDark ? <Sun size={20} /> : <Moon size={20} />}
          </button>
          <button
            onClick={openSettings}  // Use this instead of directly setting isSettingsOpen
            className={`p-2 rounded-lg transition-colors ${
              isDark 
                ? 'bg-[#2c2d31] hover:bg-[#34353a] text-gray-300' 
                : 'bg-white hover:bg-gray-100 text-gray-700 shadow-sm'
            }`}
          >
            <SettingsIcon size={20} />
          </button>
        </div>
  
        {/* Left Sidebar - File Management */}
        <div className={`w-64 h-full flex flex-col border-r ${
          isDark 
            ? 'bg-[#1f2024] border-gray-800' 
            : 'bg-white border-gray-200'
        }`}>
 {/* Upload Section with Batch Toggle */}
 <div className="p-4 space-y-4">
            <BatchModeToggle />
            
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileUpload}
              className="hidden"
              accept=".pdf,.txt,.docx,.jpg,.jpeg,.png,.gif,.bmp,.webp,.svg,.csv,.xls,.xlsx,.xml"
              multiple
            />
            
            <button
              onClick={() => fileInputRef.current?.click()}
              className={`w-full flex items-center justify-center gap-2 p-3 rounded-lg transition-colors ${
                isDark 
                  ? 'bg-[#2c2d31] hover:bg-[#34353a] text-gray-300' 
                  : 'bg-white hover:bg-gray-100 text-gray-700 border border-gray-200'
              }`}
            >
              <Upload size={20} />
              Choose Files
            </button>

            {/* Batch Progress */}
            {batchStatus && <BatchProgress status={batchStatus} />}

            {uploadStatus && (
              <p className={`text-sm mt-2 ${
                uploadStatus.includes('success') 
                  ? 'text-green-400' 
                  : uploadStatus.includes('Processing') 
                    ? 'text-blue-400' 
                    : 'text-red-400'
              }`}>
                {uploadStatus}
              </p>
            )}
          </div>
  
          {/* File List */}
          <div className="flex-1 overflow-y-auto p-4 space-y-2">
            {files.map((file, index) => (
              <FileCard
                key={index}
                file={file}
                onRemove={() => removeFile(file)}
                onSelect={() => setSelectedFile(file)}
                isSelected={selectedFile === file}
                isDark={isDark}
              />
            ))}
          </div>
  
          {/* File Preview */}
          {selectedFile && (
            <div 
              key={`${selectedFile.name}-${previewKey}`}
              className="h-1/2 border-t border-gray-800"
            >
              <div className="h-full p-4">
                <FilePreview file={selectedFile} />
              </div>
            </div>
          )}
        </div>
  
        {/* Main Chat Area */}
        <div className="flex-1 flex flex-col overflow-hidden">
          {/* Messages */}
          <div className={`flex-1 overflow-y-auto p-4 ${
            isDark ? 'bg-[#1a1b1e]' : 'bg-gray-50'
          }`}>
            <div className="max-w-3xl mx-auto space-y-4">
              {messages.map((message, index) => (
                <Message 
                  key={index} 
                  message={message} 
                  isThinking={isThinking && index === messages.length - 1} 
                />
              ))}
              <div ref={messagesEndRef} />
            </div>
          </div>
          
          {/* Input Area and Model Selection */}
          <div className={`border-t p-4 ${
            isDark 
              ? 'bg-[#1f2024] border-gray-800' 
              : 'bg-white border-gray-200'
          }`}>
            <div className="max-w-3xl mx-auto">
              {/* Model Selection and Search Toggle */}
              <div className="flex gap-2 mb-4 justify-between items-center">
                <select
                  value={selectedModel}
                  onChange={(e) => {
                    setSelectedModel(e.target.value);
                    localStorage.setItem('selectedModel', e.target.value);
                  }}
                  className={`p-2 rounded-lg border focus:outline-none focus:border-blue-500 ${
                    isDark 
                      ? 'bg-[#2c2d31] text-gray-200 border-gray-700' 
                      : 'bg-white text-gray-700 border-gray-200'
                  }`}
                >
                  {availableModels.map((model) => (
                    <option key={model.name} value={model.name}>
                      {model.name}
                    </option>
                  ))}
                </select>
  
                {/* Search Mode Toggle */}
                <div className={`flex items-center gap-2 px-3 py-2 rounded-lg border ${
                  isDark ? 'border-gray-700' : 'border-gray-200'
                }`}>
                  <span className={`text-sm ${isDark ? 'text-gray-300' : 'text-gray-700'}`}>
                    Search Mode:
                  </span>
                  <button
                    onClick={() => setIsEnhancedSearch(!isEnhancedSearch)}
                    className={`px-3 py-1 rounded-lg text-sm transition-colors ${
                      isEnhancedSearch
                        ? 'bg-blue-600 text-white'
                        : isDark
                          ? 'bg-[#2c2d31] text-gray-400'
                          : 'bg-gray-100 text-gray-600'
                    }`}
                  >
                    {isEnhancedSearch ? 'Enhanced' : 'Basic'}
                  </button>
                </div>
              </div>
  
              {/* Chat Input */}
              <form onSubmit={handleSubmit} className="flex gap-2">
                <input
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder={isEnhancedSearch ? "Ask a detailed question..." : "What's up?"}
                  className={`flex-1 p-3 rounded-lg border focus:outline-none focus:border-blue-500 ${
                    isDark 
                      ? 'bg-[#2c2d31] text-gray-200 placeholder-gray-500 border-gray-700' 
                      : 'bg-white text-gray-700 placeholder-gray-400 border-gray-200'
                  }`}
                />
                <button
                  type="submit"
                  disabled={loading}
                  className="bg-[#2563eb] text-white px-4 py-2 rounded-lg hover:bg-blue-600 disabled:bg-gray-700 disabled:cursor-not-allowed transition-colors"
                >
                  <Send size={20} />
                </button>
              </form>
            </div>
          </div>
        </div>
  
        {/* Settings Modal */}
        <Settings
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        onSave={handleSettingsSave}
        hasContext={hasContext}
        onClearContext={clearContext}
      />
      </div>
    </div>
  );
};

export default App;