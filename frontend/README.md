# Local Enhanced RAG Application - Frontend

This repository contains the React-based frontend for an Enhanced Retrieval-Augmented Generation (RAG) application, designed to work with the RAG backend API.

## Features

- **Multi-format document support**: Preview PDF, DOCX, CSV, Excel, XML, and images
- **Intelligent chat interface**: Context-aware conversation
- **Enhanced search modes**: Choose between basic and enhanced search
- **Batch processing UI**: Process multiple files efficiently
- **Dark/Light mode**: User-friendly theme switching
- **Settings management**: Configure API and model parameters
- **File management sidebar**: Upload, preview, and manage documents
- **Interactive previews**: Preview documents with toggleable visibility
- **Progress tracking**: Track batch processing operations
- **Context management**: Clear and manage document context

## Project Structure

```
src/
├── App.js                # Main application component
├── Settings.js           # Settings modal component
├── index.js              # Application entry point
├── contexts/             # React contexts
│   └── ThemeContext.js   # Theme management context
├── components/           # UI components
│   ├── FileCard.js       # File representation component
│   ├── Message.js        # Chat message component
│   └── ...               # Other components
├── utils/                # Utility functions
└── styles/               # CSS styles
```

## Technology Stack

- **React**: Frontend framework
- **Tailwind CSS**: Styling
- **Lucide React**: Icons
- **PapaParse**: CSV parsing
- **SheetJS**: Excel file handling
- **fast-xml-parser**: XML parsing
- **Local Storage API**: Settings persistence

## Setup and Installation

1. Clone the repository
   ```bash
   git clone https://github.com/yourusername/localrag.git
   cd localrag
   ```

2. Install dependencies
   ```bash
   npm install
   ```

3. Configure the backend API endpoint
   - Open `src/App.js`
   - Set the backend URL (defaults to `http://localhost:8000`)

4. Start the development server
   ```bash
   npm start
   ```

5. The application will be available at `http://localhost:3000`

## Usage Guide

### Main Interface

The interface consists of three main areas:
1. **Left Sidebar**: File management
2. **Main Area**: Chat interface
3. **Bottom Bar**: Input and settings

### File Management

- **Upload Files**: Click "Choose Files" to upload documents
- **File Preview**: Click on a file to preview its content
- **Remove File**: Click the X button to remove a file
- **Toggle Preview**: Use the eye icon to show/hide previews

### Chat Interface

- **Send Message**: Type in the input field and press Enter or click Send
- **View Responses**: AI responses appear in the chat
- **Enhanced Search**: Toggle between Basic and Enhanced search modes
- **Model Selection**: Choose different LLM models from the dropdown

### Batch Processing

- **Toggle Batch Mode**: Switch between Standard and Batch processing
- **Submit Batch**: Upload files for batch processing
- **Track Progress**: View progress indicators for batch operations
- **Cancel Batch**: Cancel ongoing batch operations

### Settings

Access settings by clicking the gear icon:

- **Ollama API Endpoint**: Configure the Ollama API URL
- **Model Parameters**: Adjust temperature, context window, etc.
- **Context Management**: Clear loaded documents
- **Available Models**: View models available through Ollama

## Supported File Types

| File Type | Extensions | Features |
|-----------|------------|----------|
| PDF | .pdf | Full preview, page navigation |
| Word | .docx | Text preview |
| Plain Text | .txt | Full text preview |
| Images | .jpg, .jpeg, .png, .gif, .bmp, .webp, .svg | Full preview |
| Spreadsheets | .csv, .xlsx, .xls | Table preview, sheet navigation |
| XML | .xml | Structure view, raw XML viewing |

## Key Components

### 1. File Preview Components

The application includes specialized preview components for different file types:

- `PDFPreview`: Shows PDF documents with native browser rendering
- `WordPreview`: Extracts and displays text from DOCX files
- `TextPreview`: Displays plain text files
- `ImagePreview`: Shows images with responsive scaling
- `SpreadsheetPreview`: Displays CSV/Excel files in tabular format
- `XMLPreview`: Shows formatted XML structure and raw content

### 2. Messaging System

The messaging system handles:
- User messages
- AI responses
- Thinking indicators
- Enhanced search results formatting
- Error handling

### 3. Settings Management

The Settings component allows configuration of:
- API endpoints
- Model parameters
- Context window size
- Response temperature
- Top-P sampling

### 4. Batch Processing UI

The batch processing interface provides:
- Batch job submission
- Progress tracking
- Job cancellation
- Status updates

## Customization

### Theme Customization

The application uses Tailwind CSS with a customizable dark/light theme system:

```jsx
// Example of customizing theme colors
const bgColors = isDark ? {
  user: 'bg-[#2563eb]',
  assistant: 'bg-[#1f2024]',
  // Add custom colors here
} : {
  user: 'bg-blue-500',
  assistant: 'bg-white',
  // Add custom colors here
};
```

### API Configuration

Modify the API endpoint in the Settings component:

```jsx
const [settings, setSettings] = useState({
  ollamaAPI: localStorage.getItem('ollamaAPI') || 'http://your-api-url:11434',
  // Other settings
});
```

## License

[MIT License](LICENSE)