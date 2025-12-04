# 🧮 Adaptive Math Practice App

AI-powered adaptive math word problems that adjust to your skill level in real-time.

## Features

- ✨ Adaptive difficulty adjustment based on performance
- 📊 Real-time feedback and progress tracking
- 🎯 Step-by-step solutions with rationale
- 📈 Performance analytics and session summaries
- 🎨 Beautiful, kid-friendly interface

## Quick Start

### Option 1: Use Deployed App

Visit the live app: **mathbuddies.streamlit.app**

### Option 2: Run Locally

1. **Clone the repository**
```bash
   git clone https://github.com/hyungjinkim8/MathBuddies.git
   cd math-practice-app
```

2. **Install dependencies**
```bash
   pip install -r requirements.txt
```

3. **Set up secrets**
```bash
   cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```
   Edit `.streamlit/secrets.toml` and add your OpenAI API key.

4. **Run the app**
```bash
   streamlit run app.py
```

5. **Open your browser**
   Navigate to `http://localhost:8501`

## Deployment to Streamlit Cloud

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click "New app"
5. Select your repository and branch
6. Set main file path to `app.py`
7. Add secrets in app settings (copy from secrets.toml.example)
8. Click "Deploy"

## Project Structure
```
math-practice-app/
├── .streamlit/
│   ├── config.toml           # Streamlit configuration
│   └── secrets.toml.example  # Secrets template
├── data/
│   ├── example_dat.json      # Example problems
│   ├── standards_mapping.json # Educational standards
│   └── data_mapped.json      # Problem mapping
├── app.py                    # Main Streamlit application
├── mwp_classes.py           # Core logic classes
├── requirements.txt         # Python dependencies
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

## Configuration

### Secrets (`.streamlit/secrets.toml`)
```toml
OPENAI_API_KEY = "your-openai-api-key"
```

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (required)

## Development

## License

MIT License - feel free to use and modify!

## Support

For issues and questions, please open an issue on GitHub.