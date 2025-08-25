# AI-Personal-Debt-Finance-Advisor
A comprehensive AI-powered financial application built with Streamlit that helps users manage debt, create repayment strategies, and improve their credit scores through intelligent analysis and personalized recommendations.
✨ Features
🎯 Core Functionality

💬 Conversational AI Advisor: Chat with an AI financial advisor using natural language
📊 Advanced Repayment Planning: Generate optimized debt payoff strategies (Avalanche, Snowball, Mathematical Optimal)
🔮 What-If Scenario Analysis: Explore the impact of extra payments, windfalls, and strategy changes
📚 Educational Hub: AI-powered explanations of financial concepts with RAG-based knowledge base
📄 Document Analyzer: Upload and analyze financial documents (PDFs, bank statements, etc.)
⭐ Credit Score Improvement: Personalized credit score improvement recommendations

🚀 Advanced Features

Slash Commands: Quick financial calculations with commands like /plan, /whatif, /rag
Interactive Data Editing: Professional debt management interface with CSV import/export
Visual Analytics: Charts and graphs for debt trajectory, payment breakdown, and progress tracking
Real-time Calculations: Live updates as you modify debts and parameters
Export Capabilities: Download reports, schedules, and analysis results

🛠️ Technology Stack

Frontend: Streamlit with responsive UI components
AI/LLM: LangChain + Groq API (Llama 3.3 70B)
Data Processing: Pandas, NumPy for financial calculations
Visualization: Matplotlib for charts and graphs
Vector Store: RAG implementation for educational content
File Processing: Support for PDF, CSV, Excel, and text documents

📋 Prerequisites

Python 3.8+
Groq API key (free tier available)
Git (for cloning)

🚀 Quick Start
1. Clone the Repository
bashgit clone https://github.com/yourusername/ai-debt-finance-advisor.git
cd ai-debt-finance-advisor
2. Install Dependencies
bashpip install -r requirements.txt
3. Set Up Environment Variables
Create a .env file in the root directory:
envGROQ_API_KEY=your_groq_api_key_here
LLM_MODEL=llama-3.3-70b-versatile
DEBUG_MODE=false
4. Run the Application
bashstreamlit run app.py
The application will open in your browser at http://localhost:8501
📁 Project Structure

ai-debt-finance-advisor/
├── app.py                 # Main Streamlit application

├── requirements.txt       # Python dependencies

├── .env.example          # Environment variables template

├── .gitignore           # Git ignore rules

├── README.md            # This file

├── LICENSE              # MIT License

├── core/                # Core application modules

│   ├── __init__.py

│   ├── schemas.py       # Data models (Debt, UserProfile, etc.)

│   ├── optimization.py  # Debt repayment algorithms

│   ├── scenarios.py     # What-if analysis functions

│   ├── education.py     # Educational content and RAG

│   ├── docsum.py        # Document analysis

│   ├── prompts.py       # AI prompts and templates

│   ├── utils.py         # Utility functions

│   ├── memory.py        # Chat history management

│   ├── plan_utils.py    # Plan visualization utilities

│   ├── chat_tools.py    # Slash command processors

│   └── recommendations.py # Personalized recommendations

├── data/                # Knowledge base and sample data

│   ├── financial_kb/    # Educational content

│   └── sample_data/     # Example datasets

├── docs/                # Additional documentation

│   ├── API.md          # API documentation

│   ├── DEPLOYMENT.md   # Deployment guide

│   └── CONTRIBUTING.md # Contribution guidelines

└── tests/               # Unit tests
    ├── test_optimization.py
    
    ├── test_scenarios.py
    
    └── test_utils.py

💡 Usage Examples
Basic Debt Entry

Navigate to the "💬 Chat Advisor" tab
Use the debt editor to input your debts
Set your monthly budget in the sidebar
Ask the AI advisor: "What's the best strategy for my situation?"

Quick Commands

/plan strategy=avalanche budget=15000 - Generate avalanche repayment plan
/whatif extra=3000 - Analyze impact of extra ₹3,000/month
/rag question="explain debt consolidation" - Get educational content

Advanced Analysis

Go to "📊 Repayment Plans" for detailed strategy comparison
Use "🔮 What-If Scenarios" to model different situations
Upload documents in "📄 Document Analyzer" for AI analysis

⚙️ Configuration
Environment Variables

GROQ_API_KEY: Your Groq API key for AI functionality
LLM_MODEL: Model name (default: llama-3.3-70b-versatile)
DEBUG_MODE: Enable debug panel (true/false)

Customization

Modify core/prompts.py to customize AI behavior
Add educational content to data/financial_kb/
Adjust debt algorithms in core/optimization.py

🧪 Testing
Run the test suite:
bashpython -m pytest tests/
Run specific tests:
bashpython -m pytest tests/test_optimization.py -v
📊 Features Deep Dive
Debt Repayment Strategies

Debt Avalanche: Pay highest interest rate debts first (mathematically optimal)
Debt Snowball: Pay smallest balances first (psychological motivation)
Mathematical Optimal: Linear programming for absolute optimization

AI Capabilities

Natural language financial advice
Document analysis and insights
Personalized recommendations based on user profile
Educational content with retrieval-augmented generation (RAG)

Visualization & Analytics

Balance trajectory over time
Payment allocation charts
Interest vs principal breakdown
Credit utilization tracking

🔒 Privacy & Security

Local Processing: All calculations performed locally
No Data Storage: Session data not permanently stored
Secure API Usage: API keys managed through environment variables
Privacy First: No personal financial data transmitted beyond necessary API calls
