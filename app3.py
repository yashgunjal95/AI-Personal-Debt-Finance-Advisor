import os
import time
import json
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Import core modules
from core.schemas import Debt, UserProfile, RepaymentPlan
from core.optimization import (
    compute_avalanche_plan,
    compute_snowball_plan,
    one_step_optimal_allocation,
)
from core.scenarios import simulate_payoff
from core.education import get_or_build_vectorstore, rag_answer
from core.docsum import summarize_docs
from core.prompts import SYSTEM_PROMPT_ADVISOR
from core.utils import money
from core.memory import get_chat_history, add_chat_message, reset_chat_history
from core.plan_utils import plan_to_dataframe, simulate_total_balance_series

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# =========================
# App + LLM Setup
# =========================

load_dotenv()
st.set_page_config(
    page_title="AI Finance Advisor", 
    page_icon="💸", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

def get_llm(model=None):
    """Initialize and return LLM with proper error handling"""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY missing in .env file. Please add your API key.")
        st.stop()
    
    try:
        return ChatGroq(
            model=model or os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
            groq_api_key=api_key,
            temperature=0.2,
        )
    except Exception as e:
        st.error(f"Failed to initialize LLM: {e}")
        st.stop()

def tab_key(name: str) -> str:
    """Generate namespaced key for session state"""
    return f"tab::{name}"

# Initialize session
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(time.time())

# Default debts for demo
DEFAULT_DEBTS = [
    {"name": "Credit Card A", "balance": 120000, "apr": 0.36, "min_payment": 3000},
    {"name": "Student Loan", "balance": 240000, "apr": 0.12, "min_payment": 2500},
    {"name": "Personal Loan", "balance": 80000, "apr": 0.18, "min_payment": 2000},
]

if "debts_json_shared" not in st.session_state:
    st.session_state["debts_json_shared"] = json.dumps(DEFAULT_DEBTS, indent=2)

# =========================
# Helper Functions
# =========================

def parse_debts_json(text: str) -> Tuple[List[Debt], Optional[str]]:
    """Parse and validate debts JSON input"""
    try:
        data = json.loads(text)
        if not isinstance(data, list):
            return [], "Debts JSON must be a list of objects."
        
        debts = []
        for i, d in enumerate(data):
            try:
                # Validate required fields
                if not all(k in d for k in ["name", "balance", "apr", "min_payment"]):
                    return [], f"Debt {i+1} missing required fields (name, balance, apr, min_payment)"
                
                # Convert to proper types
                debt = Debt(
                    name=str(d["name"]).strip(),
                    balance=float(d["balance"]),
                    apr=float(d["apr"]),
                    min_payment=float(d["min_payment"]),
                    limit=float(d.get("limit", 0)) if d.get("limit") else None
                )
                debts.append(debt)
            except (ValueError, TypeError) as e:
                return [], f"Invalid data in debt {i+1}: {e}"
        
        return debts, None
    except json.JSONDecodeError as e:
        return [], f"Invalid JSON format: {e}"
    except Exception as e:
        return [], f"Error parsing debts: {e}"

def pretty_debts_table(debts: List[Debt]) -> pd.DataFrame:
    """Create formatted DataFrame for debt display"""
    if not debts:
        return pd.DataFrame()
    
    rows = []
    for d in debts:
        rows.append({
            "Debt Name": d.name,
            "Balance (₹)": f"{d.balance:,.0f}",
            "APR (%)": f"{d.apr * 100:.2f}%",
            "Min Payment (₹)": f"{d.min_payment:,.0f}",
            "Monthly Interest (₹)": f"{d.balance * d.apr / 12:.0f}",
        })
    return pd.DataFrame(rows)

def summarize_debts(debts: List[Debt], profile: Optional[UserProfile] = None) -> str:
    """Generate debt summary text"""
    if not debts:
        return "No debts provided."
    
    total_balance = sum(d.balance for d in debts)
    total_min = sum(d.min_payment for d in debts if d.balance > 0)
    
    # Calculate weighted average APR
    weighted_apr = 0.0
    if total_balance > 0:
        weighted_apr = sum(d.apr * d.balance for d in debts) / total_balance
    
    summary = f"""
    **Debt Summary:**
    - Total Balance: {money(total_balance)}
    - Weighted Average APR: {weighted_apr*100:.2f}%
    - Total Minimum Payments: {money(total_min)}/month
    """
    
    if profile:
        available_budget = profile.monthly_income - profile.monthly_expenses
        leftover = available_budget - total_min
        summary += f"\n    - Available Budget: {money(available_budget)}/month"
        summary += f"\n    - Leftover after minimums: {money(leftover)}/month"
        
        if leftover < 0:
            summary += "\n    ⚠️ **Warning: Budget deficit detected!**"
    
    return summary

# =========================
# Streamlit App Layout
# =========================

st.title("💸 AI-Based Personal Debt Finance Advisor")
st.caption("Built with Streamlit + LangChain + Groq - Your Personal Financial Assistant")

# Sidebar for global settings
with st.sidebar:
    st.header("Settings")
    
    # User Profile
    st.subheader("Your Financial Profile")
    monthly_income = st.number_input("Monthly Income (₹)", min_value=0.0, value=50000.0, step=1000.0)
    monthly_expenses = st.number_input("Monthly Expenses (₹)", min_value=0.0, value=30000.0, step=500.0)
    extra_budget = st.number_input("Extra Budget for Debt (₹)", min_value=0.0, value=5000.0, step=500.0)
    
    if st.button("💾 Save Profile", type="primary"):
        st.session_state["user_profile"] = UserProfile(
            monthly_income=monthly_income,
            monthly_expenses=monthly_expenses,
            extra_payment=extra_budget
        )
        st.success("✅ Profile saved!")
    
    # Initialize profile if not exists
    if "user_profile" not in st.session_state:
        st.session_state["user_profile"] = UserProfile(
            monthly_income=monthly_income,
            monthly_expenses=monthly_expenses,
            extra_payment=extra_budget
        )

# Main tabs
TABS = [
    "💬 Chat Advisor",
    "📊 Repayment Plans", 
    "🔮 What-If Scenarios",
    "📚 Educational Hub",
    "📄 Document Analyzer",
    "⭐ Credit Score Tips"
]

tabs = st.tabs(TABS)

# =========================
# 1) Chat Advisor Tab
# =========================

with tabs[0]:
    st.header("💬 Conversational Financial Advisor")
    st.info("💡 **Pro Tips:** Use slash commands like `/plan strategy=avalanche budget=15000` or `/whatif extra=3000`")
    
    # Import chat tools
    from core.chat_tools import parse_slash_command, run_plan_tool, compare_baseline_vs_extra, run_rag_tool
    
    # Chat controls
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("🗑️ Clear Chat", type="secondary"):
            reset_chat_history(tab_key("chat"))
            st.rerun()
    
    with col2:
        if st.button("📥 Export Chat"):
            history = get_chat_history(tab_key("chat"))
            if history:
                md_content = ["# Chat Transcript\n"]
                for msg in history:
                    role = "**User**" if msg["role"] == "user" else "**Assistant**"
                    md_content.append(f"{role}: {msg['content']}\n\n")
                
                st.download_button(
                    "⬇️ Download chat.md",
                    "\n".join(md_content),
                    file_name=f"chat_{int(time.time())}.md",
                    mime="text/markdown"
                )
            else:
                st.warning("No chat history to export")
    
    with col3:
        show_context = st.toggle("Show Context Panel", value=True)
    
    # Context Panel
    if show_context:
        with st.expander("💳 Your Current Debts", expanded=True):
            # Parse existing debts
            debts, error = parse_debts_json(st.session_state.get("debts_json_shared", json.dumps(DEFAULT_DEBTS, indent=2)))
            
            if error:
                st.error(f"❌ {error}")
                debts = []
            
            # Professional debt management interface
            st.markdown("**Manage Your Debts:**")
            
            # Two-column layout for better organization
            debt_col1, debt_col2 = st.columns([2, 1])
            
            with debt_col1:
                # Interactive debt editor using data_editor
                if debts:
                    # Convert debts to DataFrame for editing
                    debt_data = []
                    for d in debts:
                        debt_data.append({
                            "Debt Name": d.name,
                            "Balance": d.balance,
                            "APR (%)": d.apr * 100,
                            "Min Payment": d.min_payment,
                            "Credit Limit": d.limit or 0
                        })
                else:
                    # Default data if no debts
                    debt_data = [
                        {"Debt Name": "Credit Card A", "Balance": 120000, "APR (%)": 36, "Min Payment": 3000, "Credit Limit": 200000},
                        {"Debt Name": "Student Loan", "Balance": 240000, "APR (%)": 12, "Min Payment": 2500, "Credit Limit": 0},
                        {"Debt Name": "Personal Loan", "Balance": 80000, "APR (%)": 18, "Min Payment": 2000, "Credit Limit": 0},
                    ]
                
                # Professional debt editor
                edited_debts = st.data_editor(
                    pd.DataFrame(debt_data),
                    num_rows="dynamic",
                    use_container_width=True,
                    key="chat_debt_editor",
                    column_config={
                        "Debt Name": st.column_config.TextColumn(
                            "Debt Name",
                            help="Name of your debt (e.g., 'Chase Credit Card', 'Car Loan')",
                            required=True
                        ),
                        "Balance": st.column_config.NumberColumn(
                            "Current Balance (₹)",
                            help="Outstanding balance amount",
                            min_value=0,
                            format="₹%.0f",
                            required=True
                        ),
                        "APR (%)": st.column_config.NumberColumn(
                            "Interest Rate (%)",
                            help="Annual Percentage Rate",
                            min_value=0.0,
                            max_value=100.0,
                            format="%.2f%%",
                            required=True
                        ),
                        "Min Payment": st.column_config.NumberColumn(
                            "Min Payment (₹)",
                            help="Minimum monthly payment required",
                            min_value=0,
                            format="₹%.0f",
                            required=True
                        ),
                        "Credit Limit": st.column_config.NumberColumn(
                            "Credit Limit (₹)",
                            help="Total credit limit (for credit cards only)",
                            min_value=0,
                            format="₹%.0f"
                        )
                    },
                    hide_index=True
                )
                
                # Update button
                if st.button("💾 Update Debts", type="primary", use_container_width=True):
                    try:
                        # Convert DataFrame back to JSON format for storage
                        updated_debts_json = []
                        for _, row in edited_debts.iterrows():
                            if pd.notna(row["Debt Name"]) and str(row["Debt Name"]).strip():
                                debt_dict = {
                                    "name": str(row["Debt Name"]).strip(),
                                    "balance": float(row["Balance"]),
                                    "apr": float(row["APR (%)"]) / 100,
                                    "min_payment": float(row["Min Payment"])
                                }
                                if row.get("Credit Limit", 0) > 0:
                                    debt_dict["limit"] = float(row["Credit Limit"])
                                
                                updated_debts_json.append(debt_dict)
                        
                        # Update session state
                        st.session_state["debts_json_shared"] = json.dumps(updated_debts_json, indent=2)
                        st.success("✅ Debts updated successfully!")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error updating debts: {e}")
            
            with debt_col2:
                # Quick actions and summary
                st.markdown("**Quick Actions:**")
                
                # Add common debt types quickly
                if st.button("➕ Add Credit Card", use_container_width=True):
                    new_debt = {
                        "name": "New Credit Card",
                        "balance": 50000,
                        "apr": 0.24,
                        "min_payment": 1500,
                        "limit": 100000
                    }
                    current_debts = json.loads(st.session_state.get("debts_json_shared", "[]"))
                    current_debts.append(new_debt)
                    st.session_state["debts_json_shared"] = json.dumps(current_debts, indent=2)
                    st.rerun()
                
                if st.button("🏠 Add Loan", use_container_width=True):
                    new_debt = {
                        "name": "New Loan",
                        "balance": 100000,
                        "apr": 0.12,
                        "min_payment": 2000
                    }
                    current_debts = json.loads(st.session_state.get("debts_json_shared", "[]"))
                    current_debts.append(new_debt)
                    st.session_state["debts_json_shared"] = json.dumps(current_debts, indent=2)
                    st.rerun()
                
                if st.button("🗑️ Clear All", type="secondary", use_container_width=True):
                    st.session_state["debts_json_shared"] = "[]"
                    st.rerun()
                
                # Advanced options
                with st.expander("⚙️ Advanced Options"):
                    st.markdown("**Import/Export:**")
                    
                    # Export current debts
                    if debts:
                        debt_csv_data = pd.DataFrame([{
                            "Name": d.name,
                            "Balance": d.balance,
                            "APR": d.apr,
                            "Min_Payment": d.min_payment,
                            "Limit": d.limit or 0
                        } for d in debts]).to_csv(index=False)
                        
                        st.download_button(
                            "📥 Export CSV",
                            debt_csv_data,
                            file_name="my_debts.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    # Import from CSV
                    uploaded_csv = st.file_uploader(
                        "📤 Import from CSV",
                        type=['csv'],
                        help="Upload a CSV with columns: Name, Balance, APR, Min_Payment, Limit"
                    )
                    
                    if uploaded_csv:
                        try:
                            import_df = pd.read_csv(uploaded_csv)
                            imported_debts = []
                            
                            for _, row in import_df.iterrows():
                                debt_dict = {
                                    "name": str(row.get("Name", "Imported Debt")),
                                    "balance": float(row.get("Balance", 0)),
                                    "apr": float(row.get("APR", 0.12)),
                                    "min_payment": float(row.get("Min_Payment", 0))
                                }
                                if row.get("Limit", 0) > 0:
                                    debt_dict["limit"] = float(row["Limit"])
                                
                                imported_debts.append(debt_dict)
                            
                            st.session_state["debts_json_shared"] = json.dumps(imported_debts, indent=2)
                            st.success(f"✅ Imported {len(imported_debts)} debts!")
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"Import failed: {e}")

            # Professional debt summary
            if debts:
                st.markdown("---")
                st.markdown("**📊 Debt Summary:**")
                
                # Summary metrics in cards
                summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
                
                total_balance = sum(d.balance for d in debts)
                total_min = sum(d.min_payment for d in debts if d.balance > 0)
                weighted_apr = sum(d.apr * d.balance for d in debts) / total_balance if total_balance > 0 else 0
                num_debts = len([d for d in debts if d.balance > 0])
                
                with summary_col1:
                    st.metric("💰 Total Balance", money(total_balance))
                
                with summary_col2:
                    st.metric("📅 Total Min Payments", f"{money(total_min)}/mo")
                
                with summary_col3:
                    st.metric("📈 Avg APR", f"{weighted_apr*100:.2f}%")
                
                with summary_col4:
                    st.metric("📋 Active Debts", str(num_debts))
                
                # Budget analysis
                profile = st.session_state.get("user_profile")
                if profile:
                    available_budget = profile.monthly_income - profile.monthly_expenses
                    leftover = available_budget - total_min
                    
                    budget_col1, budget_col2 = st.columns(2)
                    
                    with budget_col1:
                        if leftover >= 0:
                            st.success(f"✅ **Budget Status:** ₹{leftover:,.0f} available for extra payments")
                        else:
                            st.error(f"⚠️ **Budget Deficit:** ₹{abs(leftover):,.0f} short each month")
                    
                    with budget_col2:
                        debt_to_income = (total_min / profile.monthly_income * 100) if profile.monthly_income > 0 else 0
                        if debt_to_income > 40:
                            st.warning(f"📊 **Debt-to-Income:** {debt_to_income:.1f}% (High)")
                        elif debt_to_income > 20:
                            st.info(f"📊 **Debt-to-Income:** {debt_to_income:.1f}% (Moderate)")
                        else:
                            st.success(f"📊 **Debt-to-Income:** {debt_to_income:.1f}% (Good)")
                
                # Visual debt breakdown
                if len(debts) > 1:
                    st.markdown("**💼 Debt Breakdown:**")
                    
                    # Create a simple bar chart for debt balances
                    debt_chart_data = pd.DataFrame([
                        {"Debt": d.name, "Balance": d.balance, "APR": f"{d.apr*100:.1f}%"}
                        for d in debts if d.balance > 0
                    ])
                    
                    if not debt_chart_data.empty:
                        # Use Streamlit's built-in bar chart for simplicity
                        st.bar_chart(debt_chart_data.set_index("Debt")["Balance"])
            else:
                st.info("💡 **Getting Started:** Add your debts above to get personalized financial advice!")

    # Quick action buttons (keep the existing ones but make them more prominent)
    st.markdown("---")
    st.markdown("**🚀 Quick Actions:**")
    quick_cols = st.columns(4)

    with quick_cols[0]:
        if st.button("💡 Get Advice", use_container_width=True, help="Get personalized debt advice"):
            add_chat_message(tab_key("chat"), "user", "What's the best strategy for my debt situation?")
            st.rerun()

    with quick_cols[1]:
        if st.button("📈 Compare Strategies", use_container_width=True, help="Compare avalanche vs snowball"):
            if debts:
                budget = st.session_state.get("user_profile", UserProfile()).extra_payment + max(0, 
                    st.session_state.get("user_profile", UserProfile()).monthly_income - 
                    st.session_state.get("user_profile", UserProfile()).monthly_expenses)
                add_chat_message(tab_key("chat"), "user", f"/plan strategy=avalanche budget={budget}")
                add_chat_message(tab_key("chat"), "user", f"/plan strategy=snowball budget={budget}")
            else:
                st.warning("Please add your debts first!")
            st.rerun()

    with quick_cols[2]:
        if st.button("🔍 What-If Analysis", use_container_width=True, help="Explore extra payment scenarios"):
            if debts:
                add_chat_message(tab_key("chat"), "user", "/whatif extra=5000 strategy=avalanche")
            else:
                st.warning("Please add your debts first!")
            st.rerun()

    with quick_cols[3]:
        if st.button("📖 Learn Concepts", use_container_width=True, help="Learn about debt strategies"):
            add_chat_message(tab_key("chat"), "user", '/rag question="explain debt avalanche vs snowball"')
            st.rerun()
    
    # Chat history display
    history = get_chat_history(tab_key("chat"))
    
    # Create chat container
    chat_container = st.container()
    
    with chat_container:
        for msg in history:
            if msg["role"] == "user":
                st.chat_message("user").markdown(msg["content"])
            else:
                st.chat_message("assistant").markdown(msg["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about debt strategies, get financial advice, or use slash commands..."):
        # Add user message
        add_chat_message(tab_key("chat"), "user", prompt)
        st.chat_message("user").markdown(prompt)
        
        # Get current context
        profile = st.session_state.get("user_profile", UserProfile())
        debts, _ = parse_debts_json(st.session_state.get("debts_json_shared", "[]"))
        
        # Process slash commands
        cmd, kv = parse_slash_command(prompt)
        
        if cmd == "summary":
            response = summarize_debts(debts, profile)
            add_chat_message(tab_key("chat"), "assistant", response)
            st.chat_message("assistant").markdown(response)
            
        elif cmd == "plan":
            if not debts:
                response = "Please provide valid debts first."
                add_chat_message(tab_key("chat"), "assistant", response)
                st.chat_message("assistant").markdown(response)
            else:
                try:
                    budget = float(kv.get("budget", profile.extra_payment + max(0.0, profile.monthly_income - profile.monthly_expenses)))
                    months = int(float(kv.get("months", 60)))
                    strategy = kv.get("strategy", "avalanche")
                    
                    result = run_plan_tool(debts, budget, months, strategy=strategy)
                    
                    response = f"""
                    **{result['strategy_name']} Plan Results:**
                    - **Duration:** {result['months']} months
                    - **Total Interest:** {money(result['total_interest'])}
                    - **Monthly Budget:** {money(budget)}
                    """
                    
                    add_chat_message(tab_key("chat"), "assistant", response)
                    st.chat_message("assistant").markdown(response)
                    
                    # Display schedule if available
                    if not result["schedule_df"].empty:
                        st.dataframe(result["schedule_df"].head(10), use_container_width=True)
                        if len(result["schedule_df"]) > 10:
                            st.caption(f"Showing first 10 rows of {len(result['schedule_df'])} total months")
                            
                except Exception as e:
                    error_msg = f"Error generating plan: {e}"
                    add_chat_message(tab_key("chat"), "assistant", error_msg)
                    st.chat_message("assistant").markdown(error_msg)
        
        elif cmd == "whatif":
            if not debts:
                response = "Please provide valid debts first."
                add_chat_message(tab_key("chat"), "assistant", response)
                st.chat_message("assistant").markdown(response)
            else:
                try:
                    budget = float(kv.get("budget", profile.extra_payment + max(0.0, profile.monthly_income - profile.monthly_expenses)))
                    months = int(float(kv.get("months", 60)))
                    extra = float(kv.get("extra", 0.0))
                    strategy = kv.get("strategy", "avalanche")
                    
                    comparison = compare_baseline_vs_extra(debts, budget, months, extra=extra, strategy=strategy)
                    
                    response = f"""
                    **What-If Analysis ({strategy.title()}) with extra ₹{extra:,.0f}/month:**
                    
                    📊 **Comparison Results:**
                    - **Baseline:** {comparison['baseline']['months']} months, {money(comparison['baseline']['total_interest'])} interest
                    - **With Extra Payment:** {comparison['scenario']['months']} months, {money(comparison['scenario']['total_interest'])} interest
                    
                    💰 **Potential Savings:**
                    - **Interest Saved:** {money(comparison['interest_savings'])}
                    - **Months Saved:** {comparison['months_saved']} months
                    - **Time Reduction:** {comparison['months_saved']/12:.1f} years faster
                    """
                    
                    add_chat_message(tab_key("chat"), "assistant", response)
                    st.chat_message("assistant").markdown(response)
                    
                    # Show balance trajectory chart
                    if comparison["baseline"]["balance_series"] and comparison["scenario"]["balance_series"]:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        base_series = comparison["baseline"]["balance_series"]
                        scenario_series = comparison["scenario"]["balance_series"]
                        
                        ax.plot(range(len(base_series)), base_series, label="Baseline", linewidth=2)
                        ax.plot(range(len(scenario_series)), scenario_series, label="With Extra Payment", linewidth=2)
                        
                        ax.set_xlabel("Month")
                        ax.set_ylabel("Total Balance (₹)")
                        ax.set_title("Balance Trajectory Comparison")
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        
                        st.pyplot(fig)
                        plt.close()
                        
                except Exception as e:
                    error_msg = f"Error in what-if analysis: {e}"
                    add_chat_message(tab_key("chat"), "assistant", error_msg)
                    st.chat_message("assistant").markdown(error_msg)
        
        elif cmd == "rag":
            try:
                kb_vs = get_or_build_vectorstore()
                question = kv.get("question") or kv.get("q", "")
                
                if not question:
                    response = "Please provide a question. Use: `/rag question=\"your question here\"`"
                    add_chat_message(tab_key("chat"), "assistant", response)
                    st.chat_message("assistant").markdown(response)
                else:
                    answer, sources = run_rag_tool(question, kb_vs)
                    response = f"{answer}\n\n**Sources:** {', '.join(sources) if sources else 'Knowledge Base'}"
                    add_chat_message(tab_key("chat"), "assistant", response)
                    st.chat_message("assistant").markdown(response)
                    
            except Exception as e:
                error_msg = f"Knowledge base error: {e}"
                add_chat_message(tab_key("chat"), "assistant", error_msg)
                st.chat_message("assistant").markdown(error_msg)
        
        else:
            # Regular LLM chat
            try:
                llm = get_llm()
                
                # Build context
                context = f"""
                User Profile: 
                - Monthly Income: {money(profile.monthly_income)}
                - Monthly Expenses: {money(profile.monthly_expenses)}
                - Available for Debt: {money(profile.extra_payment)}
                
                Current Debts: {[d.model_dump() for d in debts]}
                """
                
                # Build message history
                messages = [SystemMessage(content=SYSTEM_PROMPT_ADVISOR)]
                
                # Add recent chat history (last 10 messages)
                recent_history = history[-10:] if len(history) > 10 else history
                for msg in recent_history:
                    if msg["role"] == "user":
                        messages.append(HumanMessage(content=msg["content"]))
                    else:
                        messages.append(AIMessage(content=msg["content"]))
                
                # Add current prompt with context
                messages.append(HumanMessage(content=f"Context:\n{context}\n\nUser Question: {prompt}"))
                
                # Get response with streaming
                response_container = st.chat_message("assistant")
                response_placeholder = response_container.empty()
                
                full_response = ""
                try:
                    for chunk in llm.stream(messages):
                        if chunk.content:
                            full_response += chunk.content
                            response_placeholder.markdown(full_response + "▌")
                    
                    response_placeholder.markdown(full_response)
                    
                except Exception:
                    # Fallback to non-streaming
                    response = llm.invoke(messages)
                    full_response = response.content
                    response_placeholder.markdown(full_response)
                
                add_chat_message(tab_key("chat"), "assistant", full_response)
                
            except Exception as e:
                error_msg = f"AI Assistant Error: {e}"
                st.chat_message("assistant").markdown(error_msg)
                add_chat_message(tab_key("chat"), "assistant", error_msg)

# =========================
# 2) Repayment Plans Tab  
# =========================

with tabs[1]:
    st.header("📊 Advanced Repayment Planning")
    st.markdown("Create and compare optimized debt repayment strategies with detailed analysis.")
    
    # Debt input section
    st.subheader("💳 Configure Your Debts")
    
    # Load shared debts or use defaults
    debts_for_planning, debt_error = parse_debts_json(st.session_state.get("debts_json_shared", "[]"))
    
    if debt_error:
        st.error(f"Error with shared debts: {debt_error}")
        debts_for_planning = []
    
    # Convert to editable DataFrame format
    if debts_for_planning:
        debt_data = []
        for d in debts_for_planning:
            debt_data.append({
                "Name": d.name,
                "Balance (₹)": d.balance,
                "APR (%)": d.apr * 100,
                "Min Payment (₹)": d.min_payment,
                "Credit Limit (₹)": d.limit or 0
            })
    else:
        debt_data = [
            {"Name": "Credit Card A", "Balance (₹)": 120000, "APR (%)": 36, "Min Payment (₹)": 3000, "Credit Limit (₹)": 200000},
            {"Name": "Student Loan", "Balance (₹)": 240000, "APR (%)": 12, "Min Payment (₹)": 2500, "Credit Limit (₹)": 0},
            {"Name": "Personal Loan", "Balance (₹)": 80000, "APR (%)": 18, "Min Payment (₹)": 2000, "Credit Limit (₹)": 0},
        ]
    
    # Editable debt table
    edited_debts = st.data_editor(
        pd.DataFrame(debt_data),
        num_rows="dynamic",
        use_container_width=True,
        key="debt_planner_editor",
        column_config={
            "Name": st.column_config.TextColumn("Debt Name", required=True),
            "Balance (₹)": st.column_config.NumberColumn("Balance", min_value=0, format="₹%.0f"),
            "APR (%)": st.column_config.NumberColumn("APR %", min_value=0, max_value=100, format="%.2f%%"),
            "Min Payment (₹)": st.column_config.NumberColumn("Min Payment", min_value=0, format="₹%.0f"),
            "Credit Limit (₹)": st.column_config.NumberColumn("Credit Limit", min_value=0, format="₹%.0f", help="Optional: For credit cards"),
        }
    )
    
    # Planning parameters
    st.subheader("⚙️ Planning Parameters")
    
    param_col1, param_col2, param_col3 = st.columns(3)
    
    with param_col1:
        profile = st.session_state.get("user_profile", UserProfile())
        available_budget = max(0, profile.monthly_income - profile.monthly_expenses)
        
        monthly_budget = st.number_input(
            "Monthly Debt Budget (₹)",
            min_value=0.0,
            value=min(15000.0, available_budget) if available_budget > 0 else 15000.0,
            step=500.0,
            help=f"Available from profile: ₹{available_budget:,.0f}"
        )
    
    with param_col2:
        strategy_choice = st.selectbox(
            "Repayment Strategy",
            [
                "Debt Avalanche (Highest APR First)",
                "Debt Snowball (Smallest Balance First)", 
                "Mathematical Optimal (Linear Programming)"
            ],
            help="Avalanche minimizes interest, Snowball provides psychological wins, Optimal finds mathematically best allocation"
        )
    
    with param_col3:
        max_months = st.slider(
            "Maximum Planning Horizon (months)",
            min_value=12,
            max_value=120,
            value=60,
            step=6,
            help="How many months ahead to plan"
        )
    
    # Convert edited table back to Debt objects
    def df_to_debts(df: pd.DataFrame) -> List[Debt]:
        debts = []
        for _, row in df.iterrows():
            if pd.isna(row["Name"]) or not str(row["Name"]).strip():
                continue
            
            try:
                debt = Debt(
                    name=str(row["Name"]).strip(),
                    balance=float(row["Balance (₹)"]),
                    apr=float(row["APR (%)"]) / 100.0,
                    min_payment=float(row["Min Payment (₹)"]),
                    limit=float(row["Credit Limit (₹)"]) if row.get("Credit Limit (₹)", 0) > 0 else None
                )
                debts.append(debt)
            except (ValueError, TypeError) as e:
                st.warning(f"Skipping invalid debt row: {row['Name']} - {e}")
                continue
        return debts
    
    planning_debts = df_to_debts(edited_debts)
    
    # Validation checks
    if planning_debts:
        total_minimums = sum(d.min_payment for d in planning_debts if d.balance > 0)
        
        if monthly_budget < total_minimums:
            st.error(f"⚠️ **Budget Issue:** Your budget (₹{monthly_budget:,.0f}) is less than total minimum payments (₹{total_minimums:,.0f})")
            st.info("💡 **Suggestion:** Increase your budget or consider debt restructuring options.")
        else:
            excess_budget = monthly_budget - total_minimums
            st.success(f"✅ **Budget OK:** ₹{excess_budget:,.0f} available for extra payments after minimums")
    
    # Generate plan button
    # Replace the problematic section in your Repayment Plans tab
    # Find the section after "Generate Repayment Plan" button and replace it

    # Store the plan in session state to persist across reruns
    if st.button("🚀 Generate Repayment Plan", type="primary", use_container_width=True):
        if not planning_debts:
            st.error("Please add at least one valid debt to analyze.")
        else:
            with st.spinner("Generating optimized repayment plan..."):
                try:
                    # Determine strategy
                    if strategy_choice.startswith("Debt Avalanche"):
                        plan = compute_avalanche_plan(planning_debts, monthly_budget, max_months)
                        strategy_name = "Debt Avalanche"
                    elif strategy_choice.startswith("Debt Snowball"):
                        plan = compute_snowball_plan(planning_debts, monthly_budget, max_months)
                        strategy_name = "Debt Snowball"
                    else:
                        plan = one_step_optimal_allocation(planning_debts, monthly_budget)
                        strategy_name = "Mathematical Optimal"

                    if not plan.months:
                        st.error("❌ Could not generate a valid repayment plan. Check your inputs and try again.")
                        # Clear any existing plan from session state
                        if "current_repayment_plan" in st.session_state:
                            del st.session_state["current_repayment_plan"]
                    else:
                        # Store plan and metadata in session state
                        st.session_state["current_repayment_plan"] = {
                            "plan": plan,
                            "strategy_name": strategy_name,
                            "monthly_budget": monthly_budget,
                            "planning_debts": planning_debts
                        }
                        st.rerun()  # Rerun to display the results
                        
                except Exception as e:
                    st.error(f"❌ Error generating repayment plan: {e}")
                    st.info("💡 Please check your debt information and try again.")
                    # Clear any existing plan from session state
                    if "current_repayment_plan" in st.session_state:
                        del st.session_state["current_repayment_plan"]

    # Display results if plan exists in session state
    if "current_repayment_plan" in st.session_state:
        plan_data = st.session_state["current_repayment_plan"]
        plan = plan_data["plan"]
        strategy_name = plan_data["strategy_name"]
        monthly_budget = plan_data["monthly_budget"]
        planning_debts = plan_data["planning_debts"]
        
        # Success! Display results
        st.success(f"✅ **{strategy_name} Plan Generated Successfully!**")

        # Key metrics
        total_interest = sum(m.total_interest for m in plan.months)
        total_payments = sum(m.total_paid for m in plan.months)
        principal_total = total_payments - total_interest

        # Metrics display
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

        with metric_col1:
            st.metric("🗓️ Time to Debt-Free", f"{len(plan.months)} months", f"{len(plan.months)/12:.1f} years")

        with metric_col2:
            st.metric("💰 Total Interest", money(total_interest))

        with metric_col3:
            st.metric("💳 Principal Paid", money(principal_total))

        with metric_col4:
            efficiency = (principal_total / total_payments * 100) if total_payments > 0 else 0
            st.metric("📈 Payment Efficiency", f"{efficiency:.1f}%", "principal vs total")

        # Clear plan button
        if st.button("🗑️ Clear Plan", type="secondary"):
            if "current_repayment_plan" in st.session_state:
                del st.session_state["current_repayment_plan"]
            st.rerun()

        # Detailed schedule
        st.subheader("📅 Payment Schedule")
        schedule_df = plan_to_dataframe(plan)

        if not schedule_df.empty:
            # Display options
            display_col1, display_col2 = st.columns([3, 1])

            with display_col2:
                show_all = st.checkbox("Show all months", value=False, key="show_all_months_checkbox")

            with display_col1:
                st.write(f"**Total months in plan:** {len(schedule_df)}")

            # Determine how many months to show
            months_to_show = len(schedule_df) if show_all else min(12, len(schedule_df))

            # Show schedule
            display_schedule = schedule_df.head(months_to_show).copy()

            # Format for better display
            if 'payment' in display_schedule.columns:
                display_schedule['payment'] = display_schedule['payment'].apply(lambda x: f"₹{x:,.0f}")

            if 'interest' in display_schedule.columns:
                display_schedule['interest'] = display_schedule['interest'].apply(lambda x: f"₹{x:,.0f}")

            if 'principal' in display_schedule.columns:
                display_schedule['principal'] = display_schedule['principal'].apply(lambda x: f"₹{x:,.0f}")

            st.dataframe(display_schedule, use_container_width=True)

            if not show_all and len(schedule_df) > 12:
                st.info(f"Showing first 12 months of {len(schedule_df)} total. Check 'Show all months' to see complete schedule.")

            # Download option
            csv_data = schedule_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Full Schedule (CSV)",
                csv_data,
                file_name=f"{strategy_name.lower().replace(' ', '_')}_schedule.csv",
                mime="text/csv",
                use_container_width=True
            )

            # Visualizations
            st.subheader("📈 Visual Analysis")

            viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
                "Balance Over Time", "Monthly Payments", "Interest vs Principal", "Payment Allocation"
            ])

            with viz_tab1:
                # Total balance trajectory
                balance_series = simulate_total_balance_series(planning_debts, plan)
                if balance_series:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    months_range = list(range(len(balance_series)))
                    ax.plot(months_range, balance_series, linewidth=3, color='#1f77b4', marker='o', markersize=4)
                    ax.fill_between(months_range, balance_series, alpha=0.3, color='#1f77b4')
                    ax.set_xlabel('Month')
                    ax.set_ylabel('Total Remaining Balance (₹)')
                    ax.set_title(f'{strategy_name}: Total Debt Balance Over Time')
                    ax.grid(True, alpha=0.3)
                    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'₹{x/1000:.0f}K'))
                    st.pyplot(fig)
                    plt.close()

            with viz_tab2:
                # Monthly payment breakdown
                if not schedule_df.empty:
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

                    # Total monthly payments
                    monthly_totals = schedule_df.groupby('month').agg({
                        'payment': 'sum',
                        'interest': 'sum',
                        'principal': 'sum'
                    }).reset_index()

                    ax1.plot(monthly_totals['month'], monthly_totals['payment'],
                            label='Total Payment', linewidth=2)
                    ax1.plot(monthly_totals['month'], monthly_totals['interest'],
                            label='Interest', linewidth=2)
                    ax1.plot(monthly_totals['month'], monthly_totals['principal'],
                            label='Principal', linewidth=2)
                    ax1.set_xlabel('Month')
                    ax1.set_ylabel('Amount (₹)')
                    ax1.set_title('Monthly Payment Breakdown')
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)

                    # Cumulative interest paid
                    monthly_totals['cumulative_interest'] = monthly_totals['interest'].cumsum()
                    ax2.plot(monthly_totals['month'], monthly_totals['cumulative_interest'],
                            linewidth=3, color='red')
                    ax2.set_xlabel('Month')
                    ax2.set_ylabel('Cumulative Interest (₹)')
                    ax2.set_title('Total Interest Paid Over Time')
                    ax2.grid(True, alpha=0.3)

                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

            with viz_tab3:
                # Interest vs Principal pie chart
                fig, ax = plt.subplots(figsize=(8, 8))
                labels = ['Principal Payments', 'Interest Payments']
                sizes = [principal_total, total_interest]
                colors = ['#2E8B57', '#DC143C']
                explode = (0.05, 0)

                ax.pie(sizes, explode=explode, labels=labels, colors=colors,
                    autopct='%1.1f%%', shadow=True, startangle=90)
                ax.set_title(f'Total Payment Breakdown\n({money(total_payments)} total)')
                st.pyplot(fig)
                plt.close()

            with viz_tab4:
                # Payment allocation by debt (stacked bar)
                if not schedule_df.empty and len(planning_debts) > 1:
                    pivot_data = schedule_df.pivot_table(
                        index='month',
                        columns='debt',
                        values='payment',
                        aggfunc='sum',
                        fill_value=0
                    )

                    if not pivot_data.empty:
                        fig, ax = plt.subplots(figsize=(12, 6))
                        pivot_data.plot(kind='bar', stacked=True, ax=ax)
                        ax.set_xlabel('Month')
                        ax.set_ylabel('Payment Amount (₹)')
                        ax.set_title('Monthly Payment Allocation by Debt')
                        ax.legend(title='Debts', bbox_to_anchor=(1.05, 1), loc='upper left')
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()

            # Strategy comparison
            st.subheader("🔍 Strategy Comparison")
            with st.expander("Compare All Strategies", expanded=False):
                comparison_results = {}

                # Generate all three strategies for comparison
                strategies = [
                    ("Debt Avalanche", lambda: compute_avalanche_plan(planning_debts, monthly_budget, max_months)),
                    ("Debt Snowball", lambda: compute_snowball_plan(planning_debts, monthly_budget, max_months)),
                    ("Mathematical Optimal", lambda: one_step_optimal_allocation(planning_debts, monthly_budget))
                ]

                for strategy_name_comp, strategy_func in strategies:
                    try:
                        plan_comp = strategy_func()
                        if plan_comp.months:
                            total_interest_comp = sum(m.total_interest for m in plan_comp.months)
                            comparison_results[strategy_name_comp] = {
                                "months": len(plan_comp.months),
                                "total_interest": total_interest_comp,
                                "total_payment": sum(m.total_paid for m in plan_comp.months)
                            }
                    except Exception as e:
                        st.warning(f"Could not generate {strategy_name_comp}: {e}")

                if comparison_results:
                    comparison_df = pd.DataFrame(comparison_results).T
                    comparison_df.index.name = "Strategy"
                    comparison_df["months"] = comparison_df["months"].astype(int)
                    comparison_df["total_interest"] = comparison_df["total_interest"].apply(lambda x: f"₹{x:,.0f}")
                    comparison_df["total_payment"] = comparison_df["total_payment"].apply(lambda x: f"₹{x:,.0f}")
                    comparison_df.columns = ["Months to Pay Off", "Total Interest", "Total Payments"]

                    st.dataframe(comparison_df, use_container_width=True)

            # AI Analysis
            st.subheader("🤖 AI Financial Analysis")
            try:
                llm = get_llm()
                profile = st.session_state.get("user_profile", UserProfile())

                analysis_prompt = f"""
                As a financial advisor, provide a concise analysis of this debt repayment plan:

                PLAN DETAILS:
                - Strategy: {strategy_name}
                - Time to debt-free: {len(plan.months)} months ({len(plan.months)/12:.1f} years)
                - Total interest cost: ₹{total_interest:,.0f}
                - Monthly budget: ₹{monthly_budget:,.0f}
                - Number of debts: {len(planning_debts)}

                DEBTS:
                {[f"- {d.name}: ₹{d.balance:,.0f} at {d.apr*100:.1f}% APR" for d in planning_debts]}

                Provide:
                1. Assessment of this strategy choice
                2. Key strengths and potential concerns
                3. 2-3 specific actionable recommendations
                4. What to monitor monthly

                Keep response practical and encouraging.
                """

                analysis_response = llm.invoke([HumanMessage(content=analysis_prompt)])
                st.markdown("**AI Analysis:**")
                st.write(analysis_response.content)

            except Exception as e:
                st.info(f"AI analysis unavailable: {e}")

# =========================
# 3) What-If Scenarios Tab
# =========================

with tabs[2]:
    st.header("🔮 What-If Scenario Analysis")
    st.markdown("Explore how changes in your payments, income, or debt terms affect your financial timeline.")
    
    # Scenario setup
    st.subheader("📋 Scenario Setup")
    
    scenario_col1, scenario_col2 = st.columns(2)
    
    with scenario_col1:
        st.markdown("**Base Scenario:**")
        base_budget = st.number_input("Base Monthly Budget (₹)", min_value=0.0, value=15000.0, step=500.0, key="whatif_base_budget")
        base_strategy = st.selectbox("Base Strategy", ["avalanche", "snowball", "optimal"], key="whatif_base_strategy")
        
    with scenario_col2:
        st.markdown("**What-If Scenario:**")
        scenario_type = st.selectbox(
            "Scenario Type",
            [
                "Extra Monthly Payment",
                "Windfall/Lump Sum", 
                "Budget Reduction",
                "Interest Rate Change",
                "Debt Consolidation"
            ]
        )
    
    # Scenario-specific inputs
    if scenario_type == "Extra Monthly Payment":
        extra_payment = st.number_input("Extra Monthly Payment (₹)", min_value=0.0, value=3000.0, step=500.0)
        scenario_params = {"extra_monthly": extra_payment}
        
    elif scenario_type == "Windfall/Lump Sum":
        lump_sum = st.number_input("Lump Sum Amount (₹)", min_value=0.0, value=50000.0, step=5000.0)
        apply_month = st.number_input("Apply in Month", min_value=1, value=1, step=1)
        scenario_params = {"lump_sum": lump_sum, "apply_month": apply_month}
        
    elif scenario_type == "Budget Reduction":
        budget_reduction = st.number_input("Budget Reduction (₹)", min_value=0.0, value=2000.0, step=500.0)
        scenario_params = {"budget_reduction": budget_reduction}
        
    elif scenario_type == "Interest Rate Change":
        rate_change_pct = st.number_input("APR Change (%)", value=0.0, step=0.5)
        affected_debts = st.multiselect("Affected Debts", options=["All"] + [d["name"] for d in DEFAULT_DEBTS])
        scenario_params = {"rate_change": rate_change_pct/100, "affected_debts": affected_debts}
        
    else:  # Debt Consolidation
        consolidation_rate = st.number_input("Consolidation APR (%)", min_value=0.0, value=12.0, step=0.5)
        consolidation_fee = st.number_input("One-time Fee (₹)", min_value=0.0, value=5000.0, step=1000.0)
        scenario_params = {"consolidation_rate": consolidation_rate/100, "consolidation_fee": consolidation_fee}
    
    # Analysis period
    analysis_months = st.slider("Analysis Period (months)", 12, 120, 60, step=6)
    
    # Run analysis
    if st.button("🔍 Run What-If Analysis", type="primary", use_container_width=True):
        # Get current debts
        current_debts, debt_error = parse_debts_json(st.session_state.get("debts_json_shared", "[]"))
        
        if debt_error or not current_debts:
            st.error("Please ensure valid debts are configured in the shared debt section.")
        else:
            with st.spinner("Running scenario analysis..."):
                try:
                    # Base case calculation
                    if base_strategy == "avalanche":
                        base_plan = compute_avalanche_plan(current_debts, base_budget, analysis_months)
                    elif base_strategy == "snowball":
                        base_plan = compute_snowball_plan(current_debts, base_budget, analysis_months)
                    else:
                        base_plan = one_step_optimal_allocation(current_debts, base_budget)
                    
                    base_interest = sum(m.total_interest for m in base_plan.months)
                    base_months = len(base_plan.months)
                    base_total_payment = sum(m.total_paid for m in base_plan.months)
                    
                    # Scenario case calculation
                    scenario_debts = [Debt(**d.model_dump()) for d in current_debts]  # Copy debts
                    scenario_budget = base_budget
                    
                    if scenario_type == "Extra Monthly Payment":
                        scenario_budget += scenario_params["extra_monthly"]
                        
                    elif scenario_type == "Budget Reduction":
                        scenario_budget -= scenario_params["budget_reduction"]
                        scenario_budget = max(scenario_budget, sum(d.min_payment for d in scenario_debts))
                        
                    elif scenario_type == "Interest Rate Change":
                        rate_change = scenario_params["rate_change"]
                        affected = scenario_params["affected_debts"]
                        
                        for debt in scenario_debts:
                            if "All" in affected or debt.name in affected:
                                debt.apr = max(0, debt.apr + rate_change)
                                
                    elif scenario_type == "Debt Consolidation":
                        # Combine all debts into one with new rate
                        total_balance = sum(d.balance for d in scenario_debts)
                        consolidation_balance = total_balance + scenario_params["consolidation_fee"]
                        
                        scenario_debts = [Debt(
                            name="Consolidated Loan",
                            balance=consolidation_balance,
                            apr=scenario_params["consolidation_rate"],
                            min_payment=consolidation_balance * scenario_params["consolidation_rate"] / 12 * 0.02  # Rough estimate
                        )]
                    
                    # Calculate scenario plan
                    if base_strategy == "avalanche":
                        scenario_plan = compute_avalanche_plan(scenario_debts, scenario_budget, analysis_months)
                    elif base_strategy == "snowball":
                        scenario_plan = compute_snowball_plan(scenario_debts, scenario_budget, analysis_months)
                    else:
                        scenario_plan = one_step_optimal_allocation(scenario_debts, scenario_budget)
                    
                    scenario_interest = sum(m.total_interest for m in scenario_plan.months)
                    scenario_months = len(scenario_plan.months)
                    scenario_total_payment = sum(m.total_paid for m in scenario_plan.months)
                    
                    # Display results
                    st.success("✅ Analysis Complete!")
                    
                    # Comparison metrics
                    st.subheader("📊 Scenario Comparison")
                    
                    comp_col1, comp_col2, comp_col3, comp_col4 = st.columns(4)
                    
                    with comp_col1:
                        months_saved = base_months - scenario_months
                        st.metric(
                            "Time Difference", 
                            f"{scenario_months} months",
                            f"{months_saved:+} months" if months_saved != 0 else "No change"
                        )
                    
                    with comp_col2:
                        interest_saved = base_interest - scenario_interest
                        st.metric(
                            "Interest Cost",
                            money(scenario_interest),
                            f"{money(interest_saved)} saved" if interest_saved > 0 else f"{money(abs(interest_saved))} more"
                        )
                    
                    with comp_col3:
                        payment_diff = scenario_total_payment - base_total_payment
                        if payment_diff > 0:
                            delta_text = f"+{money(payment_diff)}"
                        elif payment_diff < 0:
                            delta_text = f"-{money(abs(payment_diff))}"
                        else:
                            delta_text = "No change"
                        
                        st.metric(
                            "Total Payments",
                            money(scenario_total_payment),
                            delta_text
                        )
                    
                    with comp_col4:
                        if base_total_payment > 0:
                            efficiency_change = ((base_total_payment - scenario_total_payment) / base_total_payment) * 100
                            st.metric(
                                "Efficiency Change",
                                f"{efficiency_change:+.1f}%",
                                "vs base scenario"
                            )
                    
                    # Visual comparison
                    st.subheader("📈 Visual Comparison")
                    
                    # Balance trajectory comparison
                    base_balance_series = simulate_total_balance_series(current_debts, base_plan)
                    scenario_balance_series = simulate_total_balance_series(scenario_debts, scenario_plan)
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                    
                    # Balance over time
                    if base_balance_series and scenario_balance_series:
                        ax1.plot(range(len(base_balance_series)), base_balance_series, 
                                label="Base Scenario", linewidth=2, alpha=0.8)
                        ax1.plot(range(len(scenario_balance_series)), scenario_balance_series, 
                                label=f"What-If: {scenario_type}", linewidth=2, alpha=0.8)
                        ax1.set_xlabel("Month")
                        ax1.set_ylabel("Total Remaining Balance (₹)")
                        ax1.set_title("Balance Trajectory Comparison")
                        ax1.legend()
                        ax1.grid(True, alpha=0.3)
                    
                    # Monthly interest comparison
                    base_monthly_interest = [m.total_interest for m in base_plan.months]
                    scenario_monthly_interest = [m.total_interest for m in scenario_plan.months]
                    
                    max_months_to_show = min(len(base_monthly_interest), len(scenario_monthly_interest), 24)
                    
                    if max_months_to_show > 0:
                        months_range = list(range(1, max_months_to_show + 1))
                        ax2.plot(months_range, base_monthly_interest[:max_months_to_show], 
                                label="Base Scenario", linewidth=2, alpha=0.8)
                        ax2.plot(months_range, scenario_monthly_interest[:max_months_to_show], 
                                label=f"What-If: {scenario_type}", linewidth=2, alpha=0.8)
                        ax2.set_xlabel("Month")
                        ax2.set_ylabel("Monthly Interest (₹)")
                        ax2.set_title("Monthly Interest Comparison")
                        ax2.legend()
                        ax2.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    
                    # Summary insights
                    st.subheader("💡 Key Insights")
                    
                    insights = []
                    
                    if months_saved > 0:
                        years_saved = months_saved / 12
                        insights.append(f"✅ **Time Savings:** You could be debt-free {months_saved} months ({years_saved:.1f} years) earlier!")
                    elif months_saved < 0:
                        years_longer = abs(months_saved) / 12
                        insights.append(f"⚠️ **Time Impact:** This scenario would extend your debt payoff by {abs(months_saved)} months ({years_longer:.1f} years)")
                    
                    if interest_saved > 1000:
                        insights.append(f"💰 **Interest Savings:** You could save {money(interest_saved)} in total interest costs!")
                    elif interest_saved < -1000:
                        insights.append(f"💸 **Interest Cost:** This scenario would cost an additional {money(abs(interest_saved))} in interest")
                    
                    # Specific scenario insights
                    if scenario_type == "Extra Monthly Payment" and interest_saved > 0:
                        roi = (interest_saved / (scenario_params["extra_monthly"] * scenario_months)) * 100
                        insights.append(f"📈 **ROI:** Every extra ₹1 pays saves you ₹{interest_saved/(scenario_params['extra_monthly'] * scenario_months):.2f} in interest")
                    
                    if scenario_type == "Debt Consolidation":
                        if interest_saved > scenario_params.get("consolidation_fee", 0):
                            net_savings = interest_saved - scenario_params["consolidation_fee"]
                            insights.append(f"✅ **Consolidation Benefits:** Net savings of {money(net_savings)} after fees")
                        else:
                            insights.append("❌ **Consolidation Warning:** Fees may outweigh interest savings")
                    
                    for insight in insights:
                        st.markdown(insight)
                    
                    # Export scenario results
                    scenario_data = {
                        "Base Scenario": {
                            "Months": base_months,
                            "Total Interest": base_interest,
                            "Total Payments": base_total_payment
                        },
                        f"What-If: {scenario_type}": {
                            "Months": scenario_months,
                            "Total Interest": scenario_interest,
                            "Total Payments": scenario_total_payment
                        },
                        "Differences": {
                            "Months Saved": months_saved,
                            "Interest Saved": interest_saved,
                            "Payment Difference": payment_diff
                        }
                    }
                    
                    if st.button("📥 Export Analysis Results"):
                        import json
                        json_str = json.dumps(scenario_data, indent=2, default=str)
                        st.download_button(
                            "Download Analysis (JSON)",
                            json_str,
                            file_name=f"whatif_analysis_{int(time.time())}.json",
                            mime="application/json"
                        )
                
                except Exception as e:
                    st.error(f"❌ Analysis failed: {e}")
                    st.info("Please check your inputs and try again.")

# =========================
# 4) Educational Hub Tab
# =========================

with tabs[3]:
    st.header("📚 Financial Education Hub")
    st.markdown("Learn key financial concepts with AI-powered explanations from our knowledge base.")
    
    # Quick topic buttons
    st.subheader("🎯 Popular Topics")
    
    topic_cols = st.columns(4)
    
    with topic_cols[0]:
        if st.button("💳 Credit Utilization", use_container_width=True):
            st.session_state["education_query"] = "What is credit utilization and why is 30% the recommended target?"
    
    with topic_cols[1]:
        if st.button("⚖️ Avalanche vs Snowball", use_container_width=True):
            st.session_state["education_query"] = "Compare debt avalanche and debt snowball strategies"
    
    with topic_cols[2]:
        if st.button("🏦 Debt Consolidation", use_container_width=True):
            st.session_state["education_query"] = "When should I consider debt consolidation and what are the pros and cons?"
    
    with topic_cols[3]:
        if st.button("📈 Credit Score Basics", use_container_width=True):
            st.session_state["education_query"] = "What factors affect my credit score and how can I improve it?"
    
    # Custom query input
    st.subheader("❓ Ask Your Question")
    
    query_input = st.text_input(
        "Enter your financial question:",
        value=st.session_state.get("education_query", ""),
        placeholder="e.g., How does compound interest work in debt repayment?",
        key="education_text_input"
    )
    
    if st.button("🔍 Get Answer", type="primary") or (query_input and query_input != st.session_state.get("last_education_query", "")):
        if query_input.strip():
            st.session_state["last_education_query"] = query_input
            
            with st.spinner("🧠 Searching knowledge base and generating answer..."):
                try:
                    # Get or build knowledge base
                    kb_vs = get_or_build_vectorstore()
                    
                    # Get RAG answer
                    answer, sources = rag_answer(query_input, kb_vs, k=6)
                    
                    # Display answer
                    st.subheader("💡 Answer")
                    st.markdown(answer)
                    
                    # Show sources
                    if sources:
                        with st.expander("📖 Sources Used", expanded=False):
                            for i, source in enumerate(sources, 1):
                                st.write(f"{i}. {source}")
                    
                    # Related topics suggestion
                    st.subheader("🔗 Related Topics")
                    related_topics = [
                        "Debt-to-income ratio calculation",
                        "Emergency fund planning", 
                        "Interest rate negotiation strategies",
                        "Balance transfer considerations",
                        "Minimum payment traps"
                    ]
                    
                    related_cols = st.columns(3)
                    for i, topic in enumerate(related_topics[:3]):
                        with related_cols[i]:
                            if st.button(f"📝 {topic}", key=f"related_{i}"):
                                st.session_state["education_query"] = topic
                                st.rerun()
                
                except Exception as e:
                    st.error(f"❌ Error accessing knowledge base: {e}")
                    st.info("💡 Make sure the knowledge base is properly configured and try again.")
        else:
            st.warning("Please enter a question to get started!")
    
    # Clear query button
    if st.button("🧹 Clear", type="secondary"):
        st.session_state["education_query"] = ""
        st.rerun()
    
    # Educational resources section
    st.subheader("📋 Quick Reference")
    
    with st.expander("🎓 Key Financial Terms", expanded=False):
        terms = {
            "APR": "Annual Percentage Rate - the yearly cost of borrowing including interest and fees",
            "Principal": "The original amount of debt borrowed, excluding interest",
            "Minimum Payment": "The smallest payment required to keep an account in good standing",
            "Credit Utilization": "The percentage of available credit being used (keep below 30%)",
            "Debt-to-Income Ratio": "Monthly debt payments divided by gross monthly income",
            "Compound Interest": "Interest calculated on both principal and previously earned interest"
        }
        
        for term, definition in terms.items():
            st.write(f"**{term}**: {definition}")
    
    with st.expander("📊 Debt Payoff Strategies", expanded=False):
        st.markdown("""
        **Debt Avalanche (Mathematically Optimal):**
        - Pay minimums on all debts
        - Put extra money toward highest APR debt first
        - Saves the most money in interest
        - Best for disciplined borrowers
        
        **Debt Snowball (Psychological Wins):**
        - Pay minimums on all debts  
        - Put extra money toward smallest balance first
        - Provides quick psychological victories
        - Good for motivation and momentum
        
        **Debt Consolidation:**
        - Combine multiple debts into one loan
        - Potentially lower interest rate
        - Simplified payments
        - Consider fees and terms carefully
        """)
    
    with st.expander("⭐ Credit Score Improvement Tips", expanded=False):
        st.markdown("""
        **Payment History (35% of score):**
        - Always pay at least the minimum on time
        - Set up automatic payments
        - Catch up on past-due accounts
        
        **Credit Utilization (30% of score):**
        - Keep balances below 30% of limits
        - Even better: below 10%
        - Pay balances before statement dates
        
        **Credit History Length (15% of score):**
        - Keep old accounts open
        - Don't close your oldest credit card
        - Be patient - history builds over time
        
        **Credit Mix (10% of score):**
        - Mix of credit cards, loans, etc.
        - Don't open accounts just for mix
        
        **New Credit (10% of score):**
        - Limit hard inquiries
        - Space out new applications
        - Only apply when necessary
        """)

# =========================
# 5) Document Analyzer Tab
# =========================

with tabs[4]:
    st.header("📄 Financial Document Analyzer")
    st.markdown("Upload financial documents (PDFs, bank statements, etc.) for AI-powered analysis and insights.")
    
    # File upload section
    st.subheader("📁 Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose files to analyze",
        type=['pdf', 'txt', 'csv', 'xlsx'],
        accept_multiple_files=True,
        help="Supported formats: PDF, TXT, CSV, Excel"
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} file(s) uploaded successfully!")
        
        # Display uploaded files
        for i, file in enumerate(uploaded_files):
            file_size = len(file.getvalue()) if hasattr(file, 'getvalue') else 0
            st.write(f"{i+1}. **{file.name}** ({file_size/1024:.1f} KB)")
        
        # Analysis options
        analysis_col1, analysis_col2 = st.columns(2)
        
        with analysis_col1:
            analysis_type = st.selectbox(
                "Analysis Type",
                [
                    "General Summary",
                    "Expense Categorization",
                    "Cash Flow Analysis",
                    "Debt Account Detection",
                    "Investment Portfolio Review"
                ]
            )
        
        with analysis_col2:
            focus_areas = st.multiselect(
                "Focus Areas (optional)",
                [
                    "Monthly Spending Patterns",
                    "High-Interest Charges",
                    "Fee Analysis", 
                    "Credit Utilization",
                    "Payment History",
                    "Budget Recommendations"
                ]
            )
        
        # Analyze button
        if st.button("🔍 Analyze Documents", type="primary", use_container_width=True):
            with st.spinner("🧠 Processing documents with AI..."):
                try:
                    llm = get_llm()
                    
                    # Use the document summarizer
                    summary = summarize_docs(uploaded_files, llm)
                    
                    # Display analysis results
                    st.subheader("📊 Analysis Results")
                    
                    # Main summary
                    with st.expander("📋 Document Summary", expanded=True):
                        st.markdown(summary)
                    
                    # Enhanced analysis with LLM
                    if analysis_type != "General Summary" or focus_areas:
                        st.subheader("🎯 Focused Analysis")
                        
                        # Build analysis prompt
                        focus_prompt = f"""
                        Based on the document summary provided, perform a {analysis_type.lower()} analysis.
                        """
                        
                        if focus_areas:
                            focus_prompt += f"\nPay special attention to: {', '.join(focus_areas)}"
                        
                        focus_prompt += f"""
                        
                        Document Summary:
                        {summary}
                        
                        Please provide:
                        1. Key findings specific to {analysis_type.lower()}
                        2. Actionable recommendations
                        3. Any red flags or opportunities identified
                        4. Specific numbers and dates when available
                        
                        Keep the analysis practical and actionable for debt management.
                        """
                        
                        try:
                            analysis_response = llm.invoke([HumanMessage(content=focus_prompt)])
                            st.markdown("**AI Analysis:**")
                            st.write(analysis_response.content)
                        except Exception as e:
                            st.warning(f"Enhanced analysis unavailable: {e}")
                    
                    # Action items extraction
                    st.subheader("✅ Suggested Action Items")
                    
                    action_prompt = f"""
                    Based on this financial document analysis, suggest 3-5 specific, actionable steps the user should take:
                    
                    {summary}
                    
                    Focus on:
                    - Debt reduction opportunities
                    - Fee avoidance
                    - Credit score improvement
                    - Budget optimization
                    
                    Format as a numbered list with brief explanations.
                    """
                    
                    try:
                        action_response = llm.invoke([HumanMessage(content=action_prompt)])
                        st.write(action_response.content)
                    except Exception as e:
                        st.info(f"Action items unavailable: {e}")
                    
                    # Export analysis
                    st.subheader("💾 Export Analysis")
                    
                    full_analysis = f"""
# Financial Document Analysis Report
**Generated on:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
**Analysis Type:** {analysis_type}
**Files Analyzed:** {', '.join([f.name for f in uploaded_files])}

## Document Summary
{summary}

## Enhanced Analysis
{analysis_response.content if 'analysis_response' in locals() else 'Not available'}

## Action Items
{action_response.content if 'action_response' in locals() else 'Not available'}
                    """
                    
                    st.download_button(
                        "📥 Download Analysis Report",
                        full_analysis,
                        file_name=f"document_analysis_{int(time.time())}.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
                    
                except Exception as e:
                    st.error(f"❌ Document analysis failed: {e}")
                    st.info("Please check your files and try again.")
    
    else:
        # Show example of what can be analyzed
        st.info("📋 **What can be analyzed:**")
        
        example_col1, example_col2 = st.columns(2)
        
        with example_col1:
            st.markdown("""
            **Bank Statements:**
            - Monthly spending patterns
            - Recurring charges identification
            - Cash flow trends
            - Unusual transactions
            
            **Credit Card Statements:**
            - Interest charges breakdown
            - Fee analysis
            - Credit utilization tracking
            - Payment history patterns
            """)
        
        with example_col2:
            st.markdown("""
            **Loan Documents:**
            - Terms and rates comparison
            - Payment schedule analysis
            - Prepayment options
            - Fee structures
            
            **Investment Statements:**
            - Portfolio composition
            - Performance tracking
            - Fee analysis
            - Rebalancing suggestions
            """)

# =========================
# 6) Credit Score Tips Tab
# =========================

with tabs[5]:
    st.header("⭐ Credit Score Improvement Hub")
    st.markdown("Get personalized recommendations to improve your credit score based on your debt profile.")
    
    # Current credit score input
    st.subheader("📊 Current Credit Status")
    
    score_col1, score_col2, score_col3 = st.columns(3)
    
    with score_col1:
        current_score = st.slider(
            "Current Credit Score (if known)",
            300, 850, 650,
            help="Estimated range is fine if you don't know the exact score"
        )
    
    with score_col2:
        score_goal = st.slider(
            "Target Credit Score",
            current_score, 850, min(current_score + 100, 800)
        )
    
    with score_col3:
        last_checked = st.selectbox(
            "Last Checked",
            ["This month", "1-3 months ago", "6+ months ago", "Never checked", "Don't know"]
        )
    
    # Credit factors assessment
    st.subheader("🎯 Credit Factors Assessment")
    
    factor_col1, factor_col2 = st.columns(2)
    
    with factor_col1:
        payment_history = st.selectbox(
            "Payment History",
            ["Always on time", "1-2 late payments this year", "Several late payments", "Missed payments recently"],
            help="35% of your credit score"
        )
        
        credit_age = st.number_input(
            "Average Age of Accounts (years)",
            min_value=0.0, max_value=50.0, value=3.0, step=0.5,
            help="15% of your credit score"
        )
    
    with factor_col2:
        new_accounts = st.number_input(
            "New Accounts in Last 2 Years",
            min_value=0, max_value=20, value=2,
            help="10% of your credit score"
        )
        
        account_types = st.multiselect(
            "Types of Credit Accounts",
            ["Credit Cards", "Auto Loan", "Mortgage", "Student Loans", "Personal Loans", "Store Cards"],
            default=["Credit Cards"],
            help="10% of your credit score"
        )
    
    # Calculate utilization from current debts
    current_debts, _ = parse_debts_json(st.session_state.get("debts_json_shared", "[]"))
    
    credit_cards = [d for d in current_debts if d.limit and d.limit > 0]
    
    if credit_cards:
        total_balances = sum(d.balance for d in credit_cards)
        total_limits = sum(d.limit for d in credit_cards)
        current_utilization = (total_balances / total_limits * 100) if total_limits > 0 else 0
        
        st.metric(
            "Current Credit Utilization",
            f"{current_utilization:.1f}%",
            f"Target: <30%" if current_utilization > 30 else "✅ Good range"
        )
    else:
        current_utilization = st.slider(
            "Estimated Credit Utilization %",
            0.0, 100.0, 50.0, step=5.0,
            help="30% of your credit score - Current balances ÷ Credit limits"
        )
    
    # Generate personalized recommendations
    if st.button("🎯 Get Personalized Credit Improvement Plan", type="primary", use_container_width=True):
        with st.spinner("🧠 Generating personalized recommendations..."):
            try:
                from core.recommendations import generate_recommendations
                
                # Create a profile for recommendations
                profile = st.session_state.get("user_profile", UserProfile())
                
                # Generate debt-based recommendations
                debt_recs = generate_recommendations(profile, current_debts, top_n=10)
                
                # Generate credit-specific recommendations
                llm = get_llm()
                
                credit_context = f"""
                User Credit Profile:
                - Current Score: {current_score}
                - Target Score: {score_goal}
                - Payment History: {payment_history}
                - Credit Utilization: {current_utilization:.1f}%
                - Average Account Age: {credit_age} years
                - New Accounts: {new_accounts} in last 2 years
                - Account Types: {', '.join(account_types)}
                - Last Checked: {last_checked}
                
                Current Debts Summary:
                - Total Debt: ₹{sum(d.balance for d in current_debts):,.0f}
                - Number of Debts: {len(current_debts)}
                - Highest APR: {max([d.apr for d in current_debts], default=0)*100:.1f}%
                """
                
                credit_prompt = f"""
                As a credit counselor, provide a personalized credit improvement plan for this user:
                
                {credit_context}
                
                Please provide:
                1. **Priority Actions** (top 3 most impactful steps)
                2. **Timeline** (what to focus on first, second, etc.)
                3. **Expected Impact** (how much score improvement is realistic)
                4. **Monitoring Strategy** (how often to check, what to watch)
                5. **Red Flags to Avoid** (actions that could hurt the score)
                
                Be specific with numbers and timelines. Focus on actionable advice.
                """
                
                credit_response = llm.invoke([HumanMessage(content=credit_prompt)])
                
                # Display results
                st.success("✅ Personalized Credit Improvement Plan Generated!")
                
                # Credit-specific recommendations
                st.subheader("🎯 Credit Score Action Plan")
                st.markdown(credit_response.content)
                
                # Debt-based recommendations that impact credit
                st.subheader("💳 Debt Management for Credit Improvement")
                
                credit_relevant_recs = [r for r in debt_recs if r['type'] in ['repayment', 'credit', 'refinance']]
                
                if credit_relevant_recs:
                    for i, rec in enumerate(credit_relevant_recs[:5], 1):
                        with st.expander(f"{i}. {rec['title']}", expanded=i<=2):
                            st.write(f"**Action:** {rec['action']}")
                            st.write(f"**Explanation:** {rec['explanation']}")
                            if rec.get('estimated_savings', 0) > 0:
                                st.write(f"**Potential Savings:** {money(rec['estimated_savings'])}")
                
                # Quick wins section
                st.subheader("⚡ Quick Wins (30-90 days)")
                
                quick_wins = []
                
                if current_utilization > 30:
                    target_reduction = (current_utilization - 25) * sum(getattr(d, 'limit', 0) for d in credit_cards) / 100
                    quick_wins.append(f"**Pay down balances by ₹{target_reduction:,.0f}** to get utilization under 30%")
                
                if payment_history != "Always on time":
                    quick_wins.append("**Set up autopay** for all minimum payments to ensure on-time payments")
                
                if last_checked in ["6+ months ago", "Never checked"]:
                    quick_wins.append("**Check your credit report** for free at authorized sites to identify errors")
                
                quick_wins.append("**Don't close old credit cards** - keep them open to maintain credit history length")
                
                if new_accounts > 3:
                    quick_wins.append("**Avoid new credit applications** for the next 6-12 months")
                
                for win in quick_wins:
                    st.markdown(f"• {win}")
                
                # Progress tracking
                st.subheader("📈 Progress Tracking")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    **Monthly Checklist:**
                    - [ ] All payments made on time
                    - [ ] Balances paid down 
                    - [ ] No new credit applications
                    - [ ] Utilization under 30%
                    - [ ] Monitor for identity theft
                    """)
                
                with col2:
                    st.markdown("""
                    **Quarterly Review:**
                    - [ ] Check credit score
                    - [ ] Review credit report
                    - [ ] Assess progress toward goals
                    - [ ] Adjust strategy if needed
                    - [ ] Celebrate improvements!
                    """)
                
                # Score prediction
                st.subheader("🔮 Score Improvement Prediction")
                
                prediction_factors = {
                    "Utilization improvement": min(20, max(0, current_utilization - 25)),
                    "Payment history": 10 if payment_history != "Always on time" else 0,
                    "Account age": min(5, max(0, 2 - credit_age)),
                    "Credit mix": 5 if len(account_types) < 2 else 0
                }
                
                predicted_improvement = sum(prediction_factors.values())
                predicted_score = min(850, current_score + predicted_improvement)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Current Score", current_score)
                with col2:
                    st.metric("Predicted Score", predicted_score, f"+{predicted_improvement}")
                with col3:
                    timeline = "6-12 months" if predicted_improvement > 20 else "3-6 months"
                    st.metric("Estimated Timeline", timeline)
                
                # Export credit plan
                credit_plan = f"""
# Credit Improvement Plan
**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d')}
**Current Score:** {current_score}
**Target Score:** {score_goal}

## Personalized Recommendations
{credit_response.content}

## Quick Wins
{chr(10).join(['• ' + win.replace('**', '') for win in quick_wins])}

## Progress Tracking
Monthly: Check payments, balances, utilization
Quarterly: Review score and credit report
Target Timeline: {timeline}
                """
                
                st.download_button(
                    "📥 Download Credit Improvement Plan",
                    credit_plan,
                    file_name=f"credit_plan_{int(time.time())}.md",
                    mime="text/markdown",
                    use_container_width=True
                )
                
            except Exception as e:
                st.error(f"❌ Error generating recommendations: {e}")
                st.info("Please check your inputs and try again.")
    
    # Educational resources
    st.subheader("📚 Credit Education Resources")
    
    edu_tab1, edu_tab2, edu_tab3 = st.tabs([
        "📊 Score Factors", "🛡️ Credit Monitoring", "⚠️ Common Mistakes"
    ])
    
    with edu_tab1:
        st.markdown("""
        ### Credit Score Factors (FICO Model)
        
        **1. Payment History (35%)**
        - Most important factor
        - Late payments hurt your score
        - Bankruptcy, foreclosures have major impact
        - Recent missed payments hurt more than old ones
        
        **2. Credit Utilization (30%)**
        - Keep total utilization under 30%
        - Even better: under 10%
        - Both per-card and overall utilization matter
        - Pay before statement date to lower reported balances
        
        **3. Length of Credit History (15%)**
        - Average age of all accounts
        - Age of oldest account
        - Keep old cards open
        - Authorized user accounts can help
        
        **4. Credit Mix (10%)**
        - Different types: cards, loans, mortgages
        - Not worth opening accounts just for mix
        - Natural diversity over time is best
        
        **5. New Credit (10%)**
        - Hard inquiries lower score temporarily
        - Multiple inquiries for same loan type count as one
        - Avoid unnecessary credit applications
        """)
    
    with edu_tab2:
        st.markdown("""
        ### Free Credit Monitoring Options
        
        **Free Annual Reports:**
        - AnnualCreditReport.com (official site)
        - Review all 3 bureaus annually
        - Look for errors, unknown accounts
        
        **Free Score Monitoring:**
        - Many banks offer free FICO scores
        - Credit card companies provide scores
        - Apps like Credit Karma (VantageScore)
        
        **What to Monitor:**
        - New accounts you didn't open
        - Incorrect payment histories
        - Wrong balances or limits
        - Identity theft signs
        - Score changes and reasons
        
        **Dispute Process:**
        - Contact credit bureau in writing
        - Provide supporting documentation
        - Follow up within 30 days
        - Contact creditor if bureau doesn't fix
        """)
    
    with edu_tab3:
        st.markdown("""
        ### Common Credit Score Mistakes to Avoid
        
        **❌ Closing Old Credit Cards**
        - Reduces available credit (increases utilization)
        - Shortens credit history length
        - Keep old cards open with small purchases
        
        **❌ Maxing Out Credit Cards**
        - High utilization severely hurts scores
        - Even if you pay in full monthly
        - Keep balances low year-round
        
        **❌ Only Making Minimum Payments**
        - Keeps balances high
        - Increases utilization ratios
        - Costs more in interest long-term
        
        **❌ Co-signing Without Understanding**
        - You're responsible for full debt
        - Affects your credit utilization
        - Late payments impact your score
        
        **❌ Ignoring Credit Reports**
        - Errors can significantly hurt scores
        - Identity theft goes undetected
        - Miss opportunities for improvement
        
        **❌ Applying for Too Much Credit**
        - Multiple hard inquiries hurt score
        - Looks desperate for credit
        - Space applications at least 6 months apart
        """)

# =========================
# Footer and Additional Functions
# =========================

# Add footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>💸 <strong>AI Personal Debt Finance Advisor</strong> | Built with Streamlit + LangChain + Groq</p>
    <p><em>Disclaimer: This tool provides educational guidance only. Consult qualified financial advisors for personalized advice.</em></p>
    <p>🔒 Your data is processed locally and not stored permanently.</p>
</div>
""", unsafe_allow_html=True)

# Session state cleanup (optional)
def cleanup_old_session_data():
    """Clean up old session data to prevent memory issues"""
    keys_to_clean = []
    for key in st.session_state.keys():
        if key.startswith("temp_") or key.startswith("old_"):
            keys_to_clean.append(key)
    
    for key in keys_to_clean:
        del st.session_state[key]

# Call cleanup periodically
if len(st.session_state) > 50:  # Arbitrary threshold
    cleanup_old_session_data()

# Debug panel (only shown in development)
if os.getenv("DEBUG_MODE", "false").lower() == "true":
    with st.sidebar:
        st.markdown("---")
        st.subheader("🔧 Debug Info")
        if st.button("Show Session State"):
            st.json(dict(st.session_state))
        
        if st.button("Clear All Session Data"):
            st.session_state.clear()
            st.success("Session cleared!")
            st.rerun()