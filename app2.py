import os
import time
import io
from typing import List, Dict, Any, Optional, Tuple
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Document processing imports
import PyPDF2
import docx
from PIL import Image
import pytesseract

# ===== Core imports (keep your existing modules) =====
from core.schemas import Debt, UserProfile
from core.optimization import (
    compute_avalanche_plan,
    compute_snowball_plan,
    one_step_optimal_allocation,
)
from core.scenarios import simulate_payoff
from core.education import get_or_build_vectorstore, rag_answer
from core.prompts import SYSTEM_PROMPT_ADVISOR
from core.utils import money
from core.plan_utils import plan_to_dataframe, simulate_total_balance_series
from core.chat_tools import parse_slash_command, run_plan_tool, compare_baseline_vs_extra, run_rag_tool
from core.recommendations import generate_recommendations

# LLM (optional)
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

# ======================================
# App + CORS + Static
# ======================================
app = FastAPI(
    title="AI Personal Debt Finance Advisor",
    description="Your Personal Financial Assistant powered by AI",
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# serve /static (expects static/index.html)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ======================================
# Models
# ======================================
class ChatRequest(BaseModel):
    message: str
    debts: List[Dict[str, Any]] = []
    profile: Dict[str, Any] = {}

class DebtValidationRequest(BaseModel):
    debts: List[Dict[str, Any]]
    profile: Dict[str, Any] = {}

class PlanRequest(BaseModel):
    debts: List[Dict[str, Any]]
    budget: float
    strategy: str = "avalanche"  # avalanche | snowball | optimal
    max_months: int = 60

class WhatIfRequest(BaseModel):
    debts: List[Dict[str, Any]]
    base_budget: float
    scenario_type: str
    scenario_params: Dict[str, Any]
    max_months: int = 60

class EducationRequest(BaseModel):
    question: str

class CreditAssessmentRequest(BaseModel):
    current_score: int
    target_score: int
    payment_history: str
    credit_utilization: float
    credit_age: float
    new_accounts: int
    account_types: List[str]
    debts: List[Dict[str, Any]] = []

    




# ======================================
# Defaults
# ======================================
DEFAULT_DEBTS = [
    {"name": "Credit Card A", "balance": 120000, "apr": 0.36, "min_payment": 3000},
    {"name": "Student Loan",  "balance": 240000, "apr": 0.12, "min_payment": 2500},
    {"name": "Personal Loan", "balance":  80000, "apr": 0.18, "min_payment": 2000},
]


# ======================================
# Document Processing Functions
# ======================================
def extract_pdf_text(content: bytes) -> str:
    """Extract text from PDF content"""
    try:
        pdf_file = io.BytesIO(content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        
        for page_num, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                    print(f"PDF page {page_num + 1}: extracted {len(page_text)} chars")
            except Exception as e:
                print(f"Error extracting from PDF page {page_num + 1}: {e}")
                continue
                
        return text
    except Exception as e:
        raise Exception(f"PDF extraction failed: {e}")


def extract_docx_text(content: bytes) -> str:
    """Extract text from DOCX content"""
    try:
        doc_file = io.BytesIO(content)
        doc = docx.Document(doc_file)
        text = ""
        
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text += paragraph.text + "\n"
                
        return text
    except Exception as e:
        raise Exception(f"DOCX extraction failed: {e}")


def extract_text_file(content: bytes) -> str:
    """Extract text from plain text files"""
    try:
        # Try UTF-8 first, then fallback encodings
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                return content.decode(encoding)
            except UnicodeDecodeError:
                continue
        raise Exception("Could not decode text file with any encoding")
    except Exception as e:
        raise Exception(f"Text file extraction failed: {e}")


def extract_image_text(content: bytes) -> str:
    """Extract text from images using OCR"""
    try:
        image = Image.open(io.BytesIO(content))
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        raise Exception(f"OCR extraction failed: {e}. Make sure tesseract is installed.")


async def summarize_docs(files: List[UploadFile], llm=None) -> str:
    """
    Enhanced document summarizer with better error handling and debugging
    """
    if not files:
        return "No files provided for analysis."
    
    all_text = ""
    processed_files = []
    errors = []
    
    for file in files:
        try:
            print(f"Processing file: {file.filename}")
            
            # Reset file pointer to beginning
            await file.seek(0)
            content = await file.read()
            
            if not content:
                errors.append(f"{file.filename}: File is empty")
                continue
                
            print(f"File size: {len(content)} bytes")
            
            # Extract text based on file type
            text = ""
            file_ext = os.path.splitext(file.filename)[1].lower() if file.filename else ""
            
            if file_ext == '.pdf':
                text = extract_pdf_text(content)
            elif file_ext in ['.docx', '.doc']:
                text = extract_docx_text(content)
            elif file_ext in ['.txt', '.md']:
                text = extract_text_file(content)
            elif file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                text = extract_image_text(content)
            else:
                # Try to detect content type or treat as text
                try:
                    text = content.decode('utf-8')
                except:
                    try:
                        text = content.decode('latin-1')
                    except:
                        errors.append(f"{file.filename}: Unsupported file type or encoding")
                        continue
            
            if text.strip():
                all_text += f"\n--- {file.filename} ---\n{text}\n"
                processed_files.append(file.filename)
                print(f"Extracted {len(text)} characters from {file.filename}")
            else:
                errors.append(f"{file.filename}: No text could be extracted")
                
        except Exception as e:
            errors.append(f"{file.filename}: {str(e)}")
            print(f"Error processing {file.filename}: {e}")
    
    # Debug output
    print(f"Total text extracted: {len(all_text)} characters")
    print(f"Processed files: {processed_files}")
    print(f"Errors: {errors}")
    
    if not all_text.strip():
        error_summary = "; ".join(errors) if errors else "Unknown error"
        return f"No text could be extracted from any files. Issues: {error_summary}"
    
    # Generate summary
    if llm:
        try:
            summary_prompt = f"""
            Analyze and summarize the following documents:
            
            {all_text[:10000]}  # Limit to first 10k chars to avoid token limits
            
            Provide a concise summary highlighting:
            1. Key financial information
            2. Important dates and deadlines  
            3. Action items or recommendations
            4. Any concerning issues
            """
            
            response = llm.invoke([HumanMessage(content=summary_prompt)])
            summary = response.content if hasattr(response, 'content') else str(response)
            
            return f"""Document Analysis Summary:

{summary}

Files Processed: {', '.join(processed_files)}
{f"Issues encountered: {'; '.join(errors)}" if errors else ""}
"""
        except Exception as e:
            print(f"LLM summarization failed: {e}")
            return f"""Basic Document Summary:

Processed {len(processed_files)} files successfully: {', '.join(processed_files)}

Total content length: {len(all_text)} characters

{f"Issues encountered: {'; '.join(errors)}" if errors else ""}

Note: AI summarization unavailable, showing basic file info.
"""
    else:
        return f"""Document Processing Complete:

Files processed: {', '.join(processed_files)}
Total content extracted: {len(all_text)} characters

{f"Issues: {'; '.join(errors)}" if errors else ""}

Configure GROQ_API_KEY for AI-powered summaries.
"""


# ======================================
# Helpers
# ======================================
def get_llm(model: Optional[str] = None):
    """Return ChatGroq LLM if GROQ_API_KEY exists, else None (so endpoints can still work)."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    return ChatGroq(
        model=model or os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
        groq_api_key=api_key,
        temperature=0.2,
    )

def parse_debts_json(debts_data: List[Dict[str, Any]]) -> Tuple[List[Debt], Optional[str]]:
    try:
        debts: List[Debt] = []
        for i, d in enumerate(debts_data):
            for k in ("name", "balance", "apr", "min_payment"):
                if k not in d:
                    return [], f"Debt {i+1} missing field: {k}"
            debts.append(
                Debt(
                    name=str(d["name"]).strip(),
                    balance=float(d["balance"]),
                    apr=float(d["apr"]),
                    min_payment=float(d["min_payment"]),
                    limit=float(d.get("limit", 0)) if d.get("limit") else None,
                )
            )
        return debts, None
    except Exception as e:
        return [], f"Error parsing debts: {e}"

def ensure_serializable_number(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0

def schedule_to_frontend_records(df) -> List[Dict[str, Any]]:
    """
    Map your internal plan DataFrame to the schema expected by the frontend:
    { month, debt, payment, interest, principal, balance }
    Unknown columns are handled gracefully.
    """
    if df is None or df.empty:
        return []

    # Try to infer best columns
    cols = {c.lower(): c for c in df.columns}
    month_col     = cols.get("month") or cols.get("period") or list(df.columns)[0]
    debt_col      = cols.get("debt") or cols.get("account") or cols.get("name") or list(df.columns)[1]
    pay_col       = cols.get("payment") or cols.get("total_paid") or cols.get("paid") or None
    interest_col  = cols.get("interest") or cols.get("interest_paid") or None
    principal_col = cols.get("principal") or cols.get("principal_paid") or None
    balance_col   = cols.get("balance") or cols.get("remaining") or None

    records = []
    for _, row in df.iterrows():
        records.append({
            "month":     int(ensure_serializable_number(row.get(month_col, 0))),
            "debt":      str(row.get(debt_col, "All Debts")),
            "payment":   ensure_serializable_number(row.get(pay_col, 0)),
            "interest":  ensure_serializable_number(row.get(interest_col, 0)),
            "principal": ensure_serializable_number(row.get(principal_col, 0)),
            "balance":   ensure_serializable_number(row.get(balance_col, 0)),
        })
    return records


# ======================================
# Routes
# ======================================
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse("<h1>Frontend not found</h1><p>Put your build at static/index.html</p>", status_code=404)

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "message": "AI Debt Finance Advisor API is running!", "timestamp": time.time()}

@app.post("/api/debts/validate")
async def validate_debts(request: DebtValidationRequest):
    try:
        debts, error = parse_debts_json(request.debts)
        if error:
            return {"valid": False, "error": error}

        profile = UserProfile(**request.profile) if request.profile else UserProfile()
        total_balance = sum(d.balance for d in debts)
        total_min_pay = sum(d.min_payment for d in debts)
        weighted_apr = (sum(d.apr * d.balance for d in debts) / total_balance) if total_balance > 0 else 0.0

        avail = max(0.0, profile.monthly_income - profile.monthly_expenses)
        leftover = avail - total_min_pay

        return {
            "valid": True,
            "summary": {
                "debts_count": len(debts),
                "total_balance": total_balance,
                "total_min_payments": total_min_pay,
                "weighted_apr": weighted_apr,
                "available_budget": avail,
                "leftover_budget": leftover,
                "has_deficit": leftover < 0,
                "formatted": {
                    "total_balance": money(total_balance),
                    "total_min_payments": money(total_min_pay),
                    "weighted_apr": f"{weighted_apr*100:.2f}%",
                    "available_budget": money(avail),
                    "leftover_budget": money(leftover),
                },
            },
        }
    except Exception as e:
        return {"valid": False, "error": str(e)}

@app.post("/api/plans/generate")
async def generate_repayment_plan(request: PlanRequest):
    """
    Returns JSON in the exact shape the frontend expects:
    {
      success, strategy, formatted{months_to_payoff,total_interest,principal_total,efficiency},
      months_to_payoff, total_interest, total_payments, principal_total,
      schedule, balance_series
    }
    """
    try:
        debts, error = parse_debts_json(request.debts)
        if error:
            raise HTTPException(status_code=400, detail=error)

        if not debts:
            raise HTTPException(status_code=400, detail="No debts provided")

        min_sum = sum(d.min_payment for d in debts)
        if request.budget < min_sum:
            raise HTTPException(
                status_code=400,
                detail=f"Budget {money(request.budget)} is below total minimums {money(min_sum)}",
            )

        strat = (request.strategy or "avalanche").lower().strip()
        if strat == "snowball":
            plan = compute_snowball_plan(debts, request.budget, request.max_months)
            strategy_name = "Debt Snowball"
        elif strat == "optimal":
            # One-step allocation repeated via simulate_payoff for a reasonable horizon
            plan = one_step_optimal_allocation(debts, request.budget)
            strategy_name = "Mathematical Optimal"
        else:
            plan = compute_avalanche_plan(debts, request.budget, request.max_months)
            strategy_name = "Debt Avalanche"

        # Basic sanity
        if not getattr(plan, "months", None):
            raise HTTPException(status_code=400, detail="Could not generate valid plan")

        total_interest = float(sum(m.total_interest for m in plan.months))
        total_payments = float(sum(m.total_paid for m in plan.months))
        principal_total = float(max(0.0, total_payments - total_interest))

        # DataFrame -> JSON rows
        schedule_df = plan_to_dataframe(plan)
        schedule = schedule_to_frontend_records(schedule_df)

        # Balance series
        balance_series = [float(x) for x in simulate_total_balance_series(debts, plan)]

        months_to_payoff = int(len(plan.months))
        efficiency = f"{(principal_total/total_payments*100):.1f}%" if total_payments > 0 else "0%"

        return {
            "success": True,
            "strategy": strategy_name,
            "months_to_payoff": months_to_payoff,
            "total_interest": total_interest,
            "total_payments": total_payments,
            "principal_total": principal_total,
            "schedule": schedule,
            "balance_series": balance_series,
            "formatted": {
                "months_to_payoff": f"{months_to_payoff} months ({months_to_payoff/12:.1f} years)",
                "total_interest": money(total_interest),
                "total_payments": money(total_payments),
                "principal_total": money(principal_total),
                "efficiency": efficiency,
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        # Return success=False so frontend shows a friendly message instead of hard error
        return {"success": False, "error": f"Error generating plan: {e}"}

@app.post("/api/scenarios/whatif")
async def whatif_analysis(request: WhatIfRequest):
    try:
        debts, error = parse_debts_json(request.debts)
        if error:
            raise HTTPException(status_code=400, detail=error)
        if not debts:
            raise HTTPException(status_code=400, detail="No debts provided")

        base_plan = compute_avalanche_plan(debts, request.base_budget, request.max_months)
        base_interest = float(sum(m.total_interest for m in base_plan.months))
        base_months = int(len(base_plan.months))
        base_total_payment = float(sum(m.total_paid for m in base_plan.months))

        # clone debts for scenario
        scenario_debts = [Debt(**d.model_dump()) for d in debts]
        scenario_budget = float(request.base_budget)

        if request.scenario_type == "extra_payment":
            scenario_budget += float(request.scenario_params.get("extra_monthly", 0))
        elif request.scenario_type == "budget_reduction":
            scenario_budget -= float(request.scenario_params.get("reduction", 0))
            scenario_budget = max(scenario_budget, sum(d.min_payment for d in scenario_debts))
        elif request.scenario_type == "rate_change":
            rate_delta = float(request.scenario_params.get("rate_change", 0)) / 100.0
            for d in scenario_debts:
                d.apr = max(0.0, d.apr + rate_delta)

        scenario_plan = compute_avalanche_plan(scenario_debts, scenario_budget, request.max_months)
        scenario_interest = float(sum(m.total_interest for m in scenario_plan.months))
        scenario_months = int(len(scenario_plan.months))
        scenario_total_payment = float(sum(m.total_paid for m in scenario_plan.months))

        months_saved = int(base_months - scenario_months)
        interest_saved = float(base_interest - scenario_interest)

        return {
            "success": True,
            "scenario_type": request.scenario_type,
            "base": {
                "months": base_months,
                "total_interest": base_interest,
                "total_payments": base_total_payment,
            },
            "scenario": {
                "months": scenario_months,
                "total_interest": scenario_interest,
                "total_payments": scenario_total_payment,
            },
            "savings": {
                "months_saved": months_saved,
                "interest_saved": interest_saved,
            },
            "formatted": {
                "months_saved": f"{months_saved:+} months",
                "interest_saved": (money(interest_saved) + (" saved" if interest_saved > 0 else " more")),
                "base_summary": f"{base_months} months, {money(base_interest)} interest",
                "scenario_summary": f"{scenario_months} months, {money(scenario_interest)} interest",
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        return {"success": False, "error": f"Error in what-if analysis: {e}"}

@app.post("/api/education/ask")
async def ask_education_question(request: EducationRequest):
    try:
        kb_vs = get_or_build_vectorstore()
        answer, sources = rag_answer(request.question, kb_vs, k=6)
        return {"success": True, "question": request.question, "answer": answer, "sources": sources}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/documents/analyze")
async def analyze_documents(files: List[UploadFile] = File(...)):
    """
    Enhanced document analyzer with detailed debugging and error handling
    """
    try:
        print(f"Received {len(files)} files for analysis")
        
        # Check if files are actually received
        if not files:
            return {
                "success": False,
                "summary": "No files received",
                "files_processed": 0,
                "file_names": [],
                "error": "No files provided"
            }
        
        # Debug file info
        for i, file in enumerate(files):
            print(f"File {i}: {file.filename}, content_type: {file.content_type}")
            await file.seek(0)
            content = await file.read()
            print(f"File {i} size: {len(content)} bytes")
            await file.seek(0)  # Reset for processing
        
        llm = get_llm()
        summary = await summarize_docs(files, llm)
        
        return {
            "success": True,
            "summary": summary,
            "files_processed": len(files),
            "file_names": [f.filename for f in files],
        }
        
    except Exception as e:
        print(f"Document analysis error: {e}")
        names = []
        try:
            names = [f.filename for f in files if f.filename]
        except Exception:
            pass
        
        return {
            "success": False,
            "summary": f"Document processing failed: {str(e)}",
            "files_processed": len(names),
            "file_names": names,
            "error": str(e),
        }

@app.post("/api/test-upload")
async def test_file_upload(files: List[UploadFile] = File(...)):
    """Test endpoint to debug file upload issues"""
    results = []
    for file in files:
        await file.seek(0)
        content = await file.read()
        results.append({
            "filename": file.filename,
            "size": len(content),
            "content_type": file.content_type,
            "first_100_chars": content[:100].decode('utf-8', errors='ignore') if content else "Empty"
        })
    return {"files": results}

@app.post("/api/chat")
async def chat_with_ai(request: ChatRequest):
    try:
        debts, debt_error = parse_debts_json(request.debts)
        if debt_error:
            return {"response": f"Debt data error: {debt_error}", "type": "error"}

        profile = UserProfile(**request.profile) if request.profile else UserProfile()
        cmd, kv = parse_slash_command(request.message)

        if cmd == "summary":
            if not debts:
                return {"response": "No debts provided.", "type": "error"}
            total_balance = sum(d.balance for d in debts)
            total_min = sum(d.min_payment for d in debts if d.balance > 0)
            weighted_apr = (sum(d.apr * d.balance for d in debts) / total_balance) if total_balance > 0 else 0
            response = (
                f"**Debt Summary:**\n"
                f"- Total Balance: {money(total_balance)}\n"
                f"- Weighted Average APR: {weighted_apr*100:.2f}%\n"
                f"- Total Minimum Payments: {money(total_min)}/month\n"
                f"- Available Budget: {money(max(0, profile.monthly_income - profile.monthly_expenses))}/month\n"
                f"- Leftover after minimums: {money(max(0, profile.monthly_income - profile.monthly_expenses - total_min))}/month"
            )
            return {"response": response, "type": "summary"}

        elif cmd == "plan":
            if not debts:
                return {"response": "Please provide valid debts first.", "type": "error"}
            try:
                budget = float(kv.get("budget", profile.extra_payment + max(0.0, profile.monthly_income - profile.monthly_expenses)))
                months = int(float(kv.get("months", 60)))
                strategy = kv.get("strategy", "avalanche")
                result = run_plan_tool(debts, budget, months, strategy=strategy)

                response = (
                    f"**{result['strategy_name']} Plan Results:**\n"
                    f"- **Duration:** {result['months']} months ({result['months']/12:.1f} years)\n"
                    f"- **Total Interest:** {money(result['total_interest'])}\n"
                    f"- **Monthly Budget:** {money(budget)}"
                )
                schedule_data = result['schedule_df'].head(10).to_dict('records') if not result['schedule_df'].empty else []
                return {
                    "response": response,
                    "type": "plan",
                    "data": {
                        "strategy": result['strategy_name'],
                        "months": result['months'],
                        "total_interest": result['total_interest'],
                        "budget": budget,
                        "schedule": schedule_data,
                        "balance_series": result.get('balance_series', []),
                    },
                }
            except Exception as e:
                return {"response": f"Error generating plan: {e}", "type": "error"}

        elif cmd == "whatif":
            if not debts:
                return {"response": "Please provide valid debts first.", "type": "error"}
            try:
                budget = float(kv.get("budget", profile.extra_payment + max(0.0, profile.monthly_income - profile.monthly_expenses)))
                months = int(float(kv.get("months", 60)))
                extra = float(kv.get("extra", 0.0))
                strategy = kv.get("strategy", "avalanche")
                comparison = compare_baseline_vs_extra(debts, budget, months, extra=extra, strategy=strategy)

                response = (
                    f"**What-If Analysis ({strategy.title()}) with extra ₹{extra:,.0f}/month:**\n\n"
                    f"📊 **Comparison Results:**\n"
                    f"- **Baseline:** {comparison['baseline']['months']} months, {money(comparison['baseline']['total_interest'])} interest\n"
                    f"- **With Extra Payment:** {comparison['scenario']['months']} months, {money(comparison['scenario']['total_interest'])} interest\n\n"
                    f"💰 **Potential Savings:**\n"
                    f"- **Interest Saved:** {money(comparison['interest_savings'])}\n"
                    f"- **Months Saved:** {comparison['months_saved']} months\n"
                    f"- **Time Reduction:** {comparison['months_saved']/12:.1f} years faster"
                )
                return {"response": response, "type": "whatif", "data": comparison}
            except Exception as e:
                return {"response": f"Error in what-if analysis: {e}", "type": "error"}

        elif cmd == "rag":
            try:
                kb_vs = get_or_build_vectorstore()
                question = kv.get("question") or kv.get("q", "")
                if not question:
                    return {"response": "Please provide a question. Use: `/rag question=\"your question\"`", "type": "error"}
                answer, sources = run_rag_tool(question, kb_vs)
                return {"response": f"{answer}\n\n**Sources:** {', '.join(sources) if sources else 'Knowledge Base'}", "type": "education", "data": {"sources": sources}}
            except Exception as e:
                return {"response": f"Knowledge base error: {e}", "type": "error"}

        # Normal chat
        try:
            llm = get_llm()
            if not llm:
                return {"response": "LLM not configured (set GROQ_API_KEY).", "type": "error"}
            context = (
                f"User Profile:\n"
                f"- Monthly Income: {money(profile.monthly_income)}\n"
                f"- Monthly Expenses: {money(profile.monthly_expenses)}\n"
                f"- Available for Debt: {money(profile.extra_payment)}\n\n"
                f"Current Debts: {[d.model_dump() for d in debts]}"
            )
            messages = [
                SystemMessage(content=SYSTEM_PROMPT_ADVISOR),
                HumanMessage(content=f"Context:\n{context}\n\nUser Question: {request.message}"),
            ]
            response = llm.invoke(messages)
            return {"response": response.content, "type": "chat"}
        except Exception as e:
            return {"response": f"AI Assistant Error: {e}", "type": "error"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/credit/assess")
async def assess_credit_improvement(request: CreditAssessmentRequest):
    try:
        llm = get_llm()
        debts: List[Debt] = []
        if request.debts:
            debts, debt_error = parse_debts_json(request.debts)
            if debt_error:
                debts = []

        credit_context = f"""
User Credit Profile:
- Current Score: {request.current_score}
- Target Score: {request.target_score}
- Payment History: {request.payment_history}
- Credit Utilization: {request.credit_utilization:.1f}%
- Average Account Age: {request.credit_age} years
- New Accounts: {request.new_accounts} in last 2 years
- Account Types: {', '.join(request.account_types)}

Current Debts Summary:
- Total Debt: ₹{sum(d.balance for d in debts):,.0f}
- Number of Debts: {len(debts)}
- Highest APR: {max([d.apr for d in debts], default=0)*100:.1f}%
"""
        prompt = f"""As a credit counselor, provide a personalized, actionable plan:\n{credit_context}\n\nReturn: Priority actions, timeline, expected impact, monitoring strategy, and red flags."""
        analysis = ""
        if llm:
            analysis = llm.invoke([HumanMessage(content=prompt)]).content
        else:
            analysis = "LLM not configured. Add GROQ_API_KEY to get a personalized narrative."

        # very rough heuristic projection
        prediction_factors = {
            "utilization_improvement": min(20, max(0, request.credit_utilization - 25)),
            "payment_history": 10 if request.payment_history != "Always on time" else 0,
            "account_age": min(5, max(0, 2 - request.credit_age)),
            "credit_mix": 5 if len(request.account_types) < 2 else 0,
        }
        predicted_improvement = float(sum(prediction_factors.values()))
        predicted_score = int(min(850, request.current_score + predicted_improvement))

        return {
            "success": True,
            "current_score": request.current_score,
            "predicted_score": predicted_score,
            "improvement_potential": predicted_improvement,
            "timeline": "6-12 months" if predicted_improvement > 20 else "3-6 months",
            "analysis": analysis,
            "prediction_factors": prediction_factors,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/defaults/debts")
async def get_default_debts():
    return {"debts": DEFAULT_DEBTS}


# ======================================
# Additional Utility Routes
# ======================================
@app.get("/api/system/info")
async def get_system_info():
    """Get system information for debugging"""
    try:
        import sys
        import platform
        
        # Check available packages
        packages = {}
        try:
            import PyPDF2
            packages["PyPDF2"] = PyPDF2.__version__
        except ImportError:
            packages["PyPDF2"] = "Not installed"
        
        try:
            import docx
            packages["python-docx"] = docx.__version__
        except ImportError:
            packages["python-docx"] = "Not installed"
        
        try:
            import PIL
            packages["Pillow"] = PIL.__version__
        except ImportError:
            packages["Pillow"] = "Not installed"
        
        try:
            import pytesseract
            packages["pytesseract"] = "Available"
            # Try to get tesseract version
            try:
                tesseract_version = pytesseract.get_tesseract_version()
                packages["tesseract-ocr"] = str(tesseract_version)
            except:
                packages["tesseract-ocr"] = "Not found"
        except ImportError:
            packages["pytesseract"] = "Not installed"
            packages["tesseract-ocr"] = "Not installed"

        return {
            "python_version": sys.version,
            "platform": platform.platform(),
            "packages": packages,
            "groq_configured": bool(os.getenv("GROQ_API_KEY")),
            "static_dir_exists": os.path.exists("static"),
            "index_html_exists": os.path.exists("static/index.html"),
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/documents/supported-formats")
async def get_supported_formats():
    """Return list of supported document formats"""
    formats = {
        "text": [".txt", ".md"],
        "pdf": [".pdf"],
        "word": [".docx", ".doc"],
        "image": [".jpg", ".jpeg", ".png", ".bmp", ".tiff"],
    }
    
    # Check which formats are actually available
    available_formats = {"text": formats["text"]}
    
    try:
        import PyPDF2
        available_formats["pdf"] = formats["pdf"]
    except ImportError:
        pass
    
    try:
        import docx
        available_formats["word"] = formats["word"]
    except ImportError:
        pass
    
    try:
        import PIL
        import pytesseract
        available_formats["image"] = formats["image"]
    except ImportError:
        pass
    
    return {
        "supported_formats": available_formats,
        "all_extensions": [ext for exts in available_formats.values() for ext in exts],
        "notes": {
            "pdf": "Requires PyPDF2",
            "word": "Requires python-docx", 
            "image": "Requires Pillow and pytesseract (tesseract-ocr)",
            "text": "Native support"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
