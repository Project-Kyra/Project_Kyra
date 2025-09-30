import streamlit as st
import fitz  # PyMuPDF
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Dummy users
USERS = {
    "admin": {"password": "admin123", "role": "admin"},
    "company1": {"password": "comp123", "role": "company"},
    "evaluator1": {"password": "eval123", "role": "evaluator"},
}

# Benchmark abstracts for novelty scoring
BENCHMARKS = [
    "Optimized coal mining using IoT sensors.",
    "Advanced rare earth extraction from coal ash.",
    "AI techniques for predictive maintenance in mines.",
]

# Initialize session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "role" not in st.session_state:
    st.session_state.role = ""
if "proposals" not in st.session_state:
    st.session_state.proposals = []

# --- Scoring functions based on guidelines ---

def extract_pdf_text(pdf_bytes) -> str:
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

def score_relevance(text: str) -> float:
    keywords = ["coal mining", "safety", "environmental sustainability", "energy efficiency",
                "automation", "clean coal"]
    matches = sum(1 for kw in keywords if kw in text.lower())
    return round(100 * matches / len(keywords), 2)

def score_novelty(text: str) -> float:
    documents = BENCHMARKS + [text]
    tfidf = TfidfVectorizer().fit_transform(documents)
    cos_similarities = (tfidf[-1] @ tfidf[:-1].T).toarray()[0]
    novelty_score = 1 - np.max(cos_similarities)
    return round(novelty_score * 100, 2)

def score_technical_feasibility(text: str) -> float:
    criteria = ["objective", "methodology", "timeline", "resources", "expertise", "partnership"]
    found = sum(1 for c in criteria if c in text.lower())
    return round(100 * found / len(criteria), 2)

def score_financial_viability(budget_df: pd.DataFrame) -> (float, list):
    issues = []
    total = budget_df["Amount"].sum()
    max_total = 2000000  # Example cap 20 Lakhs INR
    cost_benefit_justified = True  # For demo, assume True

    if total > max_total:
        issues.append("Budget exceeds maximum allowed cap of INR 20 Lakhs.")
    # Further cost-benefit checks can be added here
    score = 100 if not issues else 50
    return score, issues

def score_impact_potential(text: str) -> float:
    keywords = ["efficiency", "safety", "environment", "emissions", "clean energy"]
    matches = sum(1 for kw in keywords if kw in text.lower())
    return round(100 * matches / len(keywords), 2)

def score_institutional_capability(text: str) -> float:
    keywords = ["track record", "expertise", "facility", "experience"]
    matches = sum(1 for kw in keywords if kw in text.lower())
    return round(100 * matches / len(keywords), 2)

def score_compliance_and_completeness(text: str) -> float:
    keywords = ["forms", "annexures", "financial details", "approval", "ethical", "regulatory"]
    matches = sum(1 for kw in keywords if kw in text.lower())
    return round(100 * matches / len(keywords), 2)

def compute_weighted_score(text: str, budget_df: pd.DataFrame):
    r1 = score_relevance(text)         # 20%
    r2 = score_novelty(text)           # 20%
    r3 = score_technical_feasibility(text)  # 20%
    r4, finance_issues = score_financial_viability(budget_df)  # 15%
    r5 = score_impact_potential(text) # 15%
    r6 = score_institutional_capability(text) # 5%
    r7 = score_compliance_and_completeness(text) # 5%

    weighted_score = (r1 * 0.20) + (r2 * 0.20) + (r3 * 0.20) + (r4 * 0.15) + \
                     (r5 * 0.15) + (r6 * 0.05) + (r7 * 0.05)

    if weighted_score >= 70:
        status = "Accepted"
        reasons = []
    elif weighted_score >= 50:
        status = "Conditional Acceptance (Revision Needed)"
        reasons = finance_issues if finance_issues else ["Requires revision."]
    else:
        status = "Rejected"
        reasons = finance_issues if finance_issues else ["Score below threshold."]

    scores = {
        "Relevance to CIL Goals": r1,
        "Novelty/Innovation": r2,
        "Technical Feasibility": r3,
        "Financial Viability": r4,
        "Impact Potential": r5,
        "Institutional Capability": r6,
        "Compliance & Completeness": r7,
        "Overall Score": round(weighted_score, 2),
        "Status": status,
        "Reasons": reasons
    }
    return scores

def authenticate(username, password):
    user = USERS.get(username)
    if user and user["password"] == password:
        return user["role"]
    return None

def login():
    st.title("NaCCER Proposal Evaluation System")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    login_btn = st.button("Login")
    if login_btn:
        role = authenticate(username, password)
        if role:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.role = role
            st.experimental_rerun()
        else:
            st.error("Invalid username or password")

def logout():
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.role = ""
    st.experimental_rerun()

def company_dashboard():
    st.header(f"Welcome, {st.session_state.username} (Company)")

    with st.form("proposal_form"):
        pdf_file = st.file_uploader("Upload Proposal PDF", type=["pdf"], key="pdf")
        budget_file = st.file_uploader(
            "Upload Budget CSV (with 'Amount' column)", type=["csv"], key="budget"
        )
        submit = st.form_submit_button("Submit Proposal")

        if submit:
            if pdf_file is None:
                st.error("Please upload the proposal PDF.")
                return
            if budget_file is None:
                st.error("Please upload the budget CSV file.")
                return

            proposal_text = extract_pdf_text(pdf_file.read())
            if not proposal_text.strip():
                st.error("Could not extract text from PDF.")
                return

            try:
                budget_df = pd.read_csv(budget_file)
            except Exception as e:
                st.error(f"Error reading budget CSV: {e}")
                return

            if "Amount" not in budget_df.columns:
                st.error("Budget CSV must contain 'Amount' column.")
                return

            scores = compute_weighted_score(proposal_text, budget_df)

            new_prop = {
                "id": len(st.session_state.proposals) + 1,
                "text": proposal_text,
                "scores": scores,
                "user": st.session_state.username,
                "status": scores["Status"],
                "eval_comment": "",
            }

            st.session_state.proposals.append(new_prop)
            st.success(f"Proposal submitted. Status: {scores['Status']}")

    st.subheader("Your Submitted Proposals")
    user_props = [p for p in st.session_state.proposals if p["user"] == st.session_state.username]
    for p in user_props:
        st.markdown(f"**Proposal #{p['id']}** - Status: {p['status']}")
        for k, v in p["scores"].items():
            if k not in ["Reasons", "Status"]:
                st.write(f"{k}: {v}")
        if p["scores"]["Reasons"]:
            st.warning("Reasons:")
            for r in p["scores"]["Reasons"]:
                st.write("- " + r)
        if p["eval_comment"]:
            st.info(f"Evaluator Comments: {p['eval_comment']}")

def admin_dashboard():
    st.header(f"Welcome, {st.session_state.username} (Admin)")
    st.subheader("All Proposals")
    for p in st.session_state.proposals:
        st.markdown(f"**Proposal #{p['id']}** by {p['user']} - Status: {p['status']}")
        for k, v in p["scores"].items():
            if k not in ["Reasons", "Status"]:
                st.write(f"{k}: {v}")
        if p["scores"]["Reasons"]:
            st.error("Alerts: " + ", ".join(p["scores"]["Reasons"]))
        if p["eval_comment"]:
            st.info(f"Evaluator Comments: {p['eval_comment']}")

def evaluator_dashboard():
    st.header(f"Welcome, {st.session_state.username} (Evaluator)")
    st.info("Evaluator panel coming soon.")

def main():
    if not st.session_state.logged_in:
        login()
    else:
        st.sidebar.write(f"Logged in as: {st.session_state.username} ({st.session_state.role})")
        if st.sidebar.button("Logout"):
            logout()
        role = st.session_state.role
        if role == "company":
            company_dashboard()
        elif role == "admin":
            admin_dashboard()
        elif role == "evaluator":
            evaluator_dashboard()

if __name__ == "__main__":
    main()
