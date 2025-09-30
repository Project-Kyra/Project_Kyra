import streamlit as st
import fitz
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

USERS = {
    "admin": {"password": "admin123", "role": "admin"},
    "company1": {"password": "comp123", "role": "company"},
    "evaluator1": {"password": "eval123", "role": "evaluator"},
}

BENCHMARKS = [
    "Optimized coal mining using IoT sensors.",
    "Advanced rare earth extraction from coal ash.",
    "AI techniques for predictive maintenance in mines.",
]

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "role" not in st.session_state:
    st.session_state.role = ""
if "proposals" not in st.session_state:
    st.session_state.proposals = []

def extract_pdf_text(pdf_bytes):
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except:
        return ""

def min_score(val, default=10):
    return max(val, default)

def score_relevance(text):
    keywords = ["coal mining", "safety", "environmental sustainability", "energy efficiency","automation", "clean coal"]
    val = round(100 * sum(kw in text.lower() for kw in keywords) / len(keywords), 2)
    return min_score(val)

def score_novelty(text):
    docs = BENCHMARKS + [text]
    tfidf = TfidfVectorizer().fit_transform(docs)
    sims = (tfidf[-1] @ tfidf[:-1].T).toarray()[0]
    val = round((1 - np.max(sims)) * 100, 2)
    return min_score(val)

def score_technical_feasibility(text):
    keys = ["objective", "methodology", "timeline", "resources", "expertise", "partnership"]
    val = round(100 * sum(k in text.lower() for k in keys) / len(keys), 2)
    return min_score(val)

def score_financial_viability(budget_df):
    issues = []
    total = budget_df["Amount"].sum()
    max_total = 2000000
    milestone_limit = 0.4
    first_milestone = budget_df.iloc[0]["Amount"] if not budget_df.empty else 0

    if total > max_total:
        issues.append("Budget exceeds max INR 20 lakhs")
    if total > 0 and first_milestone / total > milestone_limit:
        issues.append("First milestone > 40% of total budget")
    score = 100 if not issues else 50
    return min_score(score), issues

def score_impact_potential(text):
    keys = ["efficiency", "safety", "environment", "emissions", "clean energy"]
    val = round(100 * sum(k in text.lower() for k in keys) / len(keys), 2)
    return min_score(val)

def score_institutional_capability(text):
    keys = ["track record", "expertise", "facility", "experience"]
    val = round(100 * sum(k in text.lower() for k in keys) / len(keys), 2)
    return min_score(val)

def score_compliance_and_completeness(text):
    keys = ["forms", "annexures", "financial details", "approval", "ethical", "regulatory"]
    val = round(100 * sum(k in text.lower() for k in keys) / len(keys), 2)
    return min_score(val)

def compute_weighted_score(text, budget_df):
    r1 = score_relevance(text)
    r2 = score_novelty(text)
    r3 = score_technical_feasibility(text)
    r4, fin_issues = score_financial_viability(budget_df)
    r5 = score_impact_potential(text)
    r6 = score_institutional_capability(text)
    r7 = score_compliance_and_completeness(text)

    score = round(r1*0.2 + r2*0.2 + r3*0.2 + r4*0.15 + r5*0.15 + r6*0.05 + r7*0.05,2)
    if score >= 70:
        status = "Accepted"
        reasons = []
    elif score >= 50:
        status = "Conditional Acceptance (Revision Needed)"
        reasons = fin_issues if fin_issues else ["Requires revision"]
    else:
        status = "Rejected"
        reasons = fin_issues if fin_issues else ["Score below threshold"]

    return {"Relevance": r1, "Novelty": r2, "Technical Feasibility": r3,
            "Financial Viability": r4, "Impact": r5, "Institutional Capability": r6,
            "Compliance": r7, "Overall Score": score, "Status": status, "Reasons": reasons}

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
            try:
                st.experimental_rerun()
            except Exception:
                pass  # fail silently to avoid crash
            return
        else:
            st.error("Invalid username or password")


def logout():
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.role = ""
    st.experimental_rerun()

def company_dashboard():
    st.header(f"Welcome, {st.session_state.username} (Company)")
    with st.form("submit_proposal"):
        pdf_file = st.file_uploader("Upload Proposal PDF", type=["pdf"])
        budget_file = st.file_uploader("Upload Budget CSV (with 'Amount' column)", type=["csv"])
        submitted = st.form_submit_button("Submit Proposal")
        if submitted:
            if not pdf_file or not budget_file:
                st.error("Please upload both PDF and Budget CSV")
                return
            text = extract_pdf_text(pdf_file.read())
            if not text.strip():
                st.error("Could not extract text from PDF")
                return
            try:
                budget_df = pd.read_csv(budget_file)
            except:
                st.error("Error reading budget CSV")
                return
            if "Amount" not in budget_df.columns:
                st.error("Budget CSV must include 'Amount' column")
                return
            scores = compute_weighted_score(text, budget_df)
            proposal = {
                "id": len(st.session_state.proposals)+1,
                "text": text,
                "scores": scores,
                "user": st.session_state.username,
                "status": scores["Status"],
                "eval_comment": ""
            }
            st.session_state.proposals.append(proposal)
            st.success(f"Proposal submitted with status: {scores['Status']}")

    st.subheader("Your proposals")
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
            st.info(f"Evaluator Comment: {p['eval_comment']}")

def plot_status_distribution(proposals):
    from collections import Counter
    status_counts = Counter(p["status"] for p in proposals)
    statuses = ["Accepted", "Conditional Acceptance (Revision Needed)", "Rejected"]
    counts = [status_counts.get(s, 0) for s in statuses]

    fig, ax = plt.subplots()
    ax.bar(statuses, counts, color=['green', 'orange', 'red'])
    ax.set_title("Proposal Status Distribution")
    ax.set_ylabel("Number of Proposals")
    return fig

def plot_average_scores(proposals):
    criteria = ["Relevance", "Novelty", "Technical Feasibility",
                "Financial Viability", "Impact", "Institutional Capability", "Compliance"]
    if not proposals:
        st.info("No proposals to plot scores.")
        return None

    avg_scores = {c: np.mean([p["scores"][c] for p in proposals]) for c in criteria}
    values = list(avg_scores.values())
    angles = np.linspace(0, 2 * np.pi, len(criteria), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
    ax.plot(angles, values, 'o-', linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), criteria)
    ax.set_ylim(0, 100)
    ax.set_title("Average Proposal Scores Radar Chart")
    return fig

def admin_dashboard():
    st.header(f"Welcome, {st.session_state.username} (Admin)")
    proposals = st.session_state.proposals
    #proposals = 2
    
    total_props = len(proposals)
    accepted = len([p for p in proposals if p["status"] == "Accepted"])
    conditional = len([p for p in proposals if "Conditional" in p["status"]])
    rejected = len([p for p in proposals if p["status"] == "Rejected"])

    st.subheader("Summary Statistics")
    st.write(f"Total proposals: {total_props}")
    st.write(f"Accepted: {accepted}")
    st.write(f"Conditional Acceptance: {conditional}")
    st.write(f"Rejected: {rejected}")

    st.subheader("Graphs")
    fig1 = plot_status_distribution(proposals)
    if fig1:
        st.pyplot(fig1)

    fig2 = plot_average_scores(proposals)
    if fig2:
        st.pyplot(fig2)

    st.subheader("Proposals with Alerts")
    alerts = [p for p in proposals if p["scores"]["Reasons"]]
    if alerts:
        for p in alerts:
            st.markdown(f"**Proposal #{p['id']}** by {p['user']} - Status: {p['status']}")
            st.error("Alerts: " + ", ".join(p["scores"]["Reasons"]))
    else:
        st.write("No alerts.")

    st.subheader("All Proposals Detail")
    for p in proposals:
        st.markdown(f"**Proposal #{p['id']}** by {p['user']} - Status: {p['status']}")
        for k, v in p["scores"].items():
            if k not in ["Reasons", "Status"]:
                st.write(f"{k}: {v}")
        if p["eval_comment"]:
            st.info(f"Evaluator Comment: {p['eval_comment']}")

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
        r = st.session_state.role
        if r == "company":
            company_dashboard()
        elif r == "admin":
            admin_dashboard()
        elif r == "evaluator":
            evaluator_dashboard()

if __name__ == "__main__":
    main()
