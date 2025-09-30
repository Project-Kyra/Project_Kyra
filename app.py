import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# In-memory user db (user:pass:role)
users = [
    {"username": "admin", "password": "admin123", "role": "admin"},
    {"username": "company1", "password": "comp123", "role": "company"},
    {"username": "evaluator1", "password": "eval123", "role": "evaluator"}
]

# Session
if "login_state" not in st.session_state:
    st.session_state["login_state"] = None
if "username" not in st.session_state:
    st.session_state["username"] = None
if "proposals" not in st.session_state:
    st.session_state["proposals"] = []

# Novelty benchmark
benchmark_projects = [
    "Optimized coal mining using IoT sensors.",
    "Advanced rare earth extraction from coal ash.",
    "Novel AI techniques for predictive maintenance in mines."
]

def score_novelty(text):
    all_abstracts = benchmark_projects + [text]
    v = TfidfVectorizer().fit_transform(all_abstracts)
    cosine_sim = (v[-1] @ v[:-1].T).toarray()[0]
    novelty = 1 - np.max(cosine_sim)
    return round(novelty*100, 2)

def score_feasibility(text):
    keywords = ['AI', 'ML', 'sensors', 'smart mining', 'automation', 'sustainability']
    score = sum(1 for k in keywords if k.lower() in text.lower()) * 20
    return min(score, 100)

def check_budget(budget_df):
    max_total = 2000000
    milestone_limit = 0.4
    total = budget_df['Amount'].sum()
    first_milestone = budget_df.iloc[0]['Amount'] if not budget_df.empty else 0
    compliance = (total <= max_total) and (first_milestone / total <= milestone_limit)
    issues = []
    if total > max_total: issues.append("Total budget exceeds guideline!")
    if (total > 0) and (first_milestone / total > milestone_limit): issues.append("First milestone exceeds allowed percentage!")
    return int(compliance)*100, issues

def authenticate(username, password):
    for u in users:
        if u["username"] == username and u["password"] == password:
            return u["role"]
    return None

st.title("NaCCER R&D Proposal Evaluation Demo")

if st.session_state["login_state"] is None:
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        role = authenticate(username, password)
        if role:
            st.session_state["login_state"] = role
            st.session_state["username"] = username
            st.success(f"Logged in as {role}")
            st.experimental_rerun()
        else:
            st.error("Invalid credentials")
    st.stop()

role = st.session_state["login_state"]
username = st.session_state["username"]

if role == "company":
    st.header("Company Dashboard")
    with st.form("proposal_form"):
        title = st.text_input("Proposal Title")
        abstract = st.text_area("Proposal Abstract")
        budget_file = st.file_uploader("Upload Budget CSV", type=["csv"])
        submit = st.form_submit_button("Submit Proposal")
        if submit and title and abstract and budget_file:
            budget_df = pd.read_csv(budget_file)
            novelty_score = score_novelty(abstract)
            feas_score = score_feasibility(abstract)
            finance_score, issues = check_budget(budget_df)
            # Assign to evaluator for demo (always "evaluator1", id=2)
            prop = {
                "id": len(st.session_state["proposals"])+1,
                "title": title,
                "abstract": abstract,
                "novelty": novelty_score,
                "feasibility": feas_score,
                "finance_score": finance_score,
                "budget": budget_df.to_dict(),
                "alerts": issues,
                "user": username,
                "status": "Submitted",
                "evaluator": "evaluator1",
                "eval_comment": ""
            }
            st.session_state["proposals"].append(prop)
            st.success("Proposal submitted. Evaluation scores below:")
            st.write(f"**Novelty Score**: {novelty_score}/100")
            st.write(f"**Feasibility Score**: {feas_score}/100")
            st.write(f"**Financial Compliance Score**: {finance_score}/100")
            if issues: 
                st.warning("Financial Compliance Issues:")
                for issue in issues:
                    st.write(f"- {issue}")

    st.subheader("Your Proposals")
    for p in st.session_state["proposals"]:
        if p["user"]==username:
            st.markdown(f"**{p['title']}** – Novelty: {p['novelty']} | Feasibility: {p['feasibility']} | Finance: {p['finance_score']} | Status: {p['status']}")
            if p['eval_comment']:
                st.info(f"Evaluator: {p['evaluator']} | Comments: {p['eval_comment']}")
            if p['alerts']:
                st.error(f"Alerts: {' | '.join(p['alerts'])}")

elif role == "admin":
    st.header("Admin Dashboard")
    st.subheader("All Proposals")
    flagged_props = []
    for p in st.session_state["proposals"]:
        st.markdown(f"**{p['title']}** by {p['user']}<br>Novelty: {p['novelty']} | Feasibility: {p['feasibility']} | Finance: {p['finance_score']} | Status: {p['status']}", unsafe_allow_html=True)
        if p['alerts']:
            flagged_props.append(p)
            st.error(f"ALERT: {' | '.join(p['alerts'])}")
        if p['eval_comment']:
            st.info(f"Evaluator: {p['evaluator']} | Comments: {p['eval_comment']}")
    if flagged_props:
        st.subheader("Flagged Proposals for Review")
        for p in flagged_props:
            st.write(f"{p['title']} by {p['user']} - Issues: {', '.join(p['alerts'])}")
    st.subheader("User List")
    for u in users:
        st.write(f"{u['username']} ({u['role']})")

elif role == "evaluator":
    st.header("Evaluator Dashboard")
    # Only proposals assigned to this evaluator
    my_props = [p for p in st.session_state["proposals"] if p["evaluator"]==username]
    if my_props:
        for i, p in enumerate(my_props):
            st.markdown(f"**{p['title']}** by {p['user']}<br>Novelty: {p['novelty']} | Feasibility: {p['feasibility']} | Finance: {p['finance_score']} | Status: {p['status']}", unsafe_allow_html=True)
            if st.button(f"Review #{p['id']}"):
                comment = st.text_area(f"Evaluator Comments for {p['title']}", value=p['eval_comment'])
                status = st.selectbox("Mark status", ["Accepted", "Rejected", "Needs Revision"], index=["Accepted", "Rejected", "Needs Revision"].index(p['status']) if p['status'] in ["Accepted", "Rejected", "Needs Revision"] else 0)
                if st.button(f"Submit Review #{p['id']}"):
                    p['eval_comment'] = comment
                    p['status'] = status
                    st.success("Review submitted")
    else:
        st.info("No proposals assigned.")

if st.button("Logout"):
    st.session_state["login_state"] = None
    st.session_state["username"] = None
    st.experimental_rerun()
