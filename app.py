import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Dummy users
USERS = {
    "admin": {"password": "admin123", "role": "admin"},
    "company1": {"password": "comp123", "role": "company"},
    "evaluator1": {"password": "eval123", "role": "evaluator"},
}

# Dummy benchmark abstracts for novelty check
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


def score_novelty(text: str) -> float:
    documents = BENCHMARKS + [text]
    tfidf = TfidfVectorizer().fit_transform(documents)
    cos_similarities = (tfidf[-1] @ tfidf[:-1].T).toarray()[0]
    novelty_score = 1 - np.max(cos_similarities)
    return round(novelty_score * 100, 2)


def score_feasibility(text: str) -> float:
    keywords = ["AI", "ML", "sensors", "automation", "safety", "sustainability"]
    score = sum(1 for kw in keywords if kw.lower() in text.lower()) * 20
    return min(score, 100)


def score_finance(budget_df: pd.DataFrame) -> (float, list):
    max_total = 2000000
    milestone_limit = 0.4
    total = budget_df["Amount"].sum()
    first_milestone = budget_df.iloc[0]["Amount"] if not budget_df.empty else 0
    issues = []
    compliant = True
    if total > max_total:
        issues.append("Total budget exceeds INR 20 lakhs")
        compliant = False
    if total > 0 and first_milestone / total > milestone_limit:
        issues.append("First milestone exceeds 40% of total budget")
        compliant = False
    return (100 if compliant else 50), issues


def login():
    st.title("NaCCER Proposal Evaluation System")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        user = USERS.get(username)
        if user and password == user["password"]:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.role = user["role"]
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
        title = st.text_input("Proposal Title", key="title")
        abstract = st.text_area("Proposal Abstract", key="abstract")
        budget_file = st.file_uploader("Upload Budget CSV (must have 'Amount' column)", type=["csv"], key="budget")
        submitted = st.form_submit_button("Submit Proposal")
        if submitted:
            if not title or not abstract:
                st.error("Please fill Title and Abstract")
                return
            if not budget_file:
                st.error("Please upload budget CSV")
                return
            budget_df = pd.read_csv(budget_file)
            if "Amount" not in budget_df.columns:
                st.error("CSV must contain 'Amount' column")
                return
            novelty = score_novelty(abstract)
            feasibility = score_feasibility(abstract)
            finance_score, issues = score_finance(budget_df)
            prop = {
                "id": len(st.session_state.proposals) + 1,
                "title": title,
                "abstract": abstract,
                "budget": budget_df.to_dict(),
                "novelty": novelty,
                "feasibility": feasibility,
                "finance_score": finance_score,
                "alerts": issues,
                "user": st.session_state.username,
                "status": "Submitted",
                "evaluator": "evaluator1",
                "eval_comment": "",
            }
            st.session_state.proposals.append(prop)
            st.success("Proposal submitted successfully!")
            st.write(f"Novelty Score: {novelty}/100")
            st.write(f"Feasibility Score: {feasibility}/100")
            st.write(f"Financial Compliance Score: {finance_score}/100")
            if issues:
                st.warning("Alerts:")
                for issue in issues:
                    st.write("- " + issue)
    st.subheader("Your Proposals")
    props = [p for p in st.session_state.proposals if p["user"] == st.session_state.username]
    for p in props:
        st.markdown(f"**{p['title']}** - Status: {p['status']}")
        st.write(f"Novelty: {p['novelty']} | Feasibility: {p['feasibility']} | Finance: {p['finance_score']}")
        if p["eval_comment"]:
            st.info(f"Evaluator Comments: {p['eval_comment']}")


def admin_dashboard():
    st.header(f"Welcome, {st.session_state.username} (Admin)")
    st.subheader("All Proposals")
    for p in st.session_state.proposals:
        st.markdown(
            f"**{p['title']}** by {p['user']} | Status: {p['status']}<br>"
            f"Novelty: {p['novelty']} | Feasibility: {p['feasibility']} | Finance: {p['finance_score']}",
            unsafe_allow_html=True,
        )
        if p["alerts"]:
            st.error("Alerts: " + ", ".join(p["alerts"]))
        if p["eval_comment"]:
            st.info(f"Evaluator: {p['evaluator']} | Comments: {p['eval_comment']}")
    st.subheader("Users")
    for u, v in USERS.items():
        st.write(f"{u} ({v['role']})")


def evaluator_dashboard():
    st.header(f"Welcome, {st.session_state.username} (Evaluator)")
    assigned = [p for p in st.session_state.proposals if p["evaluator"] == st.session_state.username]
    if not assigned:
        st.info("No proposals assigned.")
        return
    for p in assigned:
        st.markdown(
            f"**{p['title']}** by {p['user']} | Status: {p['status']}<br>"
            f"Novelty: {p['novelty']} | Feasibility: {p['feasibility']} | Finance: {p['finance_score']}",
            unsafe_allow_html=True,
        )
        st.text_area("Evaluator Comments", value=p["eval_comment"], key=f"comment_{p['id']}")
        status = st.selectbox(
            "Update Status",
            options=["Submitted", "Accepted", "Rejected", "Needs Revision"],
            index=["Submitted", "Accepted", "Rejected", "Needs Revision"].index(p["status"]),
            key=f"status_{p['id']}",
        )
        if st.button(f"Submit Review for Proposal #{p['id']}", key=f"submit_{p['id']}"):
            comment_key = f"comment_{p['id']}"
            p["eval_comment"] = st.session_state[comment_key]
            p["status"] = status
            st.success(f"Proposal #{p['id']} review updated.")


def main():
    if not st.session_state.logged_in:
        login()
    else:
        role = st.session_state.role
        if st.sidebar.button("Logout"):
            logout()
        st.sidebar.write(f"Logged in as: {st.session_state.username} ({role})")
        if role == "admin":
            admin_dashboard()
        elif role == "company":
            company_dashboard()
        elif role == "evaluator":
            evaluator_dashboard()


if __name__ == "__main__":
    main()
