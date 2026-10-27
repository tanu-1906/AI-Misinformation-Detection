
import email
import streamlit as st
import pickle
import re
import pandas as pd
import os
from supabase import create_client
from supabase.client import ClientOptions
from dotenv import load_dotenv

# Load Supabase credentials
load_dotenv()

try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
except Exception:
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")

@st.cache_resource
def get_supabase_client():
    return create_client(
        SUPABASE_URL,
        SUPABASE_KEY,
        options=ClientOptions(
            flow_type="pkce"
        )
    )

supabase = get_supabase_client()
# Load trained model
with open("misinformation_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("tfidf_vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)


# Text cleaning function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# Page configuration
st.set_page_config(
    page_title="AI Misinformation Detector",
    page_icon="🔍",
    layout="wide"
)

# ===== PROFESSIONAL LIGHT UI =====

st.markdown("""
<style>

/* Main background */
.stApp {
    background-color: #f5f7fb;
    color: #1e293b;
}

/* Main content */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

/* Headings */
h1, h2, h3, h4 {
    color: #172554 !important;
    font-weight: 700 !important;
}

/* Normal text */
p, label, li,
[data-testid="stMarkdownContainer"] p {
    color: #334155 !important;
}

/* Input fields */
.stTextInput input,
.stTextArea textarea {
    background-color: #ffffff !important;
    color: #1e293b !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 8px !important;
}

/* Dropdown */
[data-baseweb="select"] > div {
    background-color: #ffffff !important;
    color: #1e293b !important;
    border-color: #cbd5e1 !important;
}

/* Buttons */
.stButton > button {
    background-color: #2563eb !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.55rem 1rem !important;
}

/* Button hover */
.stButton > button:hover {
    background-color: #1d4ed8 !important;
    color: #ffffff !important;
}

/* Download button */
.stDownloadButton > button {
    background-color: #ffffff !important;
    color: #1d4ed8 !important;
    border: 1px solid #2563eb !important;
    border-radius: 8px !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background-color: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    padding: 18px !important;
    border-radius: 12px !important;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.05);
}

[data-testid="stMetricLabel"],
[data-testid="stMetricValue"] {
    color: #172554 !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #ffffff !important;
}

/* Sidebar text */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span {
    color: #334155 !important;
}

/* Tables */
[data-testid="stDataFrame"] {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
}

/* Dividers */
hr {
    border-color: #e2e8f0 !important;
}

/* Alerts */
[data-testid="stAlert"] {
    border-radius: 10px;
}

/* Responsive */
@media (max-width: 768px) {
    .block-container {
        padding: 1rem;
    }
}

/* Login and Google authentication buttons */
.stButton button,
.stLinkButton a {
    background-color: #2563eb !important;
    color: #ffffff !important;
    border: 1px solid #2563eb !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}

/* Force button text to remain visible */
.stButton button p,
.stButton button span,
.stLinkButton a p,
.stLinkButton a span {
    color: #ffffff !important;
}

/* Google OAuth button - if rendered as a link */
a[href*="google"],
a[href*="oauth"] {
    color: #1d4ed8 !important;
}

/* ===== LOGIN PAGE PROFESSIONAL DESIGN ===== */

/* Centered authentication content */
div[data-testid="stVerticalBlock"]:has(
    div[data-testid="stTabs"]
) {
    gap: 1rem;
}

/* Login and signup tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 20px;
    border-bottom: 1px solid #dbe3ef;
}

.stTabs [data-baseweb="tab"] {
    color: #334155 !important;
    font-weight: 600;
    padding: 12px 18px;
}

.stTabs [aria-selected="true"] {
    color: #2563eb !important;
}

/* Login inputs */
.stTextInput input {
    min-height: 42px;
    border-radius: 9px !important;
    border: 1px solid #cbd5e1 !important;
    background-color: #ffffff !important;
    color: #1e293b !important;
}

/* Login buttons */
.stButton > button {
    min-height: 42px;
    border-radius: 9px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease;
}

/* Focus effect */
.stTextInput input:focus {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.12);
}

/* ===== END LOGIN PAGE DESIGN ===== */

</style>
""", unsafe_allow_html=True)

# ===== END PROFESSIONAL LIGHT UI =====



# Custom CSS
st.markdown("""
<style>
.stApp {
    background-color: #f5f7fb;
}

.main-title {
    font-size: 38px;
    font-weight: 800;
    color: #172554;
    text-align: center;
}

.subtitle {
    text-align: center;
    color: #64748b;
    font-size: 16px;
}

.result-card {
    padding: 22px;
    border-radius: 15px;
    background-color: white;
    border: 1px solid #e2e8f0;
    text-align: center;
}

div.stButton > button {
    width: 100%;
    background-color: #2563eb;
    color: white;
    border-radius: 10px;
    height: 45px;
    font-weight: bold;
}

/* ===== PROFESSIONAL DASHBOARD ===== */

/* Section headings */
h2, h3 {
    color: #17336b !important;
    font-weight: 700 !important;
}

/* Dashboard metric cards */
[data-testid="stMetric"] {
    background: #ffffff !important;
    border: 1px solid #dbe5f2 !important;
    border-radius: 14px !important;
    padding: 20px !important;
    box-shadow: 0 3px 10px rgba(30, 64, 175, 0.06);
}

/* Metric labels */
[data-testid="stMetricLabel"] {
    color: #475569 !important;
    font-size: 14px !important;
}

/* Metric values */
[data-testid="stMetricValue"] {
    color: #1d4ed8 !important;
    font-size: 30px !important;
    font-weight: 700 !important;
}

/* Text area */
.stTextArea textarea {
    background: #ffffff !important;
    color: #1e293b !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 12px !important;
    padding: 14px !important;
}

/* Section separators */
hr {
    margin-top: 25px !important;
    margin-bottom: 25px !important;
    border-color: #dbe5f2 !important;
}

/* Sidebar welcome message */
[data-testid="stSidebar"] [data-testid="stAlert"] {
    border-radius: 12px !important;
}

/* Dashboard spacing */
div[data-testid="stVerticalBlock"] {
    gap: 1rem;
}

/* ===== END DASHBOARD DESIGN ===== */

</style>
""", unsafe_allow_html=True)
# Google OAuth Callback

if "code" in st.query_params:
    auth_code = st.query_params["code"]

    try:
        response = supabase.auth.exchange_code_for_session({
            "auth_code": auth_code
        })

        st.session_state.user = response.user
        st.query_params.clear()

        st.success("Google login successful!")
        st.rerun()

    except Exception as e:
        st.error("Google login failed.")
        st.exception(e)

# ===== AUTHENTICATION UI =====

if "user" not in st.session_state:
    st.session_state.user = None

if st.session_state.user is None:

    st.markdown("""
    <div style="text-align: center; padding: 12px 0 22px 0;">
        <div style="font-size: 46px;">🛡️</div>
        <h1 style="color: #172554; font-size: 34px;">
            AI Misinformation Detector
        </h1>
        <p style="color: #64748b;">
            Sign in to analyze news and detect misinformation
        </p>
    </div>
    """, unsafe_allow_html=True)

    login_tab, signup_tab = st.tabs(["Login", "Sign Up"])

    with login_tab:
        email = st.text_input("Email", key="login_email")

        password = st.text_input(
            "Password",
            type="password",
            key="login_password"
        )

        if st.button("Login", key="login_button"):
            try:
                response = supabase.auth.sign_in_with_password({
                    "email": email,
                    "password": password
                })

                if response.user:
                    st.session_state.user = response.user
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Login failed. Please try again.")

            except Exception as e:
                st.error(f"Login failed: {e}")

    with signup_tab:
        signup_email = st.text_input(
            "Email",
            key="signup_email"
        )

        signup_password = st.text_input(
            "Password",
            type="password",
            key="signup_password"
        )

        if st.button("Create Account", key="signup_button"):
            try:
                response = supabase.auth.sign_up({
                    "email": signup_email,
                    "password": signup_password
                })

                st.success(
                    "Account created! Please check your email."
                )

            except Exception as e:
                st.error(f"Signup failed: {e}")

    if st.button("Continue with Google", key="google_login"):
        try:
            response = supabase.auth.sign_in_with_oauth({
                "provider": "google",
                "options": {
                    "redirect_to": "http://localhost:8501"
                }
            })

            st.markdown(f"[Continue with Google]({response.url})")

        except Exception as e:
            st.error(f"Google login failed: {e}")

# ===== END AUTHENTICATION UI =====

# Sidebar and Logout

user = st.session_state.user

with st.sidebar:
    st.title("🔍 AI Detector")

    if user:
        st.success("Logged in successfully")
        st.write(f"Welcome, {user.email}")

        if st.button("🚪 Logout", key="logout_button"):
            try:
                supabase.auth.sign_out()
            except Exception:
                pass

            st.session_state.user = None
            st.rerun()



# Main heading
st.markdown(
    '<div class="main-title">AI-Based Social Media '
    'Misinformation Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Analyze news and social media '
    'posts using Machine Learning</div>',
    unsafe_allow_html=True
)

st.write("")
st.divider()


# Input section

st.markdown("""
<div style="
    background: linear-gradient(110deg, #eaf2ff, #ffffff);
    padding: 20px;
    border: 1px solid #dbeafe;
    border-radius: 14px;
    margin-bottom: 18px;
">
    <h2 style="color: #17336b; margin: 0;">
        📰 News Analysis
    </h2>
    <p style="color: #475569; margin: 8px 0 0 0;">
        Analyze news and social media posts using machine learning.
    </p>
</div>
""", unsafe_allow_html=True)

news_text = st.text_area(
    "Enter News / Social Media Post",
    placeholder="Paste your news or social media post here...",
    height=220
)

analyze = st.button("🔍 Analyze News")


# Prediction



if analyze:
    if not news_text.strip():
        st.warning("Please enter some news text.")
    else:
        cleaned_news = clean_text(news_text)
        news_vector = vectorizer.transform([cleaned_news])

        prediction = model.predict(news_vector)[0]
        probabilities = model.predict_proba(news_vector)[0]
        st.write("DEBUG - Model classes:", model.classes_)
        st.write("DEBUG - Prediction:", prediction)
        st.write("DEBUG - Probabilities:", model.predict_proba(news_vector)[0])

        fake_probability = 0.0
        real_probability = 0.0

        for label, prob in zip(model.classes_, probabilities):
            label_text = str(label).strip().lower()

            if label_text in ["0", "fake", "false", "unreliable"]:
                fake_probability = float(prob) * 100
            elif label_text in ["1", "real", "true", "reliable"]:
                real_probability = float(prob) * 100

        confidence = max(probabilities) * 100

        prediction_text = str(prediction).strip().lower()

        if prediction_text in ["0", "fake", "false", "unreliable"]:
            result_label = "Fake"
        elif prediction_text in ["1", "real", "true", "reliable"]:
            result_label = "Real"
        else:
            result_label = "Unknown"

        st.subheader("📊 Analysis Results")
        st.metric("Model Confidence", f"{confidence:.2f}%")

        if result_label == "Fake":
            st.error("⚠️ Model Prediction: Potentially Misleading")
        elif result_label == "Real":
            st.success("✅ Model Prediction: Likely Reliable")
        else:
            st.warning("⚠️ Model Prediction: Unknown")

        st.write(f"Fake Probability: {fake_probability:.2f}%")
        st.write(f"Real Probability: {real_probability:.2f}%")

        with st.expander("🔧 Model Debug Information"):
            st.write("Model classes:", list(model.classes_))
            st.write("Predicted label:", prediction)
            st.write("Cleaned text:", cleaned_news)

        st.caption(
            "Model predictions are estimates, not proof of factual accuracy. "
            "Verify important news using trusted sources."
        )

        # Save prediction to Supabase
        import time

        try:
            user = st.session_state.user

            if user:

                history_data = {
                    "user_id": str(user.id),
                    "news_text": news_text,
                    "prediction": result_label,
                    "confidence": round(float(confidence), 2),
                    "fake_probability": round(float(fake_probability), 2),
                    "real_probability": round(float(real_probability), 2)
                }

                saved = False
                for attempt in range(3):
                    try:
                        supabase.table("prediction_history").insert(
                            history_data
                        ).execute()
                        saved = True
                        break
                    except Exception:
                        if attempt < 2:
                            time.sleep(2)
                        else:
                            raise

                if saved:
                    st.success("Prediction saved to history!")

        except Exception as e:
            st.warning(f"Could not save prediction: {e}")

       

# Dashboard Analytics

if st.session_state.user:
    st.divider()
    st.subheader("📊 Dashboard Analytics")

    try:
        user_id = str(st.session_state.user.id)

        analytics_response = (
            supabase.table("prediction_history")
            .select("prediction")
            .eq("user_id", user_id)
            .execute()
        )

        records = analytics_response.data or []

        total_predictions = len(records)
        fake_count = sum(
            1 for row in records
            if row["prediction"] == "Fake"
        )
        real_count = sum(
            1 for row in records
            if row["prediction"] == "Real"
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "📰 Total Predictions",
            total_predictions
        )

        col2.metric(
            "⚠️ Fake News",
            fake_count
        )

        col3.metric(
            "✅ Real News",
            real_count
        )

        if total_predictions > 0:
            st.subheader("📈 Fake vs Real News")

            analytics_df = pd.DataFrame({
                "Category": ["Fake News", "Real News"],
                "Count": [fake_count, real_count]
            })

            st.bar_chart(
                analytics_df.set_index("Category")
            )

            st.caption(
                "These counts represent model predictions, "
                "not independently verified facts."
            )
        else:
            st.info(
                "Analyze some news to see your dashboard analytics."
            )

    except Exception as e:
        st.error(f"Could not load analytics: {e}")



# Prediction History

if st.session_state.user:

    if st.session_state.get("history_delete_message"):
        st.success(st.session_state["history_delete_message"])
        del st.session_state["history_delete_message"]

    st.divider()
    st.subheader("📜 Prediction History")
    st.write("View your previously analyzed news and results.")

    if st.button("📂 View Prediction History"):
        st.session_state["show_prediction_history"] = True

    if st.session_state.get("show_prediction_history", False):

        try:
            user_id = str(st.session_state.user.id)

            response = (
                supabase.table("prediction_history")
                .select(
                    "id, news_text, prediction, confidence, "
                    "fake_probability, real_probability, created_at"
                )
                .eq("user_id", user_id)
                .order("created_at", desc=True)
                .execute()
            )

            history = response.data or []

            if history:

                history_df = pd.DataFrame(history)

                display_df = history_df.drop(columns=["id"])
                
                # Search and Filter

                st.subheader("🔎 Search & Filter History")

                search_text = st.text_input(
                    "Search news",
                    placeholder="Enter keywords to search...",
                    key="history_search"
                )

                prediction_filter = st.selectbox(
                    "Filter by Prediction",
                    ["All", "Fake", "Real"],
                    key="history_prediction_filter"
                )

                filtered_df = history_df.copy()

                if search_text:
                    filtered_df = filtered_df[
                        filtered_df["news_text"].str.contains(
                            search_text,
                            case=False,
                            na=False,
                            regex=False
                        )
                    ]

                if prediction_filter != "All":
                    filtered_df = filtered_df[
                        filtered_df["prediction"].str.lower()
                        == prediction_filter.lower()
                    ]

                st.write(
                    f"Matching records: {len(filtered_df)}"
                )

                display_df = filtered_df.drop(
                    columns=["id"]
                )


                display_df = display_df.rename(columns={
                    "news_text": "News Text",
                    "prediction": "Prediction",
                    "confidence": "Confidence (%)",
                    "fake_probability": "Fake Probability (%)",
                    "real_probability": "Real Probability (%)",
                    "created_at": "Date & Time"
                })

                csv_data = display_df.to_csv(
                    index=False
                ).encode("utf-8-sig")

                st.download_button(
                    label="⬇️ Download History as CSV",
                    data=csv_data,
                    file_name="prediction_history.csv",
                    mime="text/csv",
                    key="download_history_csv"
                )

                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True
                )

                st.success(
                    f"Found {len(history)} prediction record(s)."
                )

                st.subheader("🗑️ Delete a Prediction")

                record_options = {
                    (
                        f"ID {row['id']} — "
                        f"{row['prediction']} — "
                        f"{str(row['news_text'])[:60]}"
                    ): row["id"]
                    for row in history
                }

                selected_record = st.selectbox(
                    "Select the record to delete",
                    options=list(record_options.keys()),
                    key="delete_history_select"
                )

                if st.button(
                    "🗑️ Delete Selected Record",
                    key="delete_selected_record"
                ):
                    try:
                        selected_id = record_options[selected_record]

                        delete_response = (
                            supabase.table("prediction_history")
                            .delete()
                            .eq("id", selected_id)
                            .eq("user_id", user_id)
                            .execute()
                        )

                        st.session_state["history_delete_message"] = (
                            "Delete request completed. "
                            "Please check the updated history."
                        )

                        st.rerun()

                    except Exception as e:
                        st.error(
                            f"Could not delete record: {e}"
                        )

            else:
                st.info("No prediction history found yet.")

        except Exception as e:
            st.error(
                f"Could not load prediction history: {e}"
            )


# Footer

st.divider()
st.caption(
    "AI-Based Social Media Misinformation Detection System | "
    "Machine Learning Project"
)
