import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

DATA_PATH = "augmented_balanced_dataset_m.csv"
FEATURES = [
    "Submission.Actions",
    "Quiz.Actions",
    "Forum.Actions",
    "Create.Actions",
    "Read.Actions",
    "Update.Actions",
    "Delete.Actions",
    "Days.Logged.In",
]

st.set_page_config(page_title="EduSentry", page_icon="🎓", layout="wide")


@st.cache_resource(show_spinner="Training model...")
def load_model():
    """Same steps as the notebook; runs once and is cached."""
    df = pd.read_csv(DATA_PATH)
    X, y = df[FEATURES], df["rating"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=1000, random_state=44, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "report": pd.DataFrame(
            classification_report(y_test, y_pred, labels=["High", "Low"], output_dict=True)
        ).T,
        "cm": pd.DataFrame(
            confusion_matrix(y_test, y_pred, labels=["High", "Low"]),
            index=["Actual High", "Actual Low"],
            columns=["Predicted High", "Predicted Low"],
        ),
    }
    importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
    return model, metrics, importances


model, metrics, importances = load_model()
classes = list(model.classes_)  # ['High', 'Low']

st.title("🎓High or Low?: Early Student Engagement Monitor")
st.caption("Predicts whether a student's online engagement is High or Low from their LMS activity.")

tab_single, tab_batch, tab_model = st.tabs(["Single student", "Batch (CSV)", "Model info"])

# ---------- Single prediction ----------
with tab_single:
    st.subheader("Enter a student's activity")
    cols = st.columns(4)
    values = {}
    for i, feat in enumerate(FEATURES):
        with cols[i % 4]:
            values[feat] = st.number_input(feat, min_value=0, value=0, step=1)

    if st.button("Predict", type="primary"):
        row = pd.DataFrame([values])[FEATURES]
        pred = model.predict(row)[0]
        proba = model.predict_proba(row)[0]
        p = dict(zip(classes, proba))

        if pred == "Low":
            st.error(f"⚠️ Predicted engagement: **Low** (at risk), {p['Low']:.0%} confidence")
        else:
            st.success(f"✅ Predicted engagement: **High**, {p['High']:.0%} confidence")
        st.progress(float(p["Low"]), text=f"Probability of Low: {p['Low']:.1%}")

# ---------- Batch prediction ----------
with tab_batch:
    st.subheader("Upload weekly activity logs")
    st.write(f"CSV must contain these columns: `{'`, `'.join(FEATURES)}` (extra columns such as a student ID are kept).")
    up = st.file_uploader("CSV file", type="csv")
    if up:
        batch = pd.read_csv(up)
        missing = [c for c in FEATURES if c not in batch.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            batch["Prediction"] = model.predict(batch[FEATURES])
            batch["P(Low)"] = model.predict_proba(batch[FEATURES])[:, classes.index("Low")]
            batch = batch.sort_values("P(Low)", ascending=False)
            st.write(f"**{(batch['Prediction'] == 'Low').sum()}** of {len(batch)} students flagged as Low.")
            st.dataframe(batch, use_container_width=True)
            st.download_button(
                "Download results", batch.to_csv(index=False).encode(), "predictions.csv", "text/csv"
            )

# ---------- Model info ----------
with tab_model:
    st.subheader("Held-out test performance")
    st.metric("Test accuracy", f"{metrics['accuracy']:.1%}")
    c1, c2 = st.columns(2)
    with c1:
        st.write("Confusion matrix")
        st.dataframe(metrics["cm"])
        st.write("Classification report")
        st.dataframe(metrics["report"].round(2))
    with c2:
        st.write("Feature importance")
        st.bar_chart(importances)
