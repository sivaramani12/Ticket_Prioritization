import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

model = joblib.load("src/ticket_priority_model.pkl")
vectorizer = joblib.load("src/tfidf_vectorizer.pkl")
label_encoder = joblib.load("src/label_encoder.pkl")

HISTORY_FILE = "src/ticket_history.csv"
COLUMNS = ["ID", "Date", "Type", "Queue", "Subject", "Description", "Predicted Priority"]

if os.path.exists(HISTORY_FILE):
    history_df = pd.read_csv(HISTORY_FILE)
    if "Confidence" in history_df.columns:
        history_df.drop(columns=["Confidence"], inplace=True)
        history_df.to_csv(HISTORY_FILE, index=False)
else:
    history_df = pd.DataFrame(columns=COLUMNS)


st.set_page_config(page_title="Smart Ticket Prioritizer", layout="wide")
page = st.sidebar.radio("Select Page", ["Predict Ticket Priority", "Analytics Dashboard"])


if page == "Predict Ticket Priority":
    st.title("Smart Ticket Prioritization System")
    st.markdown("Fill out the ticket details to predict its urgency level (P1, P2, P3).")

    st.sidebar.title("Ticket Details")
    ticket_type = st.sidebar.selectbox("Type", ["Incident", "Request", "Problem", "Change"])
    ticket_queue = st.sidebar.selectbox("Queue", [
        "Technical support", "IT support", "Billing and payments",
        "Customer service", "General enquiry", "Product support"
    ])
    ticket_subject = st.sidebar.text_input("Subject")
    ticket_description = st.sidebar.text_area("Description")

    if st.sidebar.button("Predict Priority"):
        if not all([ticket_type, ticket_queue, ticket_subject, ticket_description]):
            st.warning("Please complete all fields.")
        else:
            combined_input = (
                ticket_type.lower() + "+" +
                ticket_queue.lower() + "+" +
                ticket_subject.lower() + "+" +
                ticket_description.lower()
            )

            X_input = vectorizer.transform([combined_input])
            proba = model.predict_proba(X_input)[0]
            top_class = label_encoder.inverse_transform([proba.argmax()])[0]

            col1, col2 = st.columns([1, 1])
            with col1:
                st.metric("Predicted Priority", top_class)

            with col2:
                st.markdown("### Priority Level Gauge")
                priority_to_value = {"P1": 0, "P2": 50, "P3": 100}
                needle_value = priority_to_value[top_class]

                fig = go.Figure(go.Indicator(
                    mode="gauge",
                    value=needle_value,
                    #number={'prefix': f"{top_class}"},
                    title={'text': f"Predicted Priority: {top_class}", 'font': {'size': 24}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickvals': [0, 50, 100], 'ticktext': ['P1', 'P2', 'P3']},
                        'bar': {'color': "black", 'thickness': 0.1},
                        'steps': [
                            {'range': [0, 33], 'color': "#FF4B4B"},
                            {'range': [33, 66], 'color': "#F9C74F"},
                            {'range': [66, 100], 'color': "#90BE6D"},
                        ]
                    },
                    domain={'x': [0, 1], 'y': [0, 1]}
                ))
                fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=400)
                st.plotly_chart(fig, use_container_width=True)

            new_id = int(history_df["ID"].max()) + 1 if not history_df.empty else 1
            new_record = pd.DataFrame([[
                new_id,
                datetime.now().strftime("%Y-%m-%d %H:%M"),
                ticket_type,
                ticket_queue,
                ticket_subject,
                ticket_description,
                top_class
            ]], columns=COLUMNS)

            history_df = pd.concat([history_df, new_record], ignore_index=True)
            history_df.to_csv(HISTORY_FILE, index=False)

    if not history_df.empty:
        st.markdown("### Recent Ticket Predictions")
        st.dataframe(history_df.tail(10)[::-1], use_container_width=True)


elif page == "Analytics Dashboard":
    st.title("Ticket Analytics Dashboard")

    if history_df.empty:
        st.info("No ticket data yet. Try making predictions.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Tickets Raised", len(history_df))
            st.metric("Unique Dates", history_df["Date"].str[:10].nunique())
        with col2:
            st.metric("Most Frequent Priority", history_df["Predicted Priority"].mode()[0])

        st.markdown("### Tickets by Date and Priority")
        date_df = history_df.copy()
        date_df["Date"] = pd.to_datetime(date_df["Date"]).dt.date
        fig_bar = px.histogram(
            date_df,
            x="Date",
            color="Predicted Priority",
            barmode="group",
            title="Tickets per Day",
            color_discrete_map={"P1": "#FF4B4B", "P2": "#F9C74F", "P3": "#90BE6D"}
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("### Priority Distribution")
        fig_pie = px.pie(
            history_df,
            names="Predicted Priority",
            title="Overall Priority Distribution",
            color="Predicted Priority",
            color_discrete_map={"P1": "#FF4B4B", "P2": "#F9C74F", "P3": "#90BE6D"}
        )
        st.plotly_chart(fig_pie, use_container_width=True)
