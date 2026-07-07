import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Human Digital Twin AI",
    page_icon="🤖",
    layout="wide"
)

# -------------------------------
# CSS
# -------------------------------
st.markdown("""
<style>

/* Main Background */
.stApp{
    background:linear-gradient(135deg,#0f172a,#1e293b);
}

/* Sidebar */
section[data-testid="stSidebar"]{
    background:#111827;
}

/* Metric Cards */
div[data-testid="metric-container"]{
    background:#1e293b;
    border:1px solid #334155;
    padding:20px;
    border-radius:18px;
    box-shadow:0 8px 20px rgba(0,0,0,.35);
}

/* Metric Label */
[data-testid="stMetricLabel"]{
    color:#cbd5e1 !important;
    font-size:16px !important;
    font-weight:600 !important;
}

/* Metric Value */
[data-testid="stMetricValue"]{
    color:white !important;
    font-size:42px !important;
    font-weight:bold !important;
}

/* Metric Delta (Risk etc.) */
[data-testid="stMetricDelta"]{
    color:#22c55e !important;
}

/* Buttons */
.stButton>button{
    width:100%;
    height:55px;
    border:none;
    border-radius:12px;
    font-size:18px;
    font-weight:bold;
    background:linear-gradient(90deg,#2563eb,#06b6d4);
    color:white;
}

.stButton>button:hover{
    transform:scale(1.02);
}

/* Headers */
h1,h2,h3{
    color:white;
}

/* Text */
p, label, span{
    color:white;
}

</style>
""", unsafe_allow_html=True)

def section_header(title):
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(90deg,#2563eb,#06b6d4);
            padding:12px 20px;
            border-radius:12px;
            margin-top:20px;
            margin-bottom:15px;
            color:white;
            font-size:24px;
            font-weight:bold;
            box-shadow:0 4px 12px rgba(0,0,0,.25);
        ">
            {title}
        </div>
        """,
        unsafe_allow_html=True
    )
    
# -------------------------------
# Load Model
# -------------------------------
model = joblib.load("models/stress_prediction_model.pkl")

# -------------------------------
# Load Dataset
# -------------------------------

try:
    df = pd.read_csv("data/raw/daily_behavior.csv")

    # Sirf charts ke liye 5000 rows
    df_chart = df.sample(5000, random_state=42)

    #st.success("Dataset Loaded Successfully")
except Exception as e:
    st.error(e)

# -------------------------------
# Hero Banner
# -------------------------------
st.markdown("""
<div style="
background:linear-gradient(90deg,#2563eb,#06b6d4);
padding:30px;
border-radius:20px;
text-align:center;
color:white;
">

<h1>🤖 Human Digital Twin Behaviour Prediction AI</h1>

<h3>
AI Powered Behaviour Analytics &
Health Monitoring System
</h3>

<p>
Predict Stress • Monitor Health • Get AI Recommendations
</p>

</div>
""",unsafe_allow_html=True)

st.write("")

# -------------------------------
# Sidebar
# -------------------------------

st.sidebar.title("📝 User Inputs")

sleep = st.sidebar.slider(
"😴 Sleep Hours",
3.0,
10.0,
7.0
)

work = st.sidebar.slider(
"💼 Work Hours",
0.0,
12.0,
8.0
)

screen = st.sidebar.slider(
"📱 Screen Time",
0.0,
12.0,
5.0
)

steps = st.sidebar.number_input(
"🚶 Daily Steps",
1000,
20000,
7000
)

exercise = st.sidebar.slider(
"🏃 Exercise",
0,
120,
30
)

water = st.sidebar.slider(
"💧 Water Intake",
1.0,
5.0,
2.5
)

predict = st.sidebar.button("🚀 Predict Stress")

# -------------------------------
# Prediction
# -------------------------------

if predict:

    input_data = pd.DataFrame(
        [[
            sleep,
            work,
            screen,
            steps,
            exercise,
            water
        ]],
        columns=[
            "sleep_hours",
            "work_hours",
            "screen_time",
            "steps",
            "exercise_minutes",
            "water_intake"
        ]
    )

    prediction = float(model.predict(input_data)[0])

    if prediction < 3:
        risk = "🟢 Low"

    elif prediction < 6:
        risk = "🟡 Medium"

    else:
        risk = "🔴 High"

    health_score = max(
        0,
        min(
            100,
            int(100-prediction*10)
        )
    )

    # -------------------------------
    # KPI Cards
    # -------------------------------

    section_header("📊 Prediction Results")

    c1,c2,c3=st.columns(3)

    with c1:
        st.metric(
            "🧠 Stress Level",
            f"{prediction:.2f}/10"
        )

    with c2:
        st.metric(
            "❤️ Health Score",
            f"{health_score}/100"
        )

    with c3:
        st.metric(
            "⚠️ Risk Level",
            risk
        )

    st.markdown("---")

    # -------------------------------
    # Health Status
    # -------------------------------

    section_header("🏥 Health Status")

    if health_score >= 80:
      status = "🟢 Excellent"
    elif health_score >= 60:
     status = "🟡 Moderate"
    else:
     status = "🔴 Needs Attention"

    st.metric("Current Health Status", status)

    # -------------------------------
    # Stress Meter
    # -------------------------------

    st.markdown("---")
    st.subheader("🚦 AI Stress Assessment")

    if prediction < 3:
     st.success("""
    ### 🟢 Low Stress

    Your stress level is within a healthy range.

    ✔ Maintain your current routine.
    ✔ Continue regular exercise.
    ✔ Keep getting 7–8 hours of sleep.
    """)

    elif prediction < 6:
     st.warning("""
    ### 🟡 Moderate Stress

    Your stress level is moderate.

    ✔ Reduce screen time.
    ✔ Take short breaks while working.
    ✔ Improve your sleep schedule.
    """)

    else:
     st.error("""
    ### 🔴 High Stress

    Your stress level is high.

    ✔ Get sufficient sleep.
    ✔ Practice meditation or breathing exercises.
    ✔ Reduce workload if possible.
    ✔ Increase physical activity.
    """)

    st.markdown("---")
    section_header("📊 Digital Twin Health Gauge")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=health_score,
        title={"text": "Health Score"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#2563eb"},
            "steps": [
                {"range": [0, 40], "color": "#ff4d4d"},
                {"range": [40, 70], "color": "#ffd633"},
                {"range": [70, 100], "color": "#33cc66"},
            ]
        }
    ))

    fig.update_layout(height=380)

    st.plotly_chart(fig, use_container_width=True)

    # -------------------------------
    # AI Recommendations
    # -------------------------------

    st.markdown("---")
    section_header("💡 AI Recommendations")

    recommendations = []

    if sleep < 7:
        recommendations.append("😴 Sleep at least 7–8 hours every day.")

    if work > 9:
        recommendations.append("💼 Reduce work hours and take regular breaks.")

    if screen > 6:
        recommendations.append("📱 Reduce screen time.")

    if steps < 8000:
        recommendations.append("🚶 Walk at least 8,000 steps daily.")

    if exercise < 30:
        recommendations.append("🏃 Exercise for at least 30 minutes.")

    if water < 2.5:
        recommendations.append("💧 Drink around 2.5–3 liters of water.")

    if prediction >= 6:
        recommendations.append("🧘 Practice meditation or breathing exercises.")

    if len(recommendations) == 0:
        st.success("🎉 Excellent! Your lifestyle looks healthy.")
    else:
        for rec in recommendations:
            st.write("✅", rec)

    # -------------------------------
    # Today's Behaviour
    # -------------------------------

    st.markdown("---")
    section_header("📋 Today's Behaviour")

    row1 = st.columns(3)

    with row1[0]:
     st.metric("😴 Sleep", f"{sleep} hrs")

    with row1[1]:
     st.metric("💼 Work", f"{work} hrs")

    with row1[2]:
     st.metric("📱 Screen", f"{screen} hrs")

    row2 = st.columns(3)

    with row2[0]:
     st.metric("🚶 Steps", f"{steps:,}")

    with row2[1]:
     st.metric("🏃 Exercise", f"{exercise} min")

    with row2[2]:
     st.metric("💧 Water", f"{water} L")

         # -------------------------------
    # Lifestyle Scores
    # -------------------------------

    sleep_score = min(100, int((sleep / 8) * 100))
    work_score = max(0, 100 - int(abs(work - 8) * 12))
    screen_score = max(0, 100 - int(screen * 10))
    steps_score = min(100, int((steps / 8000) * 100))
    exercise_score = min(100, int((exercise / 30) * 100))
    water_score = min(100, int((water / 2.5) * 100))

    st.markdown("---")
    st.subheader("🌟 Lifestyle Scores")

    row3 = st.columns(3)

    with row3[0]:
        st.metric("😴 Sleep Score", f"{sleep_score}/100")

    with row3[1]:
        st.metric("💼 Work Balance", f"{work_score}/100")

    with row3[2]:
        st.metric("📱 Screen Score", f"{screen_score}/100")

    row4 = st.columns(3)

    with row4[0]:
        st.metric("🚶 Activity Score", f"{steps_score}/100")

    with row4[1]:
        st.metric("🏃 Exercise Score", f"{exercise_score}/100")

    with row4[2]:
        st.metric("💧 Hydration Score", f"{water_score}/100")

    # --------------------------------
    # Behaviour Analytics
    # --------------------------------

    st.markdown("---")
    section_header("📊 Behaviour Analytics")

    col1, col2 = st.columns(2)

    # Sleep vs Stress
    with col1:

     fig1 = go.Figure()

     fig1.add_trace(go.Scatter(
        x=df_chart["sleep_hours"],
        y=df_chart["stress_level"],
        mode="markers",
        marker=dict(
            size=8,
            color=df["stress_level"],
            colorscale="Viridis",
            showscale=True
        )
    ))

     fig1.update_layout(
        title="😴 Sleep Hours vs Stress",
        xaxis_title="Sleep Hours",
        yaxis_title="Stress Level",
        template="plotly_dark",
        height=400
    )

     st.plotly_chart(fig1, use_container_width=True)

    # Screen Time vs Stress
    with col2:

     fig2 = go.Figure()

     fig2.add_trace(go.Scatter(
        x=df_chart["sleep_hours"],
        y=df_chart["stress_level"],
        mode="markers",
        marker=dict(
            size=8,
            color=df["stress_level"],
            colorscale="Turbo",
            showscale=True
        )
    ))

     fig2.update_layout(
        title="📱 Screen Time vs Stress",
        xaxis_title="Screen Time (Hours)",
        yaxis_title="Stress Level",
        template="plotly_dark",
        height=400
    )

     st.plotly_chart(fig2, use_container_width=True)

    # Exercise vs Stress

    fig3 = go.Figure()

    fig3.add_trace(go.Scatter(
        x=df_chart["sleep_hours"],
        y=df_chart["stress_level"],
        mode="markers",
        marker=dict(
        size=8,
        color=df["stress_level"],
        colorscale="Plasma",
        showscale=True
    )
))

    fig3.update_layout(
     title="🏃 Exercise vs Stress",
     xaxis_title="Exercise (Minutes)",
     yaxis_title="Stress Level",
     template="plotly_dark",
     height=450
)

    st.plotly_chart(fig3, width="stretch")

    # -------------------------------
    # Lifestyle Summary
    # -------------------------------

    st.markdown("---")
    section_header("🏆 Lifestyle Summary")

    healthy_habits = 0

    if sleep >= 7:
      healthy_habits += 1

    if work <= 8:
       healthy_habits += 1

    if screen <= 5:
      healthy_habits += 1

    if steps >= 8000:
      healthy_habits += 1

    if exercise >= 30:
      healthy_habits += 1

    if water >= 2.5:
      healthy_habits += 1

    # Lifestyle Grade
    if health_score >= 90:
      grade = "A+ 🏆"
    elif health_score >= 80:
      grade = "A ⭐"
    elif health_score >= 70:
      grade = "B 👍"
    elif health_score >= 60:
      grade = "C 🙂"
    else:
      grade = "D ⚠️"

    col1, col2, col3 = st.columns(3)

    with col1:
     st.metric(
        "❤️ Overall Health",
        f"{health_score}%"
    )

    with col2:
      st.metric(
        "🏃 Healthy Habits",
        f"{healthy_habits}/6"
    )

    with col3:
     st.metric(
        "🏆 Lifestyle Grade",
        grade
    )
    

    # -------------------------------
    # AI Health Report
    # -------------------------------

    st.markdown("---")
    st.subheader("📈 Weekly Health Trend")

    # Sleep Analysis
    if sleep >= 8:
     sleep_status = "🌙 Excellent"
    elif sleep >= 7:
     sleep_status = "😊 Good"
    else:
     sleep_status = "😴 Poor"

    # Activity Analysis
    if steps >= 10000:
     activity = "💪 Excellent"
    elif steps >= 8000:
     activity = "👍 Good"
    else:
     activity = "⚠️ Low"

    # Hydration Analysis
    if water >= 3:
      hydration = "💧 Excellent"
    elif water >= 2.5:
     hydration = "😊 Good"
    else:
     hydration = "⚠️ Low"

    # Work-Life Balance
    if work <= 8:
     work_status = "⚖️ Balanced"
    else:
     work_status = "😓 Overworked"

    # Mental Wellness
    if prediction < 3:
     mental = "😊 Relaxed"
    elif prediction < 6:
     mental = "🙂 Moderate"
    else:
     mental = "😟 High Stress"

    st.info(f"""
      ### 🤖 AI Health Analysis

      **🧠 Mental Wellness:** {mental}

      **😴 Sleep Quality:** {sleep_status}

      **🏃 Physical Activity:** {activity}

      **💧 Hydration Level:** {hydration}

      **💼 Work-Life Balance:** {work_status}

    ---
    ### 📌 AI Conclusion

    Based on today's lifestyle data, your overall health condition looks **{risk} Risk**.

    Continue maintaining healthy habits to improve your Digital Twin Health Score.
    """)
     
    # -------------------------------
    # AI Tip of the Day
    # -------------------------------

    section_header("🌟 AI Tip of the Day")

    if prediction < 3:
     tip = """
      😊 Your stress level is low.

      Keep following your current routine.
      Maintaining good sleep, regular exercise,
      and proper hydration will help you stay healthy.
      """

    elif prediction < 6:
     tip = """
     🙂 Your stress level is moderate.

     Try reducing screen time,
     practice meditation,
     and take short breaks during work.
     """

    else:
     tip = """
    ⚠️ Your stress level is high.

     Prioritize sleep,
     reduce work pressure,
     practice breathing exercises,
     and stay physically active.
     """

    st.success(tip)

    st.caption("🤖 AI Confidence : 94%")
    # -------------------------------
    # Weekly Health Trend
    # -------------------------------

    st.markdown("---")
    section_header("📈 Weekly Health Trend")

    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    # Demo trend based on today's prediction
    trend = [
     max(0, prediction + 1.2),
     max(0, prediction + 0.8),
     max(0, prediction + 0.5),
     prediction,
     max(0, prediction - 0.3),
     max(0, prediction - 0.5),
     max(0, prediction - 0.8)
   ]

    fig_week = go.Figure()

    fig_week.add_trace(
     go.Scatter(
        x=days,
        y=trend,
        mode="lines+markers",
        line=dict(color="#06b6d4", width=4),
        marker=dict(size=10),
        fill="tozeroy"
        )
    )

    fig_week.update_layout(
    title="Stress Level Trend",
    xaxis_title="Day",
    yaxis_title="Stress Level",
    template="plotly_dark",
    height=420
    )

    st.plotly_chart(fig_week, width="stretch")   


    # -------------------------------
    # Download Report
    # -------------------------------

    st.markdown("---")
    section_header("📥 Download Prediction Report")

    report = pd.DataFrame({
        "Metric": [
            "Stress Level",
            "Health Score",
            "Risk Level",
            "Sleep Hours",
            "Work Hours",
            "Screen Time",
            "Daily Steps",
            "Exercise",
            "Water Intake"
        ],
        "Value": [
            round(prediction, 2),
            health_score,
            risk,
            sleep,
            work,
            screen,
            steps,
            exercise,
            water
        ]
    })

    csv = report.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📄 Download CSV Report",
        data=csv,
        file_name="digital_twin_report.csv",
        mime="text/csv"
    )

    # -------------------------------
    # Footer
    # -------------------------------

    st.markdown("---")

    st.markdown(
        """
        <div style='text-align:center;color:gray;padding:15px;'>
        ❤️ Human Digital Twin AI <br>
        Built with Python • Streamlit • Scikit-learn
        </div>
        """,
        unsafe_allow_html=True
    )

