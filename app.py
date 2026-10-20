import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from datetime import datetime, timedelta

# ページ設定
st.set_page_config(page_title="大学生向け メンタル不調・気分予報 AI", layout="wide")

st.title("🧠 大大学生向け メンタル不調・気分予報 AI")
st.write("連携スプレッドシートから取得したアンケート項目（特徴量）と Open-Meteo 気象データ（気圧変化ΔP）から、今後1週間のコンディションを予測します。")

# ---------------- サイドバー（スプレッドシート由来の入力項目） ----------------
st.sidebar.header("📋 アンケート入力項目 (Spreadsheet Features)")

# Colabでスプレッドシートから取得・処理していた特徴量項目
sleep_hours = st.sidebar.slider("睡眠時間 (sleep_hours)", 0.0, 12.0, 7.0, 0.5)
classes_count = st.sidebar.slider("本日の授業コマ数 (classes_count)", 0, 6, 2)
meal_status = st.sidebar.selectbox("食事状況 (meal_status)", ["しっかり摂取", "軽食のみ", "欠食"])
task_load = st.sidebar.slider("課題・バイト負荷 (task_load: 1~5)", 1, 5, 3)
parttime_hours = st.sidebar.slider("バイト時間 (parttime_hours)", 0.0, 10.0, 2.0, 0.5)

st.sidebar.subheader("📍 位置情報")
lat = st.sidebar.number_input("緯度 (latitude)", value=35.6762)
lon = st.sidebar.number_input("経度 (longitude)", value=139.6503)

# ---------------- 予測ロジック ----------------
def predict_mood(sleep_h, class_c, meal_s, task_l, pt_h, p_change):
    # 睡眠影響
    sleep_score = (sleep_h - 7.0) * 5.0
    # 授業コマ数影響
    class_score = -1.0 * class_c * 2.5
    # 食事状況影響
    meal_dict = {"しっかり摂取": 5.0, "軽食のみ": 0.0, "欠食": -8.0}
    meal_score = meal_dict.get(meal_s, 0.0)
    # 課題・バイト負荷 & バイト時間影響
    task_score = -1.0 * (task_l - 1) * 4.0
    pt_score = -1.0 * pt_h * 2.0
    # 気圧変化影響
    pressure_score = -1.0 * np.abs(p_change) * 2.0
    
    score = 75.0 + sleep_score + class_score + meal_score + task_score + pt_score + pressure_score
    return np.clip(score, 0, 100)

# ---------------- 気象データ取得 (Open-Meteo API) ----------------
@st.cache_data(ttl=3600)
def get_weather_forecast(latitude, longitude):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&hourly=surface_pressure&timezone=Asia%2FTokyo"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return pd.DataFrame({
                'time': pd.to_datetime(data['hourly']['time']),
                'pressure': data['hourly']['surface_pressure']
            })
    except Exception:
        pass
    return None

weather_df = get_weather_forecast(lat, lon)

if weather_df is not None:
    weather_df['date'] = weather_df['time'].dt.date
    daily_weather = weather_df.groupby('date')['pressure'].mean().reset_index()
    daily_weather['pressure_change'] = daily_weather['pressure'].diff().fillna(0)
    
    today = datetime.now().date()
    forecast_days = daily_weather[daily_weather['date'] >= today].head(7).copy()
    
    if len(forecast_days) > 0:
        scores = []
        for p_change in forecast_days['pressure_change']:
            scores.append(predict_mood(sleep_hours, classes_count, meal_status, task_load, parttime_hours, p_change))
            
        forecast_days['predicted_mood'] = scores
        
        st.subheader("🔮 今後1週間のメンタル・気分予測結果")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(forecast_days['date'].astype(str), forecast_days['predicted_mood'], marker='o', color='#4CAF50', linewidth=2)
            ax.axhline(50, color='red', linestyle='--', label='警戒ライン (スコア 50)')
            ax.set_ylim(0, 100)
            ax.set_ylabel("気分スコア (0:不調 ~ 100:好調)")
            ax.set_title("1日ごとの予測コンディション")
            ax.grid(True, alpha=0.3)
            ax.legend()
            st.pyplot(fig)
            
        with col2:
            st.write("### 💡 AIアドバイス")
            min_score = forecast_days['predicted_mood'].min()
            min_date = forecast_days.loc[forecast_days['predicted_mood'].idxmin(), 'date']
            
            if min_score < 50:
                st.warning(f"⚠️ **{min_date}** 付近で調子が下がる予測です。生活習慣（授業・バイト・食事・睡眠）を見直してみましょう。")
            else:
                st.success("🎉 全体的にコンディションは良好な予測です！このペースを維持しましょう。")
                
        st.dataframe(forecast_days[['date', 'predicted_mood', 'pressure_change']].rename(columns={
            'date': '日付',
            'predicted_mood': '予測気分スコア',
            'pressure_change': '気圧変化 (hPa)'
        }))
    else:
        st.error("気象予報データの範囲外です。")
else:
    st.error("気象データの取得に失敗しました。")
