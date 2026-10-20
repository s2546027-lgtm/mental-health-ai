import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from datetime import datetime

# ページ設定
st.set_page_config(page_title="大学生向け メンタル不調・気分予報 AI", layout="wide")

st.title("🧠 大学生向け メンタル不調・気分予報 AI")
st.write("Pythonで処理する特徴量（sleep_hours, task_load, pressure_change）を基にコンディションを予測します。")

# --- Python上の特徴量・変数入力 ---
st.sidebar.header("⚙️ Input Features (Python Variables)")

# Colabで定義していた変数名をそのまま使用
sleep_hours = st.sidebar.slider("sleep_hours (睡眠時間)", 0.0, 12.0, 7.0, 0.5)
task_load = st.sidebar.slider("task_load (課題・バイト負荷 1-5)", 1, 5, 3)
lat = st.sidebar.number_input("latitude (緯度)", value=35.6762)
lon = st.sidebar.number_input("longitude (経度)", value=139.6503)

# 気象データ取得 (Open-Meteo API)
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

# --- 予測計算ロジック (Python関数) ---
def predict_mood(sleep_h, task_l, p_change):
    # sleep_hours, task_load, pressure_change から mood スコアを算出
    base_score = 75.0 + (sleep_h - 7.0) * 6.0 - (task_l - 1) * 7.0 - np.abs(p_change) * 2.5
    return np.clip(base_score, 0, 100)

if weather_df is not None:
    weather_df['date'] = weather_df['time'].dt.date
    daily_weather = weather_df.groupby('date')['pressure'].mean().reset_index()
    
    # 特徴量: pressure_change の算出
    daily_weather['pressure_change'] = daily_weather['pressure'].diff().fillna(0)
    
    today = datetime.now().date()
    forecast_days = daily_weather[daily_weather['date'] >= today].head(7).copy()
    
    if len(forecast_days) > 0:
        # 特徴量 DataFrame の作成 (Colabと同じ構造)
        X_test = pd.DataFrame({
            'sleep_hours': [sleep_hours] * len(forecast_days),
            'task_load': [task_load] * len(forecast_days),
            'pressure_change': forecast_days['pressure_change']
        })
        
        # 予測の実行
        forecast_days['predicted_mood'] = [
            predict_mood(row['sleep_hours'], row['task_load'], row['pressure_change'])
            for _, row in X_test.iterrows()
        ]
        
        st.subheader("🔮 予測結果 (predicted_mood)")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(forecast_days['date'].astype(str), forecast_days['predicted_mood'], marker='o', color='#4CAF50', linewidth=2)
            ax.axhline(50, color='red', linestyle='--', label='Warning Threshold (50)')
            ax.set_ylim(0, 100)
            ax.set_ylabel("predicted_mood (0-100)")
            ax.set_title("Forecast Condition Trend")
            ax.grid(True, alpha=0.3)
            ax.legend()
            st.pyplot(fig)
            
        with col2:
            st.write("### 💡 Status")
            min_score = forecast_days['predicted_mood'].min()
            if min_score < 50:
                st.warning(f"⚠️ **{forecast_days.loc[forecast_days['predicted_mood'].idxmin(), 'date']}** に不調が予測されます。")
            else:
                st.success("🎉 コンディションは良好です。")
                
        # 入力特徴量と予測結果のデータフレームを表示
        st.write("### 📊 Dataset Summary")
        st.dataframe(forecast_days[['date', 'predicted_mood', 'pressure_change']])
    else:
        st.error("気象予報データが取得できませんでした。")
else:
    st.error("API通信エラーが発生しました。")
