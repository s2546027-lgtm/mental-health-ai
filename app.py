import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from datetime import datetime, timedelta

# ページ設定
st.set_page_config(page_title="大学生向け メンタル不調・気分予報 AI", layout="wide")

st.title("🧠 大学生向け メンタル不調・気分予報 AI")
st.write("生活習慣（睡眠・課題等）と Open-Meteo 気象データ（気圧変化ΔP）から、今後1週間のコンディションを事前予測します。")

# 予測ロジック（エラー回避版）
def predict_mood(sleep_h, task_l, p_change):
    sleep_effect = (sleep_h - 7.0) * 5.0
    task_effect = -1.0 * (task_l - 1) * 4.0
    pressure_effect = -1.0 * np.abs(p_change) * 1.5
    mood_score = 70.0 + sleep_effect + task_effect + pressure_effect
    return np.clip(mood_score, 0, 100)

# サイドバー入力
st.sidebar.header("📋 あなたの現在の生活状況")
sleep = st.sidebar.slider("昨夜の睡眠時間 (時間)", 0.0, 12.0, 7.0, 0.5)
task = st.sidebar.slider("現在の課題・バイト負荷 (1:低 ~ 5:高)", 1, 5, 3)
lat = st.sidebar.number_input("緯度 (デフォルト: 東京)", value=35.6762)
lon = st.sidebar.number_input("経度 (デフォルト: 東京)", value=139.6503)

# 気象データ取得 (Open-Meteo API)
@st.cache_data(ttl=3600)
def get_weather_forecast(latitude, longitude):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&hourly=surface_pressure&timezone=Asia%2FTokyo"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            df_weather = pd.DataFrame({
                'time': pd.to_datetime(data['hourly']['time']),
                'pressure': data['hourly']['surface_pressure']
            })
            return df_weather
    except Exception:
        pass
    return None

weather_df = get_weather_forecast(lat, lon)

if weather_df is not None:
    # 日別の平均気圧と変化量を計算
    weather_df['date'] = weather_df['time'].dt.date
    daily_weather = weather_df.groupby('date')['pressure'].mean().reset_index()
    daily_weather['pressure_change'] = daily_weather['pressure'].diff().fillna(0)
    
    # 今後7日分を取得
    today = datetime.now().date()
    forecast_days = daily_weather[daily_weather['date'] >= today].head(7).copy()
    
    if len(forecast_days) > 0:
        # 予測の実行
        scores = [predict_mood(sleep, task, p_change) for p_change in forecast_days['pressure_change']]
        forecast_days['predicted_mood'] = scores
        
        # 画面表示
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
                st.warning(f"⚠️ **{min_date}** 付近で調子が下がる予測です。気圧変化や課題負荷の影響を受けやすいため、十分な睡眠を確保しましょう。")
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
