import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from datetime import datetime, timedelta

# ページ設定
st.set_page_config(page_title="大学生向け メンタル不調・気分予報 AI", layout="wide")

st.title("🧠 大学生向け メンタル不調・気分予報 AI")
st.write("生活習慣（睡眠・運動・食事・課題等）と Open-Meteo 気象データ（気圧変化ΔP）から、今後1週間のコンディションを事前予測します。")

# 予測モデル（0〜10のスコア値を基準に算出）
class SimplePredictor:
    def predict(self, df_input):
        scores = []
        for _, row in df_input.iterrows():
            # 1. 睡眠時間 (0~10) : 7以上が理想
            sleep_effect = (row['sleep_hours'] - 7.0) * 4.0
            # 2. 授業コマ数 (0~10)
            class_effect = -1.0 * row['classes_count'] * 2.5
            # 3. 食事回数 (0~10) : 3回を基準
            meal_effect = (row['meal_count'] - 3.0) * 3.0
            # 4. 運動スコア (0~10) : スコアが高いほどプラス
            exercise_effect = row['exercise_score'] * 2.5
            # 5. 課題・バイト負荷 (0~10)
            task_effect = -1.0 * row['task_load'] * 3.0
            # 6. 気圧変化
            pressure_effect = -1.0 * np.abs(row['pressure_change']) * 1.5
            
            # 総合スコア算出
            score = 65.0 + sleep_effect + class_effect + meal_effect + exercise_effect + task_effect + pressure_effect
            scores.append(np.clip(score, 0, 100))
        return np.array(scores)

@st.cache_resource
def load_model():
    return SimplePredictor()

model = load_model()

# ---------------- サイドバー（0〜10 スコア仕様のアンケート項目） ----------------
st.sidebar.header("📋 あなたの現在の生活状況 (アンケート)")

sleep = st.sidebar.slider("昨夜の睡眠時間 (0 ~ 10時間)", 0.0, 10.0, 7.0, 0.5)
classes = st.sidebar.slider("本日の授業コマ数 (0 ~ 10コマ)", 0, 10, 2)
meal = st.sidebar.slider("本日の食事回数 (0 ~ 10回)", 0, 10, 3)
exercise = st.sidebar.slider("運動したスコア (0:全くしていない ~ 10:しっかりした)", 0, 10, 5)
task = st.sidebar.slider("課題・バイト負荷スコア (0:低 ~ 10:高)", 0, 10, 5)

st.sidebar.subheader("📍 位置情報（気象データ用）")
lat = st.sidebar.number_input("緯度 (デフォルト: 東京)", value=35.6762)
lon = st.sidebar.number_input("経度 (デフォルト: 東京)", value=139.6503)

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
        # 入力データフレームの組み立て
        input_data = pd.DataFrame({
            'sleep_hours': [sleep] * len(forecast_days),
            'classes_count': [classes] * len(forecast_days),
            'meal_count': [meal] * len(forecast_days),
            'exercise_score': [exercise] * len(forecast_days),
            'task_load': [task] * len(forecast_days),
            'pressure_change': forecast_days['pressure_change']
        })
        
        forecast_days['predicted_mood'] = model.predict(input_data)
        
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
                st.warning(f"⚠️ **{min_date}** 付近で調子が下がる予測です。運動・食事・睡眠のバランスを整えましょう。")
            else:
                st.success("🎉 全体的にコンディションは良好な予測です！良い生活習慣を維持しましょう。")
                
        st.dataframe(forecast_days[['date', 'predicted_mood', 'pressure_change']].rename(columns={
            'date': '日付',
            'predicted_mood': '予測気分スコア',
            'pressure_change': '気圧変化 (hPa)'
        }))
    else:
        st.error("気象予報データの範囲外です。")
else:
    st.error("気象データの取得に失敗しました。")
