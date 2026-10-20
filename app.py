import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from datetime import datetime, timedelta

# ページ設定
st.set_page_config(page_title="大学生向け メンタル不調・気分予報 AI", layout="wide")

st.title("🧠 大学生向け メンタル不調・気分予報 AI")
st.write("現在の気分や生活習慣（睡眠・運動・食事等）と Open-Meteo 気象データ（気圧変化ΔP）から、今後1週間のコンディションを事前予測します。")

# 予測モデル（出力スコアを 0〜10 スケールに統一）
class SimplePredictor:
    def predict(self, df_input):
        scores = []
        for _, row in df_input.iterrows():
            # ベーススコア: 今の気分 (0~10)
            base_score = row['current_mood']
            
            # 各生活習慣の影響（0~10 スケールに合わせて調整）
            sleep_effect = (row['sleep_hours'] - 7.0) * 0.3
            class_effect = -1.0 * row['classes_count'] * 0.2
            meal_effect = (row['meal_count'] - 3.0) * 0.25
            exercise_effect = (row['exercise_score'] - 5.0) * 0.2
            task_effect = -1.0 * row['task_load'] * 0.25
            pressure_effect = -1.0 * np.abs(row['pressure_change']) * 0.15
            
            score = base_score + sleep_effect + class_effect + meal_effect + exercise_effect + task_effect + pressure_effect
            scores.append(np.clip(score, 0, 10))
        return np.array(scores)

@st.cache_resource
def load_model():
    return SimplePredictor()

model = load_model()

# ---------------- サイドバー（アンケート全項目: 0〜10） ----------------
st.sidebar.header("📋 あなたの現在の生活状況 (アンケート)")

current_mood = st.sidebar.slider("現在の気分 (0:非常に悪い ~ 10:非常に良い)", 0, 10, 5)
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
        # 入力データフレーム
        input_data = pd.DataFrame({
            'current_mood': [current_mood] * len(forecast_days),
            'sleep_hours': [sleep] * len(forecast_days),
            'classes_count': [classes] * len(forecast_days),
            'meal_count': [meal] * len(forecast_days),
            'exercise_score': [exercise] * len(forecast_days),
            'task_load': [task] * len(forecast_days),
            'pressure_change': forecast_days['pressure_change']
        })
        
        forecast_days['predicted_mood'] = model.predict(input_data)
        
        # ---------------- 画面表示 ----------------
        st.subheader("🔮 今後1週間のメンタル・気分予測結果")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # 見やすいグラフのデザイン設定
            fig, ax = plt.subplots(figsize=(10, 4.5))
            
            # 折れ線 & データ点プロット
            dates_str = forecast_days['date'].astype(str)
            scores = forecast_days['predicted_mood']
            
            ax.plot(dates_str, scores, marker='o', color='#1E88E5', linewidth=2.5, markersize=8, label='予測気分スコア')
            
            # 数値テキストの表示（各点の上に数値を明記）
            for i, score in enumerate(scores):
                ax.annotate(f"{score:.1f}", (dates_str.iloc[i], score),
                            textcoords="offset points", xytext=(0, 10), ha='center',
                            fontsize=10, fontweight='bold', color='#1565C0')
            
            # 警戒ライン（5.0）
            ax.axhline(5.0, color='#E53935', linestyle='--', linewidth=1.5, label='警戒ライン (スコア 5)')
            
            # 軸・範囲の設定 (0 ~ 10)
            ax.set_ylim(-0.5, 10.5)
            ax.set_yticks(range(0, 11))
            ax.set_ylabel("予測気分スコア (0:不調 ~ 10:好調)", fontsize=11)
            ax.set_title("1日ごとのコンディション推移", fontsize=13, fontweight='bold', pad=12)
            
            # グリッド & 背景デザイン調整
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.legend(loc='lower right', frameon=True)
            plt.xticks(rotation=15)
            plt.tight_layout()
            
            st.pyplot(fig)
            
        with col2:
            st.write("### 💡 AIアドバイス")
            min_score = forecast_days['predicted_mood'].min()
            min_date = forecast_days.loc[forecast_days['predicted_mood'].idxmin(), 'date']
            
            if min_score < 5.0:
                st.warning(f"⚠️ **{min_date}** 付近で調子が下がる予測（スコア: {min_score:.1f}）です。十分な休養を意識しましょう。")
            else:
                st.success(f"🎉 1週間を通してコンディションは良好な予測です！（最低スコア: {min_score:.1f}）")
                
        # 表表示（0〜10スコア）
        st.dataframe(forecast_days[['date', 'predicted_mood', 'pressure_change']].rename(columns={
            'date': '日付',
            'predicted_mood': '予測気分スコア (0-10)',
            'pressure_change': '気圧変化 (hPa)'
        }))
    else:
        st.error("気象予報データの範囲外です。")
else:
    st.error("気象データの取得に失敗しました。")
