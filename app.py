from datetime import datetime, timedelta
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st

# ページ設定
st.set_page_config(
    page_title="大学生向け メンタル不調・気分予報 AI", layout="wide"
)

st.title("🧠 大学生向け メンタル不調・気分予報 AI")
st.write(
    "現在の気分や生活習慣（睡眠・運動・食事等）と Open-Meteo"
    " 気象データ（気圧変化ΔP）から、今後1週間のコンディションを事前予測します。"
)


# ---------------- Colabで学習したモデル (.pkl) の読み込み ----------------
@st.cache_resource
def load_trained_model():
  try:
    # GitHub上の mental_model.pkl を読み込む
    model = joblib.load("mental_model.pkl")
    return model
  except Exception as e:
    st.error(
        f"モデルファイル (mental_model.pkl) の読み込みに失敗しました: {e}"
    )
    return None


model = load_trained_model()

# ---------------- サイドバー（アンケート全項目: 0〜10） ----------------
st.sidebar.header("📋 あなたの現在の生活状況 (アンケート)")

current_mood = st.sidebar.slider(
    "現在の気分 (0:非常に悪い ~ 10:非常に良い)", 0, 10, 5
)
sleep = st.sidebar.slider("昨夜の睡眠時間 (0 ~ 10時間)", 0.0, 10.0, 7.0, 0.5)
classes = st.sidebar.slider("本日の授業コマ数 (0 ~ 10コマ)", 0, 10, 2)
meal = st.sidebar.slider("本日の食事回数 (0 ~ 10回)", 0, 10, 3)
exercise = st.sidebar.slider(
    "運動したスコア (0:全くしていない ~ 10:しっかりした)", 0, 10, 5
)
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
          "time": pd.to_datetime(data["hourly"]["time"]),
          "pressure": data["hourly"]["surface_pressure"],
      })
  except Exception:
    pass
  return None


weather_df = get_weather_forecast(lat, lon)

if weather_df is not None and model is not None:
  weather_df["date"] = weather_df["time"].dt.date
  daily_weather = weather_df.groupby("date")["pressure"].mean().reset_index()
  daily_weather["pressure_change"] = (
      daily_weather["pressure"].diff().fillna(0)
  )

  today = datetime.now().date()
  forecast_days = daily_weather[daily_weather["date"] >= today].head(7).copy()

  if len(forecast_days) > 0:
    # Colabで学習させた特徴量（列名）に合わせてデータフレームを作成
    # ※ Colab学習時の列名と完全に一致させてください
    input_data = pd.DataFrame({
        "health_score": [health_score] * len(forecast_days),
        "sleep_hours": [sleep] * len(forecast_days),
        "meal_count": [meal] * len(forecast_days),
        "exercise_score": [exercise] * len(forecast_days),
        "class_count": [class_count] * len(forecast_days),
        "work_hours": [work] * len(forecast_days),
        "has_task": [has] * len(forecast_days),
        "task_unaccomplished_rate": [task_unaccomplished_rate] * len(forecast_days),
        "task_deadline_days": [task_deadline_days] * len(forecast_days),
        "pressure": [pressure] * len(forecast_days),
        "pressure_diff": [pressure_diff] * len(forecast_days),
        "max_temp": forecast_days["max_temp"],
    })

    # Colabモデルでの予測実行
    raw_predictions = model.predict(input_data)

    # 予測結果を 0.0 ~ 10.0 の範囲に正規化・整形
    forecast_days["predicted_mood"] = np.clip(raw_predictions, 0.0, 10.0)

    # ---------------- 画面表示 ----------------
    st.subheader("🔮 今後1週間のメンタル・気分予測結果 (Colab AIモデル)")
    col1, col2 = st.columns([2, 1])

    with col1:
      fig, ax = plt.subplots(figsize=(10, 4.5))

      x_labels = [d.strftime("%m/%d") for d in forecast_days["date"]]
      y_values = forecast_days["predicted_mood"].tolist()

      # 折れ線描画
      ax.plot(
          x_labels,
          y_values,
          marker="o",
          color="#1E88E5",
          linewidth=2.5,
          markersize=8,
          label="AI予測スコア (0~10)",
      )

      # 各データ点の上に数値を表示
      for i, val in enumerate(y_values):
        ax.text(
            i,
            val + 0.3,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="#1565C0",
        )

      # 警戒ライン（5.0）
      ax.axhline(
          5.0,
          color="#E53935",
          linestyle="--",
          linewidth=1.5,
          label="警戒ライン (5.0)",
      )

      # 軸・範囲設定
      ax.set_ylim(0, 10.5)
      ax.set_yticks(range(0, 11))
      ax.set_ylabel("気分スコア (0:不調 ~ 10:好調)", fontsize=11)
      ax.set_title(
          "1日ごとの気分予測推移 (0~10スケール)",
          fontsize=13,
          fontweight="bold",
      )

      ax.grid(True, linestyle=":", alpha=0.6)
      ax.legend(loc="lower right")

      st.pyplot(fig)

    with col2:
      st.write("### 💡 AIアドバイス")
      min_score = forecast_days["predicted_mood"].min()
      min_date = forecast_days.loc[
          forecast_days["predicted_mood"].idxmin(), "date"
      ]

      if min_score < 5.0:
        st.warning(
            f"⚠️ **{min_date}** 付近で調子が下がる予測（スコア:"
            f" {min_score:.1f}）です。十分な休養を意識しましょう。"
        )
      else:
        st.success(
            "🎉 1週間を通してコンディションは良好な予測です！（最低スコア:"
            f" {min_score:.1f}）"
        )

    # データテーブル
    st.dataframe(
        forecast_days[["date", "predicted_mood", "pressure_change"]].rename(
            columns={
                "date": "日付",
                "predicted_mood": "AI予測気分スコア (0-10)",
                "pressure_change": "気圧変化 (hPa)",
            }
        )
    )
  else:
    st.error("気象予報データの範囲外です。")
