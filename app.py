from datetime import datetime, timedelta
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
import japanize_matplotlib
from google.oauth2.service_account import Credentials
import gspread

# 1. ページ設定
st.set_page_config(
    page_title="大学生向け メンタル不調・気分予報", layout="wide"
)

st.title("大学生向け メンタル不調・気分予報")
st.write(
    "生活習慣や現在のコンディション、および Open-Meteo"
    " 気象データ（気圧変化ΔP）から、今後1週間の気分推移を事前予測します。"
)


# 2. Colabで学習したモデル (.pkl) の読み込み
@st.cache_resource
def load_trained_model():
  try:
    return joblib.load("mental_model.pkl")
  except Exception as e:
    return None


model = load_trained_model()

# 3. サイドバー（アンケートの全項目を忠実に再現）
st.sidebar.header("📋 あなたの現在の生活状況 (アンケート)")

# 気分・体調 (0~10)
mood_score = st.sidebar.slider(
    "今の気分を0から10で表してください (悪ければ0, 良ければ10)", 0, 10, 5
)
health_score = st.sidebar.slider(
    "今の体調を0から10で表してください (悪ければ0, 良ければ10)", 0, 10, 5
)

# 生活習慣
sleep_hours = st.sidebar.slider(
    "本日の睡眠時間は何時間ですか？", 0.0, 10.0, 7.0, 0.5
)
meal_count = st.sidebar.slider(
    "今日の食事回数または予定回数 (回)", 0, 10, 3
)
exercise_score = st.sidebar.slider(
    "今日どのくらい運動しましたか？ (0:全くしていない ~ 10:がっつり)",
    0,
    10,
    5,
)

# 大学・バイト
class_count = st.sidebar.slider(
    "今日の大学の授業コマ数", 0, 10, 2
)
work_hours = st.sidebar.slider(
    "今日のアルバイト等の実働時間 (時間)", 0.0, 10.0, 0.0, 0.5
)

# 課題関連
has_task_str = st.sidebar.radio(
    "残っている課題や仕事はありますか？", ["いいえ", "はい"]
)
has_task = 1 if has_task_str == "はい" else 0

if has_task == 1:
  task_unaccomplished_rate = st.sidebar.slider(
      "残っている課題の達成率 (1:少し残っている ~ 10:手を付けていない)",
      1,
      10,
      5,
  )
  task_deadline_days = st.sidebar.number_input(
      "残っている課題の直近の締め切りは何日後ですか？",
      min_value=0,
      max_value=30,
      value=3,
  )
else:
  task_unaccomplished_rate = 0
  task_deadline_days = 999  # 課題なしの場合は影響のない大きい値

st.sidebar.subheader("位置情報（気象データ用）")
lat = st.sidebar.number_input("緯度 (デフォルト: 東京)", value=35.6762)
lon = st.sidebar.number_input("経度 (デフォルト: 東京)", value=139.6503)


# 4. 気象データ取得 (Open-Meteo API)
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

if weather_df is not None:
  weather_df["date"] = weather_df["time"].dt.date
  daily_weather = weather_df.groupby("date")["pressure"].mean().reset_index()
  daily_weather["pressure_change"] = (
      daily_weather["pressure"].diff().fillna(0)
  )

  today = datetime.now().date()
  forecast_days = daily_weather[daily_weather["date"] >= today].head(7).copy()

  if len(forecast_days) > 0:
    # 5. Colabでの学習用カラム名に完全対応させた DataFrame の作成
    input_data = pd.DataFrame({
        "meal_count": [meal_count] * len(forecast_days),
        "sleep_hours": [sleep_hours] * len(forecast_days),
        "health_score": [health_score] * len(forecast_days),
        "mood_score": [mood_score] * len(forecast_days),
        "exercise_score": [exercise_score] * len(forecast_days),
        "class_count": [class_count] * len(forecast_days),
        "work_hours": [work_hours] * len(forecast_days),
        "has_task": [has_task] * len(forecast_days),
        "task_unaccomplished_rate": [task_unaccomplished_rate]
        * len(forecast_days),
        "task_deadline_days": [task_deadline_days] * len(forecast_days),
        "pressure_change": forecast_days["pressure_change"],
    })
# Google スプレッドシート接続関数
def get_spreadsheet():
  scopes = [
      "https://www.googleapis.com/auth/spreadsheets",
      "https://www.googleapis.com/auth/drive",
  ]
  credentials = Credentials.from_service_account_info(
      st.secrets["gcp_service_account"], scopes=scopes
  )
  client = gspread.authorize(credentials)
  return client.open("mental_app_logs").sheet1
    
    # 6. 推論処理（Colabモデルがある場合は推論、エラー時はルールベース）
    
    predicted_scores = []
    if model is not None:
      try:
        raw_pred = model.predict(input_data)
        predicted_scores = np.clip(raw_pred, 0.0, 10.0)
      except Exception:
        model = None  # エラー時はフォールバックに移行

    if model is None:
      # ルールベース計算（0~10スケール）
      for _, row in input_data.iterrows():
        base = float(row["mood_score"])
        h_eff = (row["health_score"] - 5.0) * 0.15
        s_eff = (row["sleep_hours"] - 7.0) * 0.2
        m_eff = (row["meal_count"] - 3.0) * 0.15
        e_eff = (row["exercise_score"] - 5.0) * 0.1
        c_eff = -1.0 * row["class_count"] * 0.1
        w_eff = -1.0 * row["work_hours"] * 0.1
        t_eff = -1.0 * (row["task_unaccomplished_rate"] / 10.0) * 0.5
        p_eff = -1.0 * np.abs(row["pressure_change"]) * 0.15

        s = base + h_eff + s_eff + m_eff + e_eff + c_eff + w_eff + t_eff + p_eff
        predicted_scores.append(round(float(np.clip(s, 0.0, 10.0)), 1))

    forecast_days["predicted_mood"] = predicted_scores

# --- データ蓄積・答え合わせ用フォーム ---
    st.markdown("---")
    st.subheader("📝 本日のデータ蓄積・答え合わせ")
    st.write(
        "入力したコンディションと「実際の気分」を送信することで、今後のAI予測精度向上に役立てられます。"
    )

    actual_mood = st.slider(
        "今日の「実際の気分」はどうでしたか？ (0:非常に悪い ~ 10:非常に良い)",
        0,
        10,
        5,
    )

    if st.button("💾 データを送信して保存する"):
      try:
        sheet = get_spreadsheet()
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 保存する1行データ
        row_data = [
            now_str,
            mood_score,
            health_score,
            sleep_hours,
            meal_count,
            exercise_score,
            class_count,
            work_hours,
            has_task,
            task_unaccomplished_rate,
            task_deadline_days,
            actual_mood,
        ]

        sheet.append_row(row_data)
        st.success(
            "🎉 データをスプレッドシートに正常に保存しました！ご協力ありがとうございます。"
        )
      except Exception as e:
        st.error(f"⚠️ 保存に失敗しました。設定を確認してください: {e}")

    # 7. 画面表示
    st.subheader("今後1週間のメンタル・気分予測結果")
    col1, col2 = st.columns([2, 1])

    with col1:
      fig, ax = plt.subplots(figsize=(10, 4.5))

      x_labels = [d.strftime("%m/%d") for d in forecast_days["date"]]
      y_values = forecast_days["predicted_mood"].tolist()

      ax.plot(
          x_labels,
          y_values,
          marker="o",
          color="#1E88E5",
          linewidth=2.5,
          markersize=8,
          label="予測気分スコア (0~10)",
      )

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

      ax.axhline(
          5.0,
          color="#E53935",
          linestyle="--",
          linewidth=1.5,
          label="警戒ライン (5.0)",
      )

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
      st.write("### AIアドバイス")
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

    st.dataframe(
        forecast_days[["date", "predicted_mood", "pressure_change"]].rename(
            columns={
                "date": "日付",
                "predicted_mood": "予測気分スコア (0-10)",
                "pressure_change": "気圧変化 (hPa)",
            }
        )
    )
  else:
    st.error("気象予報データの範囲外です。")
else:
  st.error("気象データの取得に失敗しました。")
