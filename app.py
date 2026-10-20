import datetime
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st

# 日本語フォント対応設定
try:
  import japanize_matplotlib
except ImportError:
  pass

# ==============================================================================
# ページ基本設定
# ==============================================================================
st.set_page_config(
    page_title="大学生向け メンタル不調・気分予測AI",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 大学生向け メンタル不調・気分予報 AI")
st.caption(
    "生活習慣（睡眠・課題等）と Open-Meteo 気象データ（気圧変化ΔP）から、今後1週間のコンディションを事前予測します。"
)

# ==============================================================================
# 1. データの読み込み & 汎用モデル構築（キャッシュ利用で高速化）
# ==============================================================================
SPREADSHEET_URL = "https://docs.google.com/spreadsheets/d/1f_Xd7KpQ2od3_kw7aUM8jH12ZD8E_doww7VBA9MeUDc/export?format=csv&gid=857912218"


@st.cache_data
def load_and_train():
  try:
    df_raw = pd.read_csv(SPREADSHEET_URL)
  except Exception as e:
    st.error("スプレッドシートデータの取得に失敗しました。")
    return None, None

  # 列名マッピング
  column_mapping = {
      "タイムスタンプ": "timestamp",
      "メールアドレス": "user_id",
      "今日の食事回数または、予定回数は何回ですか？": "meal_count",
      "本日の睡眠時間は何時間ですか？(数字を半角で入力してください)": (
          "sleep_hours"
      ),
      "今の体調を0から10で表してください。(悪ければ0、良ければ10)": (
          "health_score"
      ),
      "今の気分を0から10で表してください。(悪ければ0、良ければ10)": (
          "mood_score"
      ),
      (
          "今日どのくらい運動しましたか？(全くしていない場合は0、がっつりしている場合は10)"
      ): "exercise_score",
      "今日の大学の授業のコマ数はいくつでしたか。": "class_count",
      "今日のアルバイト等の実働時間は何時間ですか。": "work_hours",
      (
          "残っている課題や仕事はありますか？\n「はい」と答えた方は次の質問にもお答えください"
      ): "has_task",
      (
          "「はい」と答えた方のみ\n残っている課題や仕事の達成率はどのくらいですか？\n(少しだけ残っている場合は1、手を付けていない場合は10)"
      ): "task_unaccomplished_rate",
      (
          "「はい」と答えた方のみ\n残っている課題や仕事の直近の締め切りは何日後ですか？"
      ): "task_deadline_days",
  }
  df = df_raw.rename(columns=column_mapping).copy()

  # 前処理
  df["datetime"] = pd.to_datetime(df["timestamp"])
  df["date"] = df["datetime"].dt.strftime("%Y-%m-%d")
  df["user_id"] = (
      df["user_id"].fillna("user_unknown").astype(str)
      if "user_id" in df.columns
      else "user_default"
  )
  df = df.sort_values("datetime").drop_duplicates(
      subset=["user_id", "date"], keep="last"
  )

  numeric_cols = [
      "meal_count",
      "sleep_hours",
      "health_score",
      "mood_score",
      "exercise_score",
      "class_count",
      "work_hours",
      "task_unaccomplished_rate",
      "task_deadline_days",
  ]
  for col in numeric_cols:
    if col in df.columns:
      df[col] = pd.to_numeric(
          df[col].astype(str).str.extract(r"(\d+\.?\d*)")[0], errors="coerce"
      )

  if "has_task" in df.columns:
    df["has_task"] = (
        df["has_task"].astype(str).apply(lambda x: 1 if "はい" in x else 0)
    )

  df["task_unaccomplished_rate"] = df["task_unaccomplished_rate"].fillna(0)
  df["task_deadline_days"] = df["task_deadline_days"].fillna(999)
  df = df.fillna(0)

  # 過去気象データの取得（東京）
  start_date, end_date = df["date"].min(), df["date"].max()
  weather_url = f"https://archive-api.open-meteo.com/v1/archive?latitude=35.6895&longitude=139.6917&start_date={start_date}&end_date={end_date}&daily=surface_pressure_mean,temperature_2m_max&timezone=Asia%2FTokyo"

  try:
    res = requests.get(weather_url).json()
    df_weather = pd.DataFrame({
        "date": res["daily"]["time"],
        "pressure": res["daily"]["surface_pressure_mean"],
        "max_temp": res["daily"]["temperature_2m_max"],
    })
    df_weather["pressure_diff"] = (
        df_weather["pressure"].diff().fillna(0).round(1)
    )
    df = pd.merge(df, df_weather, on="date", how="left").fillna(0)
  except Exception:
    pass

  # 汎用特徴量リスト
  generic_features = [
      "health_score",
      "sleep_hours",
      "meal_count",
      "exercise_score",
      "class_count",
      "work_hours",
      "has_task",
      "task_unaccomplished_rate",
      "task_deadline_days",
      "pressure",
      "pressure_diff",
      "max_temp",
  ]
  valid_features = [f for f in generic_features if f in df.columns]

  X = df[valid_features]
  y = df["mood_score"]

  # 汎用LightGBMモデルの学習
  model = lgb.LGBMRegressor(
      n_estimators=100,
      learning_rate=0.05,
      max_depth=3,
      min_child_samples=1,
      random_state=42,
  )
  model.fit(X, y)

  return model, valid_features


model, valid_features = load_and_train()

# ==============================================================================
# 2. サイドバー：ユーザーの想定生活習慣パラメータ入力
# ==============================================================================
st.sidebar.header("📋 あなたの平常時の予定・コンディション")

user_health = st.sidebar.slider(
    "現在の体調スコア (0:絶不調 〜 10:絶好調)", 0, 10, 7
)
user_sleep = st.sidebar.number_input(
    "目標睡眠時間 (時間)", 0.0, 12.0, 6.5, step=0.5
)
user_meal = st.sidebar.slider("食事・予定回数", 0, 5, 3)
user_exercise = st.sidebar.slider("運動スコア (0:無 〜 10:激しい)", 0, 10, 3)
user_class = st.sidebar.slider("1日の平均授業コマ数", 0, 6, 2)
user_work = st.sidebar.slider("1日のバイト時間 (時間)", 0, 10, 0)
has_task_bool = st.sidebar.checkbox("残っている課題・仕事がある", value=True)
has_task = 1 if has_task_bool else 0
user_task_rate = st.sidebar.slider(
    "課題の未達成度 (1:残り僅か 〜 10:未着手)", 1, 10, 5
) if has_task_bool else 0

# ==============================================================================
# 3. Open-Meteo 天気予報 API から今後1週間（8日間）の気象データ取得
# ==============================================================================
@st.cache_data(ttl=3600)  # 1時間キャッシュ
def get_weather_forecast():
  forecast_url = "https://api.open-meteo.com/v1/forecast?latitude=35.6895&longitude=139.6917&daily=surface_pressure_mean,temperature_2m_max&timezone=Asia%2FTokyo&forecast_days=8"
  res = requests.get(forecast_url).json()
  df_f = pd.DataFrame({
      "date": res["daily"]["time"],
      "pressure": res["daily"]["surface_pressure_mean"],
      "max_temp": res["daily"]["temperature_2m_max"],
  })
  df_f["pressure_diff"] = df_f["pressure"].diff().fillna(0).round(1)
  return df_f


if model is not None:
  df_forecast = get_weather_forecast()

  # 予測計算処理
  weekly_results = []
  for i in range(1, 8):
    target_date = df_forecast.loc[i, "date"]
    w_pressure = df_forecast.loc[i, "pressure"]
    w_p_diff = df_forecast.loc[i, "pressure_diff"]
    w_temp = df_forecast.loc[i, "max_temp"]

    input_data = {
        "health_score": user_health,
        "sleep_hours": user_sleep,
        "meal_count": user_meal,
        "exercise_score": user_exercise,
        "class_count": user_class,
        "work_hours": user_work,
        "has_task": has_task,
        "task_unaccomplished_rate": user_task_rate,
        "task_deadline_days": max(1, 7 - i),
        "pressure": w_pressure,
        "pressure_diff": w_p_diff,
        "max_temp": w_temp,
    }

    input_df = pd.DataFrame([input_data])[valid_features]
    pred_mood = np.clip(model.predict(input_df)[0], 0, 10)

    if pred_mood < 4.0:
      status = "⚠️ 不調注意"
      advice = "気圧低下や負荷の影響で気分が沈みやすい日です。十分な睡眠を意識しましょう。"
    elif pred_mood < 6.0:
      status = "🟡 通常"
      advice = "平常運転のコンディションです。マイペースに過ごしましょう。"
    else:
      status = "🟢 好調"
      advice = "非常に良好なコンディションの予測です！集中したいタスクを進めましょう。"

    weekly_results.append({
        "日付": target_date,
        "予測気分スコア": round(pred_mood, 1),
        "判定": status,
        "気圧(hPa)": w_pressure,
        "前日気圧差(hPa)": w_p_diff,
        "最高気温(℃)": w_temp,
        "アドバイス": advice,
    })

  df_res = pd.DataFrame(weekly_results)

  # ==============================================================================
  # 4. メイン画面の可視化 & 結果表示
  # ==============================================================================
  # サマリー指標（メトリクス）
  col1, col2, col3 = st.columns(3)
  col1.metric("明日の予測気分スコア", f"{df_res.loc[0, '予測気分スコア']} / 10.0")
  col2.metric("明日の気圧変化(ΔP)", f"{df_res.loc[0, '前日気圧差(hPa)']} hPa")
  col3.metric("明日のコンディション", df_res.loc[0, "判定"])

  st.divider()

  # 1週間推移グラフ
  st.subheader("📈 今後1週間の気分スコア & 気圧変化(ΔP) 推移")

  fig, ax1 = plt.subplots(figsize=(10, 4))
  ax2 = ax1.twinx()

  # 気分スコア（折れ線）
  ax1.plot(
      df_res["日付"],
      df_res["予測気分スコア"],
      marker="o",
      color="#1f77b4",
      linewidth=2.5,
      label="予測気分スコア (0-10)",
  )
  ax1.set_ylabel("気分スコア", color="#1f77b4", fontsize=12)
  ax1.set_ylim(-0.5, 10.5)
  ax1.grid(True, linestyle="--", alpha=0.5)

  # 前日気圧差（棒グラフ）
  colors = ["#ff7f0e" if x < -3 else "#2ca02c" for x in df_res["前日気圧差(hPa)"]]
  ax2.bar(
      df_res["日付"],
      df_res["前日気圧差(hPa)"],
      alpha=0.3,
      color=colors,
      label="前日気圧差 ΔP (hPa)",
  )
  ax2.set_ylabel("前日気圧差 ΔP (hPa)", color="#2ca02c", fontsize=12)
  ax2.axhline(0, color="gray", linewidth=0.8, linestyle="--")

  plt.title("1週間のメンタルバイオリズム予報", fontsize=14)
  fig.tight_layout()
  st.pyplot(fig)

  st.divider()

  # 日別詳細カード & テーブル
  st.subheader("📅 日別アドバイス一覧")
  for idx, row in df_res.iterrows():
    with st.expander(
        f"📅 {row['日付']} | スコア: {row['予測気分スコア']} ({row['判定']})"
    ):
      st.write(
          f"・**気象条件**: 気圧 {row['気圧(hPa)']} hPa (前日比:"
          f" {row['前日気圧差(hPa)']} hPa) / 最高気温 {row['最高気温(℃)']} ℃"
      )
      st.write(f"・**アドバイス**: {row['アドバイス']}")
