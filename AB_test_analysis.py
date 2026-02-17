# %% [markdown]
# Раздел 0. Описание проекта и метрик
#
# Цель ноутбука — воспроизвести end-to-end анализ A/B-теста скидки на премиум-брони
# с расчётом ARPU, ARPPU и средних трат внутриигровой валюты по группам и платформам.
# В пайплайне реализованы загрузка и проверка качества данных, детект читеров,
# построение аналитического датасета и экспорт всех целевых метрик и срезов.

# %%
# Раздел 1. Импорт библиотек и конфигурация путей
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

pd.set_option("display.max_columns", 50)

DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REQUIRED_FILES = [
    "Money.csv",
    "Cash.csv",
    "ABgroup.csv",
    "Platforms.csv",
    "Cheaters.csv",
]

if not DATA_DIR.exists():
    raise FileNotFoundError(f"Каталог с данными не найден: {DATA_DIR.absolute()}")

missing_files = [fname for fname in REQUIRED_FILES if not (DATA_DIR / fname).exists()]
if missing_files:
    raise FileNotFoundError(
        f"Не найдены исходники: {', '.join(missing_files)} в {DATA_DIR.absolute()}"
    )


# %%
# Раздел 2. Загрузка данных и Data Quality
def load_money(path: Path) -> pd.DataFrame:
    """Загружает платежи, не парся даты на этапе read_csv."""
    df = pd.read_csv(path, dtype={"user_id": "string"}, na_values=["", " "])
    df["date"] = pd.to_datetime(df["date"], format="%d.%m.%Y", errors="coerce")
    df["money"] = pd.to_numeric(df["money"], errors="coerce")
    return df


def load_cash(path: Path) -> pd.DataFrame:
    """Загружает траты внутриигровой валюты."""
    df = pd.read_csv(path, dtype={"user_id": "string"}, na_values=["", " "])
    df["date"] = pd.to_datetime(df["date"], format="%d.%m.%Y", errors="coerce")
    df["cash"] = pd.to_numeric(df["cash"], errors="coerce")
    return df


def load_abgroup(path: Path) -> pd.DataFrame:
    """Загружает распределение по группам эксперимента."""
    return pd.read_csv(path, dtype={"user_id": "string", "group": "string"})


def load_platforms(path: Path) -> pd.DataFrame:
    """Загружает платформу пользователя."""
    return pd.read_csv(path, dtype={"user_id": "string", "platform": "string"})


def load_cheaters(path: Path) -> pd.DataFrame:
    """Загружает список явных читеров."""
    df = pd.read_csv(path, dtype={"user_id": "string"})
    df["cheaters"] = pd.to_numeric(df["cheaters"], errors="coerce").fillna(0).astype(int)
    return df


def dq_stats_transactions(df: pd.DataFrame, value_col: str, table: str) -> Dict[str, float]:
    """Формирует метрики качества для транзакционных таблиц."""
    return {
        "table": table,
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "share_nat_date": float(df["date"].isna().mean()),
        "share_nan_user_id": float(df["user_id"].isna().mean()),
        "negative_value_count": int((df[value_col] < 0).sum()),
        "duplicate_keys": np.nan,
    }


def dq_stats_user_level(df: pd.DataFrame, table: str) -> Dict[str, float]:
    """Формирует метрики качества для пользовательских таблиц."""
    return {
        "table": table,
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "share_nat_date": np.nan,
        "share_nan_user_id": float(df["user_id"].isna().mean()),
        "negative_value_count": np.nan,
        "duplicate_keys": int(df["user_id"].duplicated().sum()),
    }


money_df = load_money(DATA_DIR / "Money.csv")
cash_df = load_cash(DATA_DIR / "Cash.csv")
ab_group_df = load_abgroup(DATA_DIR / "ABgroup.csv")
platforms_df = load_platforms(DATA_DIR / "Platforms.csv")
cheaters_df = load_cheaters(DATA_DIR / "Cheaters.csv")

dq_records = [
    dq_stats_transactions(money_df, "money", "Money"),
    dq_stats_transactions(cash_df, "cash", "Cash"),
    dq_stats_user_level(ab_group_df, "ABgroup"),
    dq_stats_user_level(platforms_df, "Platforms"),
    dq_stats_user_level(cheaters_df, "Cheaters"),
]

dq_summary = pd.DataFrame(dq_records)
dq_summary_path = OUTPUT_DIR / "dq_summary.csv"
dq_summary.to_csv(dq_summary_path, index=False)
print(f"[DQ] Сводка сохранена в {dq_summary_path}")


# %%
# Раздел 3. Детект читеров
def sanitize_transaction_df(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Удаляет строки с пустыми user_id, некорректными датами и отрицательными значениями."""
    clean_df = df.dropna(subset=["user_id", "date"]).copy()
    clean_df = clean_df[clean_df[value_col] >= 0]
    return clean_df


def deduplicate_dimension(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Удаляет строки без user_id и оставляет по одному значению на пользователя."""
    clean_df = df.dropna(subset=["user_id"]).copy()
    clean_df["user_id"] = clean_df["user_id"].astype("string")
    return clean_df.drop_duplicates(subset=["user_id"], keep="last")


money_clean = sanitize_transaction_df(money_df, "money")
cash_clean = sanitize_transaction_df(cash_df, "cash")
ab_group_clean = deduplicate_dimension(ab_group_df, "group")
platforms_clean = deduplicate_dimension(platforms_df, "platform")
cheaters_clean = deduplicate_dimension(cheaters_df, "cheaters")

money_daily = (
    money_clean.groupby(["user_id", "date"], as_index=False)["money"]
    .sum()
    .rename(columns={"money": "money_day"})
)
cash_daily = (
    cash_clean.groupby(["user_id", "date"], as_index=False)["cash"]
    .sum()
    .rename(columns={"cash": "cash_day"})
)

money_user = (
    money_daily.groupby("user_id", as_index=False)
    .agg(total_revenue=("money_day", "sum"), days_with_payments=("date", "nunique"))
)
cash_user = (
    cash_daily.groupby("user_id", as_index=False)
    .agg(
        total_cash=("cash_day", "sum"),
        max_daily_cash=("cash_day", "max"),
        days_with_cash=("date", "nunique"),
    )
)

cheater_base = (
    ab_group_clean[["user_id"]]
    .merge(money_user, on="user_id", how="left")
    .merge(cash_user, on="user_id", how="left")
)
for col in ["total_revenue", "days_with_payments", "total_cash", "max_daily_cash", "days_with_cash"]:
    if col in cheater_base:
        cheater_base[col] = cheater_base[col].fillna(0)

CHEATER_RULES: Dict[str, float] = {
    "cash_quantile": 0.999,
    "money_threshold": 1.0,
    "max_daily_cash_quantile": 0.999,
}

cash_threshold = (
    cheater_base["total_cash"].quantile(CHEATER_RULES["cash_quantile"])
    if not cheater_base["total_cash"].empty
    else np.inf
)
max_daily_cash_threshold = (
    cheater_base["max_daily_cash"].quantile(CHEATER_RULES["max_daily_cash_quantile"])
    if not cheater_base["max_daily_cash"].empty
    else np.inf
)

explicit_cheaters = set(
    cheaters_clean.loc[cheaters_clean["cheaters"] == 1, "user_id"].dropna().astype(str)
)
high_cash_low_money = set(
    cheater_base.loc[
        (cheater_base["total_cash"] >= cash_threshold)
        & (cheater_base["total_revenue"] <= CHEATER_RULES["money_threshold"]),
        "user_id",
    ].astype(str)
)
extreme_daily_cash = set(
    cheater_base.loc[cheater_base["max_daily_cash"] >= max_daily_cash_threshold, "user_id"].astype(str)
)

print(f"[Cheaters] Явные читеры: {len(explicit_cheaters)}")
print(
    "[Cheaters] Высокий total_cash при низком total_money: "
    f"{len(high_cash_low_money)} (порог cash ≥ {cash_threshold:.2f}, money ≤ {CHEATER_RULES['money_threshold']})"
)
print(
    "[Cheaters] Экстремальные max_daily_cash: "
    f"{len(extreme_daily_cash)} (порог cash_day ≥ {max_daily_cash_threshold:.2f})"
)

all_user_ids = pd.Index(ab_group_clean["user_id"].dropna().astype(str).unique())
all_user_ids = all_user_ids.union(pd.Index(list(explicit_cheaters)))

cheater_flags = pd.DataFrame({"user_id": all_user_ids})
cheater_flags["explicit_cheater"] = cheater_flags["user_id"].isin(explicit_cheaters)
cheater_flags["rule_high_cash_low_revenue"] = cheater_flags["user_id"].isin(high_cash_low_money)
cheater_flags["rule_extreme_daily_cash"] = cheater_flags["user_id"].isin(extreme_daily_cash)
cheater_flags["any_cheater_flag"] = cheater_flags[
    ["explicit_cheater", "rule_high_cash_low_revenue", "rule_extreme_daily_cash"]
].any(axis=1)

clean_users = set(
    cheater_flags.loc[~cheater_flags["any_cheater_flag"], "user_id"].astype(str)
)
print(f"[Cheaters] Чистых пользователей после фильтров: {len(clean_users)}")


# %%
# Раздел 4. Аналитический датасет
daily_merged = pd.merge(money_daily, cash_daily, on=["user_id", "date"], how="outer")
daily_merged["money_day"] = daily_merged["money_day"].fillna(0)
daily_merged["cash_day"] = daily_merged["cash_day"].fillna(0)

daily_dataset = (
    daily_merged.merge(ab_group_clean, on="user_id", how="inner")
    .merge(platforms_clean, on="user_id", how="left")
)
daily_dataset["platform"] = daily_dataset["platform"].fillna("Unknown")
daily_dataset = daily_dataset[daily_dataset["user_id"].isin(clean_users)].copy()

user_level_agg = (
    daily_dataset.groupby("user_id", as_index=False)
    .agg(total_revenue=("money_day", "sum"), total_cash=("cash_day", "sum"))
)

user_level = (
    ab_group_clean.merge(platforms_clean, on="user_id", how="left")
    .merge(user_level_agg, on="user_id", how="left")
)
user_level["platform"] = user_level["platform"].fillna("Unknown")
user_level["total_revenue"] = user_level["total_revenue"].fillna(0)
user_level["total_cash"] = user_level["total_cash"].fillna(0)
user_level = user_level[user_level["user_id"].isin(clean_users)].copy()
user_level["is_payer"] = user_level["total_revenue"] > 0


# %%
# Раздел 5. Метрики и доверительные интервалы
def aggregate_metrics(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """Считает ARPU, ARPPU и cash_per_user для переданных группировок."""
    grouped = (
        df.groupby(group_cols, dropna=False)
        .agg(
            total_revenue=("money_day", "sum"),
            total_cash=("cash_day", "sum"),
            user_count=("user_id", pd.Series.nunique),
            payer_count=("money_day", lambda x: (x > 0).sum()),
        )
        .reset_index()
    )
    grouped["arpu"] = grouped["total_revenue"] / grouped["user_count"].replace(0, np.nan)
    grouped["arppu"] = grouped["total_revenue"] / grouped["payer_count"].replace(0, np.nan)
    grouped["cash_per_user"] = grouped["total_cash"] / grouped["user_count"].replace(0, np.nan)
    return grouped


metrics_daily = aggregate_metrics(
    daily_dataset, ["date", "group", "platform"]
).sort_values(["date", "group", "platform"])

metrics_total = (
    user_level.groupby(["group", "platform"], dropna=False)
    .agg(
        total_revenue=("total_revenue", "sum"),
        total_cash=("total_cash", "sum"),
        user_count=("user_id", pd.Series.nunique),
        payer_count=("is_payer", "sum"),
    )
    .reset_index()
)
metrics_total["arpu"] = metrics_total["total_revenue"] / metrics_total["user_count"].replace(0, np.nan)
metrics_total["arppu"] = metrics_total["total_revenue"] / metrics_total["payer_count"].replace(0, np.nan)
metrics_total["cash_per_user"] = metrics_total["total_cash"] / metrics_total["user_count"].replace(0, np.nan)


def mean_confidence_interval(series: pd.Series, alpha: float = 0.05) -> Tuple[float, float]:
    """Возвращает границы t-интервала для среднего."""
    clean_series = series.dropna()
    n = clean_series.shape[0]
    if n == 0:
        return (np.nan, np.nan)
    if n == 1:
        mean_val = float(clean_series.iloc[0])
        return (mean_val, mean_val)
    mean_val = float(clean_series.mean())
    se = float(clean_series.std(ddof=1) / np.sqrt(n))
    delta = stats.t.ppf(1 - alpha / 2, df=n - 1) * se
    return (mean_val - delta, mean_val + delta)


ci_records: List[Dict[str, object]] = []
for (group_value, platform_value), subset in user_level.groupby(["group", "platform"], dropna=False):
    arpu_series = subset["total_revenue"]
    cash_series = subset["total_cash"]
    payers_series = subset.loc[subset["is_payer"], "total_revenue"]
    arpu_lower, arpu_upper = mean_confidence_interval(arpu_series)
    cash_lower, cash_upper = mean_confidence_interval(cash_series)
    arppu_lower, arppu_upper = mean_confidence_interval(payers_series)
    ci_records.extend(
        [
            {
                "group": group_value,
                "platform": platform_value,
                "metric": "arpu",
                "mean": float(arpu_series.mean()),
                "ci_lower": arpu_lower,
                "ci_upper": arpu_upper,
                "n": int(arpu_series.shape[0]),
            },
            {
                "group": group_value,
                "platform": platform_value,
                "metric": "cash_per_user",
                "mean": float(cash_series.mean()),
                "ci_lower": cash_lower,
                "ci_upper": cash_upper,
                "n": int(cash_series.shape[0]),
            },
            {
                "group": group_value,
                "platform": platform_value,
                "metric": "arppu",
                "mean": float(payers_series.mean()) if not payers_series.empty else np.nan,
                "ci_lower": arppu_lower,
                "ci_upper": arppu_upper,
                "n": int(payers_series.shape[0]),
            },
        ]
    )

metrics_ci = pd.DataFrame(ci_records)


# %%
# Раздел 6. Экспорт
metrics_daily_path = OUTPUT_DIR / "metrics_daily.csv"
metrics_total_path = OUTPUT_DIR / "metrics_total.csv"
metrics_ci_path = OUTPUT_DIR / "metrics_ci.csv"
user_level_path = OUTPUT_DIR / "user_level_clean.csv"
excel_path = OUTPUT_DIR / "ARPU_by_group_platform.xlsx"

metrics_daily.to_csv(metrics_daily_path, index=False)
metrics_total.to_csv(metrics_total_path, index=False)
metrics_ci.to_csv(metrics_ci_path, index=False)
user_level.to_csv(user_level_path, index=False)

arpu_pivot = metrics_total.pivot(index="platform", columns="group", values="arpu")
arppu_pivot = metrics_total.pivot(index="platform", columns="group", values="arppu")
with pd.ExcelWriter(excel_path) as writer:
    arpu_pivot.to_excel(writer, sheet_name="ARPU")
    arppu_pivot.to_excel(writer, sheet_name="ARPPU")
    metrics_total.to_excel(writer, sheet_name="Metrics_Total", index=False)

print(f"[Export] metrics_daily → {metrics_daily_path}")
print(f"[Export] metrics_total → {metrics_total_path}")
print(f"[Export] metrics_ci → {metrics_ci_path}")
print(f"[Export] user_level_clean → {user_level_path}")
print(f"[Export] ARPU/ARPPU_by_group_platform.xlsx → {excel_path}")
