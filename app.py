"""
VN30 Crash Risk & Walk-Forward Simulator
Institutional-Grade Quantitative Risk & AI Strategy Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    precision_recall_curve
)

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="VN30 Crash Risk & Walk-Forward Simulator",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# FINTECH DARK THEME STYLING (CSS)
# -----------------------------------------------------------------------------
FINTECH_CSS = """
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    :root {
        --bg-main: #0A0E17;
        --bg-card: #111827;
        --bg-card-hover: #172136;
        --border-color: rgba(255, 255, 255, 0.08);
        --text-main: #F8FAFC;
        --text-muted: #94A3B8;
        --accent-cyan: #00F2FE;
        --accent-blue: #38BDF8;
        --accent-emerald: #10B981;
        --accent-crimson: #EF4444;
        --accent-amber: #F59E0B;
        --accent-violet: #8B5CF6;
    }

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    code, .stCode, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Background overrides */
    .stApp {
        background-color: var(--bg-main);
        color: var(--text-main);
    }

    /* Top Header Section */
    .app-header {
        background: linear-gradient(135deg, rgba(17, 24, 39, 0.95) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.6);
        position: relative;
        overflow: hidden;
    }

    .app-header::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #00F2FE, #8B5CF6, #EF4444);
    }

    .header-title {
        font-size: 1.85rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(120deg, #FFFFFF 30%, #38BDF8 70%, #00F2FE 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .header-subtitle {
        color: var(--text-muted);
        font-size: 0.95rem;
        font-weight: 400;
        margin-top: 6px;
    }

    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(56, 189, 248, 0.12);
        color: #38BDF8;
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 9999px;
        padding: 4px 12px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .badge-pill-violet {
        background: rgba(139, 92, 246, 0.12);
        color: #A78BFA;
        border: 1px solid rgba(139, 92, 246, 0.25);
    }

    .badge-pill-emerald {
        background: rgba(16, 185, 129, 0.12);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.25);
    }

    /* Metric KPI Cards */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-bottom: 24px;
    }

    @media (max-width: 900px) {
        .kpi-container {
            grid-template-columns: repeat(2, 1fr);
        }
    }

    .kpi-card {
        background: linear-gradient(180deg, rgba(23, 32, 54, 0.7) 0%, rgba(17, 24, 39, 0.85) 100%);
        border: 1px solid var(--border-color);
        border-radius: 14px;
        padding: 18px 20px;
        transition: all 0.25s ease-in-out;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.3);
        position: relative;
    }

    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.35);
        box-shadow: 0 8px 24px rgba(0, 242, 254, 0.12);
    }

    .kpi-label {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--text-muted);
        margin-bottom: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .kpi-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.05rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        line-height: 1.1;
        margin-bottom: 6px;
    }

    .kpi-subtext {
        font-size: 0.8rem;
        color: var(--text-muted);
        display: flex;
        align-items: center;
        gap: 4px;
    }

    /* Section Cards */
    .content-box {
        background: rgba(17, 24, 39, 0.85);
        border: 1px solid var(--border-color);
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 24px;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0D1321;
        border-right: 1px solid var(--border-color);
    }

    .sidebar-header {
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--text-main);
        letter-spacing: -0.01em;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
        padding-bottom: 8px;
        border-bottom: 1px solid var(--border-color);
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: transparent;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        padding-bottom: 4px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: rgba(17, 24, 39, 0.6);
        border: 1px solid var(--border-color);
        border-radius: 10px 10px 0 0;
        color: var(--text-muted);
        font-weight: 600;
        font-size: 0.95rem;
        padding: 10px 20px;
        transition: all 0.2s;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(180deg, rgba(30, 41, 59, 0.9) 0%, rgba(17, 24, 39, 0.95) 100%) !important;
        color: #00F2FE !important;
        border-color: rgba(0, 242, 254, 0.4) !important;
        border-bottom: 2px solid #00F2FE !important;
    }

    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: #0A0E17;
    }
    ::-webkit-scrollbar-thumb {
        background: #1E293B;
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #334155;
    }
</style>
"""
st.markdown(FINTECH_CSS, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# VN30 SECTOR CLASSIFICATION & METADATA
# -----------------------------------------------------------------------------
VN30_SECTOR_MAP = {
    'ACB': 'Ngân hàng (Banking)',
    'BCM': 'Bất động sản KCN (Industrial Real Estate)',
    'BID': 'Ngân hàng (Banking)',
    'BVH': 'Bảo hiểm (Insurance)',
    'CTG': 'Ngân hàng (Banking)',
    'FPT': 'Công nghệ & Viễn thông (Technology)',
    'GAS': 'Dầu khí & Tiện ích (Gas & Utilities)',
    'GVR': 'Cao su & Hóa chất (Rubber & Chemicals)',
    'HDB': 'Ngân hàng (Banking)',
    'HPG': 'Thép & Vật liệu (Materials & Steel)',
    'MBB': 'Ngân hàng (Banking)',
    'MSN': 'Hàng tiêu dùng (Consumer Goods)',
    'MWG': 'Bán lẻ (Retail)',
    'PLX': 'Xăng dầu (Petroleum Distribution)',
    'POW': 'Điện & Năng lượng (Power Utilities)',
    'SAB': 'Bia & Đồ uống (Beverages)',
    'SHB': 'Ngân hàng (Banking)',
    'SSB': 'Ngân hàng (Banking)',
    'SSI': 'Chứng khoán (Financial Services)',
    'STB': 'Ngân hàng (Banking)',
    'TCB': 'Ngân hàng (Banking)',
    'TPB': 'Ngân hàng (Banking)',
    'VCB': 'Ngân hàng (Banking)',
    'VHM': 'Bất động sản (Real Estate)',
    'VIB': 'Ngân hàng (Banking)',
    'VIC': 'Tập đoàn đa ngành (Conglomerate)',
    'VJC': 'Hàng không (Aviation)',
    'VNM': 'Sữa & Thực phẩm (Dairy & Food)',
    'VPB': 'Ngân hàng (Banking)',
    'VRE': 'Bất động sản bán lẻ (Retail Real Estate)'
}

# -----------------------------------------------------------------------------
# DATA LOADING & CACHING
# -----------------------------------------------------------------------------
@st.cache_data
def load_walkforward_data():
    csv_path = 'vn30_walkforward_results.csv'
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.to_period('M').astype(str)
    return df

try:
    df_raw = load_walkforward_data()
except Exception as e:
    st.error(f"❌ Không thể tải file dữ liệu 'vn30_walkforward_results.csv': {e}")
    st.stop()

# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
st.sidebar.markdown('<div class="sidebar-header">⚡ THAM SỐ CHIẾN LƯỢC</div>', unsafe_allow_html=True)

ticker_list = sorted(df_raw['company'].unique())
default_ticker = 'HPG' if 'HPG' in ticker_list else ticker_list[0]
selected_ticker = st.sidebar.selectbox(
    "Mã cổ phiếu VN30",
    options=ticker_list,
    index=ticker_list.index(default_ticker),
    help="Chọn 1 trong 30 mã cổ phiếu thành phần rổ VN30 để mô phỏng rủi ro và backtest."
)

model_options = {
    "LightGRU (Recurrent Deep Learning)": "gru_prob",
    "XGBoost (Gradient Boosted Trees)": "xgb_prob"
}
selected_model_label = st.sidebar.radio(
    "Mô hình dự báo rủi ro",
    options=list(model_options.keys()),
    index=0,
    help="Chọn mô hình sinh xác suất rủi ro sập bất thường."
)
selected_prob_col = model_options[selected_model_label]
model_short_name = "LightGRU" if "GRU" in selected_model_label else "XGBoost"

risk_threshold = st.sidebar.slider(
    "Ngưỡng kích hoạt rủi ro (Risk Threshold)",
    min_value=0.20,
    max_value=0.85,
    value=0.50,
    step=0.01,
    format="%.2f",
    help="Khi xác suất rủi ro của mô hình ≥ ngưỡng này, hệ thống phát tín hiệu né rủi ro (Risk-Off)."
)

# Sidebar metadata summary
ticker_sector = VN30_SECTOR_MAP.get(selected_ticker, 'Thành viên VN30')
st.sidebar.markdown("---")
st.sidebar.markdown('<div class="sidebar-header">📋 THÔNG TIN MÃ ĐANG CHỌN</div>', unsafe_allow_html=True)
ticker_df_sidebar = df_raw[df_raw['company'] == selected_ticker].sort_values('date')
latest_price = ticker_df_sidebar['price_adjusted'].iloc[-1]
price_chg = ((latest_price / ticker_df_sidebar['price_adjusted'].iloc[0]) - 1) * 100
total_sessions = len(ticker_df_sidebar)
total_actual_drops = int(ticker_df_sidebar['target_abnormal_drop_3d'].sum())
current_warnings = int((ticker_df_sidebar[selected_prob_col] >= risk_threshold).sum())

st.sidebar.markdown(f"""
<div style="font-size: 0.85rem; line-height: 1.7; color: #94A3B8;">
    • <b>Nhóm ngành:</b> <span style="color: #F8FAFC;">{ticker_sector}</span><br>
    • <b>Giá gần nhất:</b> <span style="color: #38BDF8; font-family: 'JetBrains Mono'; font-weight: 700;">{latest_price:,.0f} VND</span><br>
    • <b>Hiệu suất kỳ:</b> <span style="color: {'#10B981' if price_chg >= 0 else '#EF4444'}; font-family: 'JetBrains Mono';">{price_chg:+.2f}%</span><br>
    • <b>Số phiên sập thật:</b> <span style="color: #EF4444; font-weight: 700;">{total_actual_drops}</span> / {total_sessions} phiên<br>
    • <b>Số phiên cảnh báo:</b> <span style="color: #F59E0B; font-weight: 700;">{current_warnings}</span> phiên (ngưỡng {risk_threshold:.2f})
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# APPLICATION HEADER
# -----------------------------------------------------------------------------
st.markdown(f"""
<div class="app-header">
    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 12px;">
        <div>
            <h1 class="header-title">
                <span>⚡ VN30 Crash Risk & Walk-Forward Simulator</span>
            </h1>
            <div class="header-subtitle">
                Hệ thống Giám sát & Mô phỏng Rủi ro Đuôi (Tail-Risk) Rổ VN30 | Đánh giá Walk-Forward Out-Of-Sample
            </div>
        </div>
        <div style="display: flex; gap: 8px; flex-wrap: wrap;">
            <span class="badge-pill">VN30 BASKET (30 TICKERS)</span>
            <span class="badge-pill badge-pill-violet">WALK-FORWARD OOS</span>
            <span class="badge-pill badge-pill-emerald">3-DAY HORIZON</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# MAIN NAVIGATION TABS
# -----------------------------------------------------------------------------
tab1, tab2 = st.tabs([
    "🎯 Tab 1 — Ticker Risk Radar & Strategy Simulator",
    "📊 Tab 2 — Walk-Forward Benchmark (Deep Learning vs. XGBoost)"
])

# =============================================================================
# TAB 1: TICKER RISK RADAR & STRATEGY SIMULATOR
# =============================================================================
with tab1:
    # Filter ticker data
    df_t = df_raw[df_raw['company'] == selected_ticker].sort_values('date').copy().reset_index(drop=True)
    
    # Target and predictions
    y_true = df_t['target_abnormal_drop_3d'].astype(int).values
    probs = df_t[selected_prob_col].values
    y_pred = (probs >= risk_threshold).astype(int)
    
    # Classification Metrics
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    
    # Backtest Calculations
    # Daily simple returns from price_adjusted
    df_t['daily_return'] = df_t['price_adjusted'].pct_change().fillna(0.0)
    
    # Strategy Rule: "tạm đứng ngoài cầm tiền mặt ở phiên tiếp theo nếu xác suất rủi ro hôm nay vượt ngưỡng"
    # signal today (t) -> position tomorrow (t+1) is Cash (0 return)
    # in_market tomorrow is True if prob today < threshold
    is_risk_session = probs >= risk_threshold
    in_market_next = pd.Series(~is_risk_session, index=df_t.index).shift(1).fillna(True).astype(bool)
    
    df_t['in_market'] = in_market_next
    df_t['strat_return'] = np.where(df_t['in_market'], df_t['daily_return'], 0.0)
    
    # Cumulative Wealth & Return
    df_t['wealth_bh'] = (1.0 + df_t['daily_return']).cumprod()
    df_t['wealth_strat'] = (1.0 + df_t['strat_return']).cumprod()
    
    df_t['cum_ret_bh'] = (df_t['wealth_bh'] - 1.0) * 100.0
    df_t['cum_ret_strat'] = (df_t['wealth_strat'] - 1.0) * 100.0
    
    # Drawdowns
    peak_bh = df_t['wealth_bh'].cummax()
    dd_bh = (df_t['wealth_bh'] - peak_bh) / peak_bh
    max_dd_bh = dd_bh.min()
    
    peak_strat = df_t['wealth_strat'].cummax()
    dd_strat = (df_t['wealth_strat'] - peak_strat) / peak_strat
    max_dd_strat = dd_strat.min()
    
    # Max Drawdown Reduction
    if abs(max_dd_bh) > 1e-6:
        mdd_reduction_pct = (abs(max_dd_bh) - abs(max_dd_strat)) / abs(max_dd_bh) * 100.0
    else:
        mdd_reduction_pct = 0.0
    mdd_delta_pts = (abs(max_dd_bh) - abs(max_dd_strat)) * 100.0

    # 4 KPI CARDS
    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-card">
            <div class="kpi-label">
                <span>Precision</span>
                <span style="color: #38BDF8;">🎯 ĐỘ CHÍNH XÁC</span>
            </div>
            <div class="kpi-value" style="color: #38BDF8;">{prec * 100:.1f}%</div>
            <div class="kpi-subtext">
                <span>TP: <b>{tp}</b> / Tổng cảnh báo: <b>{tp + fp}</b></span>
            </div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">
                <span>Recall</span>
                <span style="color: #A78BFA;">📡 ĐỘ PHỦ SẬP</span>
            </div>
            <div class="kpi-value" style="color: #A78BFA;">{rec * 100:.1f}%</div>
            <div class="kpi-subtext">
                <span>Bắt được <b>{tp}</b> / <b>{tp + fn}</b> phiên sập thật</span>
            </div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">
                <span>F1-Score</span>
                <span style="color: #F59E0B;">⚖️ CÂN BẰNG P/R</span>
            </div>
            <div class="kpi-value" style="color: #F59E0B;">{f1:.3f}</div>
            <div class="kpi-subtext">
                <span>Ngưỡng tối ưu: <b>{risk_threshold:.2f}</b></span>
            </div>
        </div>
        <div class="kpi-card" style="border-left: 3px solid {'#10B981' if mdd_reduction_pct >= 0 else '#EF4444'};">
            <div class="kpi-label">
                <span>Giảm Max Drawdown</span>
                <span style="color: {'#10B981' if mdd_reduction_pct >= 0 else '#EF4444'};">🛡️ BẢO VỆ VỐN</span>
            </div>
            <div class="kpi-value" style="color: {'#10B981' if mdd_reduction_pct >= 0 else '#EF4444'};">
                {'+' if mdd_reduction_pct >= 0 else ''}{mdd_reduction_pct:.1f}%
            </div>
            <div class="kpi-subtext">
                <span>MDD: {max_dd_bh * 100:.1f}% ➔ {max_dd_strat * 100:.1f}% ({mdd_delta_pts:+.1f}% pts)</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # BIỂU ĐỒ 1: PLOTLY 2 TẦNG (PRICE + RISK SIGNALS / RISK PROBABILITY)
    # -------------------------------------------------------------------------
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 10px; margin-bottom: 8px;">
        <h3 style="font-size: 1.15rem; font-weight: 700; color: #F8FAFC; margin: 0;">
            📈 Radar Rủi Ro 2 Tầng: Diễn Biến Giá & Xác Suất Sập Bất Thường — {selected_ticker}
        </h3>
        <span style="font-size: 0.8rem; color: #94A3B8; font-family: 'JetBrains Mono';">
            Model: {model_short_name} | Threshold: {risk_threshold:.2f}
        </span>
    </div>
    """, unsafe_allow_html=True)

    fig1 = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.64, 0.36],
        subplot_titles=(
            f"<b>Đường Giá Điều Chỉnh (VND) & Các Điểm Cảnh Báo Sập</b>",
            f"<b>Đường Xác Suất Rủi Ro ({model_short_name} Prob) & Ngưỡng Kích Hoạt</b>"
        )
    )

    # Tầng 1: Price line
    fig1.add_trace(
        go.Scatter(
            x=df_t['date'],
            y=df_t['price_adjusted'],
            name=f"Giá {selected_ticker}",
            line=dict(color="#38BDF8", width=2.2),
            hovertemplate="<b>Ngày:</b> %{x|%Y-%m-%d}<br><b>Giá:</b> %{y:,.0f} VND<extra></extra>"
        ),
        row=1, col=1
    )

    # Tầng 1: Actual Crash points (target_abnormal_drop_3d == 1)
    crash_mask = df_t['target_abnormal_drop_3d'] == 1
    if crash_mask.any():
        fig1.add_trace(
            go.Scatter(
                x=df_t.loc[crash_mask, 'date'],
                y=df_t.loc[crash_mask, 'price_adjusted'],
                mode='markers',
                name='Sập Thật (Actual Crash)',
                marker=dict(
                    symbol='circle',
                    size=10,
                    color='#EF4444',
                    line=dict(color='#FFFFFF', width=1.5)
                ),
                hovertemplate="<b>⚠️ SẬP THẬT (Actual Crash)</b><br>Ngày: %{x|%Y-%m-%d}<br>Giá: %{y:,.0f} VND<extra></extra>"
            ),
            row=1, col=1
        )

    # Tầng 1: Model warning points (prob >= threshold)
    warn_mask = df_t[selected_prob_col] >= risk_threshold
    if warn_mask.any():
        fig1.add_trace(
            go.Scatter(
                x=df_t.loc[warn_mask, 'date'],
                y=df_t.loc[warn_mask, 'price_adjusted'],
                mode='markers',
                name=f'Cảnh Báo Vượt Ngưỡng ({model_short_name} ≥ {risk_threshold:.2f})',
                marker=dict(
                    symbol='triangle-up',
                    size=11,
                    color='#F59E0B',
                    line=dict(color='#111827', width=1.2)
                ),
                hovertemplate="<b>🔔 CẢNH BÁO RỦI RO</b><br>Ngày: %{x|%Y-%m-%d}<br>Giá: %{y:,.0f} VND<extra></extra>"
            ),
            row=1, col=1
        )

    # Tầng 2: Risk Probability Line with area fill
    fig1.add_trace(
        go.Scatter(
            x=df_t['date'],
            y=df_t[selected_prob_col],
            name=f"Xác suất {model_short_name}",
            line=dict(color="#A78BFA", width=2),
            fill='tozeroy',
            fillcolor='rgba(139, 92, 246, 0.15)',
            hovertemplate="<b>Ngày:</b> %{x|%Y-%m-%d}<br><b>Xác suất rủi ro:</b> %{y:.4f}<extra></extra>"
        ),
        row=2, col=1
    )

    # Tầng 2: Dashed Red Threshold Line
    fig1.add_trace(
        go.Scatter(
            x=[df_t['date'].min(), df_t['date'].max()],
            y=[risk_threshold, risk_threshold],
            mode='lines',
            name=f'Ngưỡng rủi ro ({risk_threshold:.2f})',
            line=dict(color='#EF4444', width=2, dash='dash'),
            hoverinfo='skip'
        ),
        row=2, col=1
    )

    fig1.update_layout(
        template='plotly_dark',
        paper_bgcolor='#0A0E17',
        plot_bgcolor='#0F172A',
        height=620,
        margin=dict(l=50, r=30, t=50, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(15, 23, 42, 0.8)',
            bordercolor='rgba(255, 255, 255, 0.1)',
            borderwidth=1,
            font=dict(size=11)
        ),
        hovermode='x unified',
        font=dict(family="Plus Jakarta Sans", color="#F8FAFC")
    )
    fig1.update_xaxes(
        gridcolor='rgba(255, 255, 255, 0.05)',
        zerolinecolor='rgba(255, 255, 255, 0.1)',
        showspikes=True,
        spikethickness=1,
        spikedash='dot',
        spikecolor='#38BDF8'
    )
    fig1.update_yaxes(
        gridcolor='rgba(255, 255, 255, 0.05)',
        zerolinecolor='rgba(255, 255, 255, 0.1)'
    )
    fig1.update_yaxes(title_text="Giá (VND)", row=1, col=1)
    fig1.update_yaxes(title_text="Xác suất (0-1)", range=[-0.02, 1.05], row=2, col=1)

    st.plotly_chart(fig1, use_container_width=True)

    # -------------------------------------------------------------------------
    # BIỂU ĐỒ 2: BACKTEST ĐƠN GIẢN (CUMULATIVE RETURN & RISK-OFF SIMULATOR)
    # -------------------------------------------------------------------------
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 18px; margin-bottom: 8px;">
        <h3 style="font-size: 1.15rem; font-weight: 700; color: #F8FAFC; margin: 0;">
            💰 Mô Phỏng Chiến Lược Né Rủi Ro (Risk-Off) vs. Mua & Nắm Giữ (Buy & Hold)
        </h3>
        <span style="font-size: 0.8rem; color: #94A3B8; font-family: 'JetBrains Mono';">
            Chiến lược: Tạm cầm tiền mặt ở phiên t+1 nếu Prob(t) ≥ {risk_threshold:.2f}
        </span>
    </div>
    """, unsafe_allow_html=True)

    final_bh_ret = df_t['cum_ret_bh'].iloc[-1]
    final_strat_ret = df_t['cum_ret_strat'].iloc[-1]
    alpha_ret = final_strat_ret - final_bh_ret
    cash_sessions = int((~df_t['in_market']).sum())
    cash_pct = (cash_sessions / len(df_t)) * 100

    fig2 = go.Figure()

    # Buy & Hold Line
    fig2.add_trace(
        go.Scatter(
            x=df_t['date'],
            y=df_t['cum_ret_bh'],
            name='Buy & Hold (Nắm Giữ Thụ Động)',
            line=dict(color='#94A3B8', width=2, dash='dot'),
            hovertemplate="<b>Buy & Hold:</b> %{y:+.2f}%<extra></extra>"
        )
    )

    # Strategy Line
    fig2.add_trace(
        go.Scatter(
            x=df_t['date'],
            y=df_t['cum_ret_strat'],
            name=f'Risk-Off Strategy ({model_short_name} ≥ {risk_threshold:.2f})',
            line=dict(color='#10B981', width=3),
            fill='tonexty',
            fillcolor='rgba(16, 185, 129, 0.08)',
            hovertemplate="<b>Risk-Off Strategy:</b> %{y:+.2f}%<extra></extra>"
        )
    )

    # Zero Line
    fig2.add_hline(y=0, line_width=1, line_dash="solid", line_color="rgba(255, 255, 255, 0.2)")

    fig2.update_layout(
        template='plotly_dark',
        paper_bgcolor='#0A0E17',
        plot_bgcolor='#0F172A',
        height=420,
        margin=dict(l=50, r=30, t=30, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(15, 23, 42, 0.8)',
            bordercolor='rgba(255, 255, 255, 0.1)',
            borderwidth=1
        ),
        hovermode='x unified',
        font=dict(family="Plus Jakarta Sans", color="#F8FAFC"),
        yaxis=dict(
            title="Lợi Nhuận Tích Lũy (%)",
            gridcolor='rgba(255, 255, 255, 0.05)',
            ticksuffix="%"
        ),
        xaxis=dict(
            gridcolor='rgba(255, 255, 255, 0.05)',
            showspikes=True,
            spikethickness=1,
            spikedash='dot',
            spikecolor='#10B981'
        )
    )

    st.plotly_chart(fig2, use_container_width=True)

    # Strategy Statistics Breakdown Ribbon
    col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)
    with col_s1:
        st.metric(
            label="Lợi Nhuận Buy & Hold",
            value=f"{final_bh_ret:+.2f}%",
            help="Hiệu suất khi nắm giữ liên tục không bán."
        )
    with col_s2:
        st.metric(
            label="Lợi Nhuận Chiến Lược",
            value=f"{final_strat_ret:+.2f}%",
            delta=f"{alpha_ret:+.2f}% Alpha",
            help="Lợi nhuận khi đứng ngoài cầm tiền mặt ở những phiên bị cảnh báo rủi ro."
        )
    with col_s3:
        st.metric(
            label="Max Drawdown B&H",
            value=f"{max_dd_bh * 100:.2f}%",
            help="Mức sụt giảm tài sản lớn nhất từ đỉnh khi Buy & Hold."
        )
    with col_s4:
        st.metric(
            label="Max Drawdown Chiến Lược",
            value=f"{max_dd_strat * 100:.2f}%",
            delta=f"{mdd_delta_pts:+.2f}% bảo vệ vốn",
            delta_color="normal",
            help="Mức sụt giảm lớn nhất từ đỉnh khi né các phiên cảnh báo sập."
        )
    with col_s5:
        st.metric(
            label="Tỷ Lệ Giữ Tiền Mặt",
            value=f"{cash_pct:.1f}%",
            delta=f"{cash_sessions} / {len(df_t)} phiên",
            delta_color="off",
            help="Tỷ lệ phiên đứng ngoài cầm tiền mặt để né cú sập bất thường."
        )

# =============================================================================
# TAB 2: WALK-FORWARD BENCHMARK (DEEP LEARNING VS. XGBOOST)
# =============================================================================
with tab2:
    st.markdown("""
    <div style="background: rgba(17, 24, 39, 0.85); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 14px; padding: 20px; margin-bottom: 24px;">
        <h3 style="margin-top: 0; font-size: 1.25rem; color: #F8FAFC; display: flex; align-items: center; gap: 8px;">
            <span>🔬 Walk-Forward Out-Of-Sample Benchmark: LightGRU vs. XGBoost</span>
        </h3>
        <p style="color: #94A3B8; font-size: 0.92rem; line-height: 1.6; margin-bottom: 0;">
            Quy trình kiểm định <b>Walk-Forward</b> mô phỏng chính xác điều kiện triển khai thực tế trên toàn bộ <b>30 mã cổ phiếu rổ VN30</b> (2,730 mẫu quan sát) từ tháng 08/2025 đến 12/2025.
            Tại mỗi bước thời gian, mô hình chỉ được huấn luyện trên dữ liệu quá khứ và dự báo rủi ro sập bất thường trong 3 phiên tiếp theo (<code>target_abnormal_drop_3d</code>) hoàn toàn <b>Out-Of-Sample</b>, triệt tiêu rủi ro nhìn trước dữ liệu (Look-ahead Bias).
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Calculate overall metrics
    all_y_true = df_raw['target_abnormal_drop_3d'].astype(int).values
    all_gru_prob = df_raw['gru_prob'].values
    all_xgb_prob = df_raw['xgb_prob'].values
    base_rate = float(np.mean(all_y_true))

    overall_roc_gru = roc_auc_score(all_y_true, all_gru_prob)
    overall_roc_xgb = roc_auc_score(all_y_true, all_xgb_prob)
    overall_pr_gru = average_precision_score(all_y_true, all_gru_prob)
    overall_pr_xgb = average_precision_score(all_y_true, all_xgb_prob)

    roc_lift = ((overall_roc_gru - overall_roc_xgb) / overall_roc_xgb) * 100.0
    pr_lift = ((overall_pr_gru - overall_pr_xgb) / overall_pr_xgb) * 100.0

    # Overall Metric Comparison Banner
    col_b1, col_b2, col_b3, col_b4 = st.columns(4)
    with col_b1:
        st.markdown(f"""
        <div class="kpi-card" style="border-left: 3px solid #8B5CF6;">
            <div class="kpi-label">LightGRU ROC-AUC</div>
            <div class="kpi-value" style="color: #A78BFA;">{overall_roc_gru:.4f}</div>
            <div class="kpi-subtext" style="color: #10B981; font-weight: 600;">
                +{roc_lift:.1f}% vs. XGBoost (+{overall_roc_gru - overall_roc_xgb:.4f})
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_b2:
        st.markdown(f"""
        <div class="kpi-card" style="border-left: 3px solid #F97316;">
            <div class="kpi-label">XGBoost ROC-AUC</div>
            <div class="kpi-value" style="color: #FB923C;">{overall_roc_xgb:.4f}</div>
            <div class="kpi-subtext" style="color: #94A3B8;">
                Baseline ngẫu nhiên: 0.5000
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_b3:
        st.markdown(f"""
        <div class="kpi-card" style="border-left: 3px solid #00F2FE;">
            <div class="kpi-label">LightGRU PR-AUC</div>
            <div class="kpi-value" style="color: #00F2FE;">{overall_pr_gru:.4f}</div>
            <div class="kpi-subtext" style="color: #10B981; font-weight: 600;">
                +{pr_lift:.1f}% vs. XGBoost (+{overall_pr_gru - overall_pr_xgb:.4f})
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_b4:
        st.markdown(f"""
        <div class="kpi-card" style="border-left: 3px solid #64748B;">
            <div class="kpi-label">XGBoost PR-AUC</div>
            <div class="kpi-value" style="color: #94A3B8;">{overall_pr_xgb:.4f}</div>
            <div class="kpi-subtext" style="color: #94A3B8;">
                Tỷ lệ sập mẫu (Baseline): {base_rate * 100:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # Visualizations Section: Curves and Comparisons
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown("<h4 style='font-size: 1.05rem; font-weight: 700; color: #F8FAFC;'>📉 Đường Cong ROC (Receiver Operating Characteristic)</h4>", unsafe_allow_html=True)
        fpr_gru, tpr_gru, _ = roc_curve(all_y_true, all_gru_prob)
        fpr_xgb, tpr_xgb, _ = roc_curve(all_y_true, all_xgb_prob)

        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr_gru, y=tpr_gru,
            mode='lines',
            name=f'LightGRU (AUC = {overall_roc_gru:.4f})',
            line=dict(color='#8B5CF6', width=2.5)
        ))
        fig_roc.add_trace(go.Scatter(
            x=fpr_xgb, y=tpr_xgb,
            mode='lines',
            name=f'XGBoost (AUC = {overall_roc_xgb:.4f})',
            line=dict(color='#F97316', width=2, dash='dot')
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Guess (AUC = 0.5000)',
            line=dict(color='#64748B', width=1, dash='dash')
        ))
        fig_roc.update_layout(
            template='plotly_dark',
            paper_bgcolor='#0A0E17',
            plot_bgcolor='#0F172A',
            height=360,
            margin=dict(l=40, r=20, t=20, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(title="False Positive Rate (FPR)", gridcolor='rgba(255, 255, 255, 0.05)'),
            yaxis=dict(title="True Positive Rate (TPR)", gridcolor='rgba(255, 255, 255, 0.05)'),
            font=dict(family="Plus Jakarta Sans", color="#F8FAFC")
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    with col_chart2:
        st.markdown("<h4 style='font-size: 1.05rem; font-weight: 700; color: #F8FAFC;'>🎯 Đường Cong Precision-Recall (PR Curve)</h4>", unsafe_allow_html=True)
        prec_gru, rec_gru, _ = precision_recall_curve(all_y_true, all_gru_prob)
        prec_xgb, rec_xgb, _ = precision_recall_curve(all_y_true, all_xgb_prob)

        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(
            x=rec_gru, y=prec_gru,
            mode='lines',
            name=f'LightGRU (PR-AUC = {overall_pr_gru:.4f})',
            line=dict(color='#00F2FE', width=2.5)
        ))
        fig_pr.add_trace(go.Scatter(
            x=rec_xgb, y=prec_xgb,
            mode='lines',
            name=f'XGBoost (PR-AUC = {overall_pr_xgb:.4f})',
            line=dict(color='#FB923C', width=2, dash='dot')
        ))
        fig_pr.add_hline(
            y=base_rate,
            line_dash="dash",
            line_color="#64748B",
            annotation_text=f"Baseline: {base_rate:.3f}",
            annotation_position="bottom right"
        )
        fig_pr.update_layout(
            template='plotly_dark',
            paper_bgcolor='#0A0E17',
            plot_bgcolor='#0F172A',
            height=360,
            margin=dict(l=40, r=20, t=20, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(title="Recall", gridcolor='rgba(255, 255, 255, 0.05)'),
            yaxis=dict(title="Precision", gridcolor='rgba(255, 255, 255, 0.05)'),
            font=dict(family="Plus Jakarta Sans", color="#F8FAFC")
        )
        st.plotly_chart(fig_pr, use_container_width=True)

    # -------------------------------------------------------------------------
    # MONTHLY ROC-AUC COMPARISON (REGIME SHIFT ANALYSIS)
    # -------------------------------------------------------------------------
    st.markdown("""
    <div style="margin-top: 24px; margin-bottom: 8px;">
        <h3 style="font-size: 1.25rem; font-weight: 700; color: #F8FAFC; margin: 0;">
            🌪️ Khả Năng Thích Nghi Với Regime Shift: So Sánh ROC-AUC Theo Từng Tháng
        </h3>
        <p style="color: #94A3B8; font-size: 0.9rem; margin-top: 4px;">
            Thị trường chứng khoán liên tục chuyển đổi trạng thái (chuyển biến thanh khoản, biến động, tin tức). Dưới đây là hiệu suất Out-Of-Sample của 2 mô hình theo từng tháng:
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Monthly calculation
    monthly_data = []
    for month_str, group in df_raw.groupby('month'):
        y_g = group['target_abnormal_drop_3d'].astype(int).values
        n_obs = len(group)
        n_pos = int(np.sum(y_g))
        pos_ratio = n_pos / n_obs if n_obs > 0 else 0
        
        if len(np.unique(y_g)) > 1:
            gru_auc = roc_auc_score(y_g, group['gru_prob'])
            xgb_auc = roc_auc_score(y_g, group['xgb_prob'])
            gru_pr = average_precision_score(y_g, group['gru_prob'])
            xgb_pr = average_precision_score(y_g, group['xgb_prob'])
        else:
            gru_auc, xgb_auc, gru_pr, xgb_pr = np.nan, np.nan, np.nan, np.nan

        monthly_data.append({
            'month': month_str,
            'samples': n_obs,
            'crash_events': n_pos,
            'crash_pct': pos_ratio * 100,
            'gru_auc': gru_auc,
            'xgb_auc': xgb_auc,
            'gru_pr': gru_pr,
            'xgb_pr': xgb_pr,
            'auc_delta': gru_auc - xgb_auc
        })

    df_monthly = pd.DataFrame(monthly_data)

    fig_month = go.Figure()

    # LightGRU Monthly AUC Bar + Line
    fig_month.add_trace(go.Bar(
        x=df_monthly['month'],
        y=df_monthly['gru_auc'],
        name='LightGRU ROC-AUC',
        marker_color='#8B5CF6',
        opacity=0.85,
        hovertemplate="<b>Tháng:</b> %{x}<br><b>LightGRU AUC:</b> %{y:.4f}<extra></extra>"
    ))

    # XGBoost Monthly AUC Bar + Line
    fig_month.add_trace(go.Bar(
        x=df_monthly['month'],
        y=df_monthly['xgb_auc'],
        name='XGBoost ROC-AUC',
        marker_color='#F97316',
        opacity=0.85,
        hovertemplate="<b>Tháng:</b> %{x}<br><b>XGBoost AUC:</b> %{y:.4f}<extra></extra>"
    ))

    # Benchmark 0.50 Line
    fig_month.add_hline(
        y=0.50,
        line_dash="dash",
        line_color="#EF4444",
        line_width=1.5,
        annotation_text="Ngẫu nhiên (0.50)",
        annotation_position="top left",
        annotation_font_color="#EF4444"
    )

    fig_month.update_layout(
        template='plotly_dark',
        paper_bgcolor='#0A0E17',
        plot_bgcolor='#0F172A',
        barmode='group',
        height=420,
        margin=dict(l=40, r=20, t=30, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(title="Tháng Giao Dịch (Walk-Forward Horizon)", gridcolor='rgba(255, 255, 255, 0.05)'),
        yaxis=dict(title="ROC-AUC Score", range=[0.25, 0.82], gridcolor='rgba(255, 255, 255, 0.05)'),
        font=dict(family="Plus Jakarta Sans", color="#F8FAFC")
    )
    st.plotly_chart(fig_month, use_container_width=True)

    # Monthly Insights & Table
    col_t1, col_t2 = st.columns([1.2, 1.8])

    with col_t1:
        st.markdown("""
        <div style="background: rgba(17, 24, 39, 0.7); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 18px;">
            <h4 style="color: #38BDF8; font-size: 1rem; margin-top: 0;">💡 Phân Tích Thích Nghi Regime Shift</h4>
            <ul style="color: #94A3B8; font-size: 0.85rem; line-height: 1.7; padding-left: 18px; margin-bottom: 0;">
                <li><b>Tháng 10/2025 (Cú sốc sụt giảm mạnh):</b> Số lượng phiên sập lên đỉnh <b>205 phiên (31.1%)</b>. Mô hình XGBoost bị gãy hiệu suất nghiêm trọng (ROC-AUC sụt xuống <b>0.3719</b> - nghịch đảo phân lớp do cây quyết định overfit trạng thái bình thường). Trong khi đó, <b>LightGRU</b> giữ vững phong độ với <b>0.6392 AUC</b>.</li>
                <li><b>Tháng 12/2025 (Sóng tăng tốc & thanh khoản lớn):</b> LightGRU bứt phá ngoạn mục đạt <b>0.7211 ROC-AUC</b> nhờ năng lực học các phụ thuộc chuỗi thời gian phi tuyến (recurrent temporal memory).</li>
                <li><b>Kết luận cốt lõi:</b> Deep Learning (LightGRU) thể hiện ưu thế vượt trội về tính ổn định và khả năng khái quát hóa xuyên qua các giai đoạn Regime Shift so với mô hình cây dạng tĩnh (Tree-based).</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_t2:
        st.markdown("<h4 style='font-size: 1rem; font-weight: 700; color: #F8FAFC; margin-top: 0;'>📋 Bảng Thống Kê Chi Tiết Từng Tháng</h4>", unsafe_allow_html=True)
        display_month_df = df_monthly[[
            'month', 'samples', 'crash_events', 'crash_pct', 'gru_auc', 'xgb_auc', 'auc_delta'
        ]].copy()
        display_month_df.columns = [
            'Tháng', 'Tổng Mẫu', 'Số Phiên Sập', 'Tỷ Lệ Sập (%)', 'LightGRU AUC', 'XGBoost AUC', 'Chênh Lệch (Δ)'
        ]
        st.dataframe(
            display_month_df,
            column_config={
                "LightGRU AUC": st.column_config.ProgressColumn(
                    "LightGRU AUC",
                    help="ROC-AUC Score của mô hình LightGRU",
                    format="%.4f",
                    min_value=0.0,
                    max_value=1.0,
                ),
                "XGBoost AUC": st.column_config.ProgressColumn(
                    "XGBoost AUC",
                    help="ROC-AUC Score của mô hình XGBoost",
                    format="%.4f",
                    min_value=0.0,
                    max_value=1.0,
                ),
                "Chênh Lệch (Δ)": st.column_config.NumberColumn(
                    "Chênh Lệch (Δ)",
                    help="Hiệu số ROC-AUC (LightGRU - XGBoost)",
                    format="%+.4f",
                ),
                "Tỷ Lệ Sập (%)": st.column_config.NumberColumn(
                    "Tỷ Lệ Sập (%)",
                    format="%.1f%%",
                )
            },
            use_container_width=True,
            hide_index=True
        )

    # -------------------------------------------------------------------------
    # CROSS-SECTIONAL TICKER LEADERBOARD
    # -------------------------------------------------------------------------
    st.markdown("""
    <div style="margin-top: 24px; margin-bottom: 12px;">
        <h3 style="font-size: 1.25rem; font-weight: 700; color: #F8FAFC; margin: 0;">
            🏆 Bảng Xếp Hạng ROC-AUC 30 Mã VN30 (Cross-Sectional Leaderboard)
        </h3>
        <p style="color: #94A3B8; font-size: 0.9rem; margin-top: 4px;">
            Đánh giá chi tiết năng lực dự báo rủi ro của từng mã cổ phiếu thành phần rổ VN30 trong suốt giai đoạn Walk-Forward:
        </p>
    </div>
    """, unsafe_allow_html=True)

    ticker_perf = []
    for comp, g in df_raw.groupby('company'):
        y_c = g['target_abnormal_drop_3d'].astype(int).values
        if len(np.unique(y_c)) > 1:
            g_auc = roc_auc_score(y_c, g['gru_prob'])
            x_auc = roc_auc_score(y_c, g['xgb_prob'])
            g_pr = average_precision_score(y_c, g['gru_prob'])
            x_pr = average_precision_score(y_c, g['xgb_prob'])
        else:
            g_auc, x_auc, g_pr, x_pr = np.nan, np.nan, np.nan, np.nan
        
        ticker_perf.append({
            'company': comp,
            'sector': VN30_SECTOR_MAP.get(comp, 'VN30 Basket'),
            'crashes': int(y_c.sum()),
            'gru_auc': g_auc,
            'xgb_auc': x_auc,
            'gru_pr': g_pr,
            'xgb_pr': x_pr,
            'delta_auc': g_auc - x_auc,
            'winner': 'LightGRU' if g_auc > x_auc else 'XGBoost'
        })

    df_tp = pd.DataFrame(ticker_perf).sort_values('gru_auc', ascending=False).reset_index(drop=True)

    # Horizontal Bar chart showing delta
    fig_delta = go.Figure()
    colors = ['#10B981' if d >= 0 else '#EF4444' for d in df_tp['delta_auc']]
    fig_delta.add_trace(go.Bar(
        x=df_tp['delta_auc'],
        y=df_tp['company'],
        orientation='h',
        marker_color=colors,
        hovertemplate="<b>Mã:</b> %{y}<br><b>Delta (GRU - XGB):</b> %{x:+.4f}<extra></extra>"
    ))
    fig_delta.update_layout(
        template='plotly_dark',
        paper_bgcolor='#0A0E17',
        plot_bgcolor='#0F172A',
        height=680,
        margin=dict(l=50, r=30, t=20, b=40),
        xaxis=dict(title="Chênh lệch ROC-AUC (LightGRU - XGBoost)", gridcolor='rgba(255, 255, 255, 0.05)'),
        yaxis=dict(autorange="reversed", gridcolor='rgba(255, 255, 255, 0.05)'),
        font=dict(family="Plus Jakarta Sans", color="#F8FAFC")
    )
    fig_delta.add_vline(x=0, line_dash="solid", line_color="rgba(255, 255, 255, 0.3)")

    col_l1, col_l2 = st.columns([1.1, 1.4])
    with col_l1:
        st.markdown("<h4 style='font-size: 1rem; font-weight: 700; color: #F8FAFC;'>📊 Mức Vượt Trội Của LightGRU (Δ ROC-AUC)</h4>", unsafe_allow_html=True)
        st.plotly_chart(fig_delta, use_container_width=True)

    with col_l2:
        st.markdown("<h4 style='font-size: 1rem; font-weight: 700; color: #F8FAFC;'>📑 Bảng Chi Tiết Cả 30 Mã VN30</h4>", unsafe_allow_html=True)
        df_show_tp = df_tp[['company', 'sector', 'crashes', 'gru_auc', 'xgb_auc', 'delta_auc', 'winner']].copy()
        df_show_tp.columns = ['Mã CP', 'Nhóm Ngành', 'Phiên Sập', 'LightGRU AUC', 'XGBoost AUC', 'Chênh Lệch (Δ)', 'Mô Hình Thắng']
        st.dataframe(
            df_show_tp,
            column_config={
                "LightGRU AUC": st.column_config.ProgressColumn(
                    "LightGRU AUC",
                    help="ROC-AUC Score của LightGRU",
                    format="%.4f",
                    min_value=0.0,
                    max_value=1.0,
                ),
                "XGBoost AUC": st.column_config.ProgressColumn(
                    "XGBoost AUC",
                    help="ROC-AUC Score của XGBoost",
                    format="%.4f",
                    min_value=0.0,
                    max_value=1.0,
                ),
                "Chênh Lệch (Δ)": st.column_config.NumberColumn(
                    "Chênh Lệch (Δ)",
                    help="Hiệu số ROC-AUC (LightGRU - XGBoost)",
                    format="%+.4f",
                ),
                "Phiên Sập": st.column_config.NumberColumn(
                    "Phiên Sập",
                    format="%d",
                ),
                "Mô Hình Thắng": st.column_config.TextColumn(
                    "Mô Hình Thắng"
                )
            },
            use_container_width=True,
            height=630,
            hide_index=True
        )

# -----------------------------------------------------------------------------
# APPLICATION FOOTER
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748B; font-size: 0.8rem; padding: 12px 0;">
    ⚡ <b>VN30 Crash Risk & Walk-Forward Simulator</b> | Được thiết kế chuyên biệt cho Quản trị Rủi ro Danh mục Định lượng & AI Trading.<br>
    Dữ liệu kiểm thử: 30 cổ phiếu rổ VN30 | Đánh giá Walk-Forward Out-Of-Sample không rò rỉ dữ liệu (No Look-Ahead Bias).
</div>
""", unsafe_allow_html=True)
