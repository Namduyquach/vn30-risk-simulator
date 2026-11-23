# ⚡ VN30 Crash Risk & Walk-Forward Simulator

**VN30 Crash Risk & Walk-Forward Simulator** là nền tảng phân tích định lượng (Quantitative Finance) và mô phỏng chiến lược phòng vệ rủi ro sập giá bất thường (Abnormal Tail-Risk Drop) cho toàn bộ 30 mã cổ phiếu thuộc rổ **VN30 Index**.

Ứng dụng sở hữu giao diện **Fintech Dark Theme** hiện đại, trực quan, hỗ trợ các nhà đầu tư tổ chức, chuyên viên quản trị rủi ro (Risk Managers) và thuật toán giao dịch (Quant Traders) kiểm định năng lực dự báo Out-Of-Sample của mô hình Học sâu (Deep Learning - LightGRU) so với mô hình Cây quyết định (Machine Learning - XGBoost).

---

## 🌟 Tính Năng Nổi Bật

### 🎯 Tab 1 — Ticker Risk Radar & Strategy Simulator
- **Bộ điều khiển tương tác (Sidebar Controls):**
  - Lựa chọn linh hoạt từng mã cổ phiếu trong 30 mã VN30 (ACB, HPG, VIC, VHM, FPT, SSI, VCB, v.v.).
  - Tùy chọn mô hình dự báo: **LightGRU (Recurrent Deep Learning)** hoặc **XGBoost (Gradient Boosted Trees)**.
  - Thanh trượt điều chỉnh **Risk Threshold** từ `0.20` đến `0.85` (mặc định `0.50`, bước nhảy `0.01`).
- **4 Thẻ KPI Thời Gian Thực (Real-time Metric Cards):**
  - **Precision (Độ chính xác cảnh báo):** Tỷ lệ phiên thực sự có cú sập bất thường trên tổng số phiên mô hình phát tín hiệu báo động.
  - **Recall (Tỷ lệ bao phủ cú sập):** Tỷ lệ cú sập thật được phát hiện thành công, tránh bỏ sót rủi ro.
  - **F1-Score:** Đo lường sự cân bằng hài hòa giữa Precision và Recall tại ngưỡng đang chọn.
  - **% Giảm Max Drawdown:** Phần trăm giảm mức sụt giảm tài sản tối đa khi áp dụng chiến lược né rủi ro so với Buy & Hold.
- **Biểu đồ 1 (Plotly 2 Tầng — Multi-tier Risk Radar):**
  - **Tầng trên:** Diễn biến giá điều chỉnh (`price_adjusted`) đánh dấu rõ các phiên sập thật (`target_abnormal_drop_3d == 1`) bằng điểm tròn đỏ nổi bật và các phiên mô hình cảnh báo bằng tam giác vàng.
  - **Tầng dưới:** Đường xác suất rủi ro dự báo (`gru_prob` / `xgb_prob`) kèm đường ngưỡng nét đứt màu đỏ và vùng cảnh báo trực quan.
- **Biểu đồ 2 (Backtest Chiến Lược Né Rủi Ro):**
  - So sánh đường tài sản tích lũy (**Cumulative Return %**) giữa:
    - **Buy & Hold (Mua & Nắm giữ thụ động)**
    - **Risk-Off Strategy (Chiến lược bảo vệ vốn):** Tạm đứng ngoài nắm giữ tiền mặt ở phiên $t+1$ khi xác suất rủi ro phiên $t$ vượt ngưỡng kích hoạt.
  - Bảng tổng kết hiệu suất: Lợi nhuận tích lũy, Alpha, Max Drawdown trước và sau khi né rủi ro, và tỷ lệ phiên giữ tiền mặt.

---

### 📊 Tab 2 — Walk-Forward Benchmark (Deep Learning vs. XGBoost)
- **Kiểm định Out-Of-Sample không nhìn trước (No Look-Ahead Bias):**
  - Toàn bộ 2,730 mẫu quan sát trên 30 mã VN30 từ tháng 08/2025 đến tháng 12/2025 được đánh giá theo phương pháp **Walk-Forward Validation**.
- **So sánh Hiệu suất Tổng thể (Overall Benchmark):**
  - **Overall ROC-AUC:** LightGRU (`0.6313`) vượt trội hoàn toàn so với XGBoost (`0.5006` - mức ngẫu nhiên).
  - **Overall PR-AUC:** LightGRU (`0.3640`) vượt trội `+44.8%` so với XGBoost (`0.2513`), trong khi tỷ lệ sập ngẫu nhiên cơ bản (Base Rate) là `25.42%`.
  - Trực quan hóa song song đường cong **ROC Curve** và **Precision-Recall Curve (PR Curve)**.
- **Phân tích Thích nghi với Chuyển dịch Trạng thái Thị trường (Regime Shift):**
  - Biểu đồ biến động ROC-AUC theo từng tháng:
    - **Tháng 10/2025 (Cú sập diện rộng):** XGBoost bị bẻ gãy phân lớp (`ROC-AUC = 0.3719`), trong khi LightGRU duy trì vững vàng `0.6392`.
    - **Tháng 12/2025 (Tăng tốc cuối năm):** LightGRU bứt phá ngoạn mục lên `0.7211 ROC-AUC`.
- **Bảng Xếp Hạng 30 Mã VN30 (Cross-Sectional Leaderboard):**
  - Xếp hạng độ chênh lệch năng lực dự báo rủi ro ($\Delta \text{ROC-AUC}$) của LightGRU so với XGBoost trên từng cổ phiếu riêng lẻ.

---

## 🏗️ Cấu Trúc Thư Mục

```
vn30-risk-simulator/
├── .streamlit/
│   └── config.toml                # Cấu hình giao diện chuẩn Dark Theme
├── app.py                         # Ứng dụng Web chính Streamlit + Plotly
├── requirements.txt               # Danh sách thư viện siêu nhẹ (Không cần PyTorch)
├── vn30_walkforward_results.csv   # Dữ liệu kết quả Walk-Forward rổ VN30
└── README.md                      # Tài liệu hướng dẫn sử dụng chuyên nghiệp
```

---

## ⚡ Hướng Dẫn Cài Đặt & Chạy Ứng Dụng

Ứng dụng được thiết kế tối ưu hóa, đã tính sẵn kết quả Walk-Forward nên **không cần cài đặt PyTorch** hay CUDA nặng hàng gigabyte, giúp triển khai siêu nhẹ trên mọi môi trường (Local, Streamlit Cloud, Docker, VPS).

### 1. Cài đặt các thư viện cần thiết
```bash
pip install -r requirements.txt
```

### 2. Khởi chạy ứng dụng Web
```bash
streamlit run app.py
```

Ứng dụng sẽ tự động mở trên trình duyệt tại địa chỉ: `http://localhost:8501`.

---

## 📐 Công Thức Định Lượng Cốt Lõi

1. **Precision & Recall:**
   $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}, \quad \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

2. **F1-Score:**
   $$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

3. **Lợi Nhuận Tích Lũy (Cumulative Wealth):**
   $$W_t = \prod_{i=1}^{t} (1 + R_i)$$
   Trong đó:
   - $R_i$: Lợi nhuận ngày $i$ của cổ phiếu nếu ở trạng thái đầu tư ($\text{In-Market}$).
   - $R_i = 0$: Nếu phiên $i-1$ kích hoạt tín hiệu rủi ro ($\text{Prob}_{i-1} \ge \text{Threshold}$), nhà đầu tư tạm đứng ngoài cầm tiền mặt.

4. **Max Drawdown (MDD) & % Giảm MDD:**
   $$\text{DD}_t = \frac{W_t - \max_{s \le t} W_s}{\max_{s \le t} W_s}, \quad \text{MDD} = \min_{t} \text{DD}_t$$
   $$\% \text{ Giảm MDD} = \frac{|\text{MDD}_{\text{Buy\&Hold}}| - |\text{MDD}_{\text{Strategy}}|}{|\text{MDD}_{\text{Buy\&Hold}}|} \times 100\%$$

---

## 🛡️ Bản Quyền & Giấy Phép
Dự án được xây dựng phục vụ nghiên cứu và phát triển hệ thống đầu tư định lượng trên thị trường chứng khoán Việt Nam.
Mọi thắc mắc và đóng góp xin vui lòng tạo issue hoặc pull request.
