# Tái lập và mở rộng DSS cho bài toán Order Promising

Mã nguồn đi kèm tiểu luận phân tích bài báo:

> Framinan, J. M., Guerrero, F., Perez-Gonzalez, P., & Toscano, S. (2024). *Matching inventory and demand in a Fast Moving Consumer Goods company: A Decision Support System*. Computers & Industrial Engineering, 194, 110377. https://doi.org/10.1016/j.cie.2024.110377

Dữ liệu của HEINEKEN Spain là dữ liệu bảo mật, nên mọi thực nghiệm ở đây dùng **dữ liệu nhân tạo**. Dữ liệu được sinh theo đúng mô tả của bài báo: testbed ở §5.1 và đặc điểm doanh nghiệp ở §3. Kết quả vì vậy dùng để kiểm chứng **xu hướng**, không thể so trực tiếp với số liệu thật của công ty.

## 1. Cấu trúc thư mục

```
dss_op/
├── op/                          # Lõi mô hình
│   ├── model.py                 # MILP (1)–(7): bản như in trong bài + bản đã sửa
│   ├── testbed.py               # Bộ sinh instance theo §5.1 (SOZ/MOZ/BOZ, HH/LH, ...)
│   └── sim.py                   # Mô phỏng nhiều ngày: FCFS, GO, RE-NOCM, RE (SATP, OCM)
├── experiments/
│   ├── exp1_testbed.py          # TN1: sàng lọc ANOVA, lưới CPU, so sánh 2 bản MILP
│   ├── exp_sim.py               # TN2 (so sánh kịch bản) và TN3a–3f (mở rộng)
│   └── analyze.py               # Sinh bảng (CSV), hình (PNG) và results/SUMMARY.md
├── tests/test_all.py            # Kiểm thử tính đúng (ràng buộc, nghiệm tối ưu ví dụ)
├── results/                     # *.csv kết quả thô, tables/, figures/, SUMMARY.md
├── logs/
└── run_pipeline.sh              # Chạy lại toàn bộ (~2 giờ trên 2 nhân CPU)
```

## 2. Cài đặt và chạy

```bash
pip install "pulp==2.9.0" numpy pandas statsmodels matplotlib   # PuLP 2.9 có sẵn CBC
python tests/test_all.py                     # kiểm thử nhanh (~1 phút)
./run_pipeline.sh                            # chạy toàn bộ thực nghiệm
python experiments/analyze.py                # chỉ tạo lại bảng/hình từ results/*.csv
```

Solver là **Coin-OR/Cbc**, giống bài báo. Có thể chạy từng phần, ví dụ `python experiments/exp_sim.py exp2 --seeds 3 --days 20`.

## 3. Ánh xạ giữa bài báo và mã nguồn

| Bài báo | Mã nguồn |
|---|---|
| §4, Hình 1: khung lặp green / orange / red | `sim.py` → `Simulator._run_day` (nhánh iterative) |
| §4.1, Hình 2: OPP 5 bước (ATP → SATP → MILP lần 1 → UATP → MILP lần 2) | `sim.py`, cùng chỗ trên |
| §4.1.1: SATPⱼ(t) = ATPⱼ − d̂ⱼ·p̂(t), d̂ là trung bình động 5 ngày | `sim.py` → `pending_ratio_curves`, `_p_hat` |
| §4.1.2: MILP (1)–(7) | `model.py` → `solve_order_promising` |
| §4.2: OCM, thương lượng đơn red với tầm nhìn tồn kho còn lại | `sim.py`, khối `# OCM` |
| §5.1: testbed, Bảng 1–3, Hình 3–5 | `testbed.py`, `exp1_testbed.py` |
| §5.2: bước tiền xử lý loại đơn không thể đáp ứng | `sim.py`, `# pre-processing` |
| §5.2: kịch bản RE, FCFS, GO, RE-NOCM, Hình 6–8 | `exp_sim.py exp2` |
| §6: hướng nghiên cứu (SATP theo sản phẩm, SLA, ...) | `exp_sim.py 3b`, `3f` |

## 4. Các giả định và cách hiểu (bài báo không nêu rõ)

1. **Ràng buộc (6)** chỉ áp dụng cho các khách có đơn trong instance. Nếu áp cho mọi khách, L bị chặn bởi những khách không đặt hàng hôm đó.
2. **Freshness coverage (HFC/MFC).** Bài viết rằng phần tồn kho bị hạn chế được gán tuổi bằng yêu cầu của một đơn cụ thể, "bắt đầu từ đơn khắt khe nhất". Nhưng batch khớp yêu cầu khắt khe nhất thì mọi đơn đều dùng được, nên không hạn chế gì cả. Nhóm vì vậy duyệt các dòng đơn theo thứ tự ngẫu nhiên.
3. **SATP theo batch.** Bài định nghĩa SATP theo sản phẩm, còn MILP làm việc theo batch. Nhóm giảm đều mọi batch của sản phẩm j theo tỷ lệ SATPⱼ/ATPⱼ. ATP ở đây là **tồn kho chưa cam kết tại thời điểm chạy**: nếu dùng ATP đầu ngày, lượng đã cam kết ở các iteration trước sẽ bị đếm hai lần.
4. **Hàng thực sự xuất kho** cho đơn được phục vụ là đúng oᵢⱼ, lấy theo nguyên tắc batch cũ trước (FEFO) trong số các batch mà MILP đã gán.
5. **Mô phỏng doanh nghiệp** (`WorldConfig`):
   - 60 SKU, chia thành 20 dòng bia × 3 quy cách. Các quy cách cùng dòng là sản phẩm thay thế khi thương lượng.
   - 200 khách HORECA (2–8 dòng/đơn, 1–12 thùng/dòng, hạn tuổi 40–90 ngày).
   - 15 khách FOOD (6–12 dòng/đơn, 10–60 thùng/dòng, hạn tuổi 30–60 ngày).
   - Trung bình khoảng 98 đơn/ngày; 35% ngày khan hàng (tồn kho bằng 65–85% nhu cầu kỳ vọng); 3% sản phẩm bị sự cố sản xuất.
   - Một phần đơn đã có sẵn lúc 8:00 (đơn đặt qua đêm).
6. **OCM.**
   - Khách đồng ý sửa đơn với xác suất `p_accept` = 0,7.
   - Với mỗi dòng gây thiếu: CMT đề xuất quy cách khác cùng dòng bia nếu còn đủ hàng hợp freshness; nếu không có thì bỏ dòng đó.
   - Đơn đã sửa quay lại ở iteration sau. Đơn red ở iteration cuối bị từ chối.
   - Khi có tầm nhìn tổng thể (global visibility), lượng đã chào cho khách trước được trừ khỏi lượng chào cho khách sau.
7. **Chỉ số.**
   - `fill_rate`: tỷ lệ đơn được giao đủ (đơn đã sửa vẫn tính là được phục vụ, như kịch bản RE trong bài).
   - `avg_sl_D`: service level trung bình theo 5 ngày của các khách.
   - `pct_cust_full`: tỷ lệ khách có mọi đơn được giao đủ.
   - `avg_sl_strict`: chỉ tính đơn vị của đúng sản phẩm khách đã đặt (loại phần hàng thay thế).

## 5. Phát hiện về mô hình MILP như được in trong bài

Mô hình (1)–(7) như được in có hai kẽ hở. Cả hai được kiểm chứng trong `tests/test_all.py::test_paper_model_flaw` và Bảng `T1c`:

- **Không có cận trên của Tᵢⱼ.** Ràng buộc (3) chỉ ép Tᵢⱼ ≥ oᵢⱼ(1−Yᵢⱼ), nên mô hình có thể cấp *dư* hàng.
- **Đơn không được phục vụ (Pᵢ = 1) vẫn được gán hàng Xᵢₗ.**

Ở cả hai trường hợp, lượng hàng này vẫn được cộng vào service level ở ràng buộc (6). Hệ quả là L của mô hình bị thổi phồng (có lúc vượt 100%), và tồn kho bị "đốt" vào những đơn không giao, nên số đơn được phục vụ giảm.

**Bản sửa** (`variant="corrected"`) thêm hai ràng buộc Tᵢⱼ ≤ oᵢⱼ và Tᵢⱼ ≤ oᵢⱼ(1−Pᵢ), đồng thời dùng |Jᵢ| thay cho |J| ở (4). Nhiều khả năng phần mềm thật của tác giả không mắc lỗi này. Phát hiện chỉ áp dụng cho công thức như được in trong bài.

## 6. Lưu ý kỹ thuật về solver CBC

CBC 2.10.3 (đi kèm PuLP 2.9) đôi khi báo trạng thái **"Optimal"** nhưng trả về một nghiệm **không khả thi**: có biến âm, vượt tồn kho batch. Hiện tượng này xảy ra khi bước tiền xử lý của CBC thất bại; log của CBC ghi "Postprocessed model is infeasible – possible tolerance issue". Nhóm gặp lỗi này ở khoảng 5% số lần giải.

`model.py` vì vậy **kiểm tra tính khả thi của mọi nghiệm**:
1. Nếu nghiệm vi phạm ràng buộc, mô hình được giải lại với tùy chọn `preprocess off`.
2. Nếu vẫn không khả thi, nghiệm được sửa theo cách tham lam: giữ lại các đơn còn đủ hàng.

Số lần phải giải lại được ghi ở cột `resolved` / `n_resolved` của kết quả. Test hồi quy: `tests/test_all.py::test_cbc_preprocessing_bug`.

Bài học cho DSS thực tế: **không nên tin trạng thái mà solver báo về** khi chưa kiểm tra lại nghiệm, nhất là khi kết quả được ERP tự động xác nhận (đơn green).

## 7. Kết quả chính

Bảng đầy đủ nằm trong `results/SUMMARY.md` và `results/tables/`; hình nằm trong `results/figures/`.

### TN1 – Tái lập testbed §5.1 (Bảng 1–3, Hình 3–5)

**Sàng lọc** (8 yếu tố × 2 mức × 2 lần lặp = 512 instance, MILP như in trong bài):
- ANOVA khớp Bảng 1 của bài báo ở **16/24** cặp yếu tố–chỉ số (`T1_anova_screening.csv`).
- Trên cùng 256 instance, **MILP đã sửa khớp 17/24, MILP như in khớp 14/24** (`T1d`). Như vậy bản sửa tái lập bài báo tốt hơn chính công thức được in.
- **α:** bài báo nói α không ảnh hưởng.
  - Với MILP như in, α ảnh hưởng mạnh đến fill rate (0,554 khi α = 0,1 so với 0,666 khi α = 0,9; p < 0,001). Khi α nhỏ, mô hình được thưởng vì thổi phồng L, nên nó dồn hàng vào các đơn không giao.
  - Với MILP đã sửa, α **không** ảnh hưởng (0,663 so với 0,673; p = 0,555), đúng như bài báo (`T1e`).
- **Kiểm định MILP như in:**
  - 60% instance có hàng được gán cho đơn không giao; 61% có cấp dư.
  - 31% instance có L > 1.
  - L trung bình mà mô hình báo là 0,90, trong khi service level tối thiểu thực tế là 0,52 (`T1c`).

**Lưới CPU** (162 tổ hợp):
- Với MILP đã sửa, CPU tăng khi freshness coverage giảm (FFC 4,2 giây → HFC 7,8 giây → MFC 9,8 giây), đúng xu hướng Hình 4 của bài báo. CPU cũng tăng theo số đơn và cỡ đơn (Hình 3).
- Với MILP như in, đường gần như phẳng (khoảng 13 giây) và 39% instance chạm giới hạn 30 giây. Bản sửa chỉ có 10% chạm giới hạn (`T2c`).

### TN2 – Tái lập §5.2 (so sánh 4 kịch bản, Hình 6–8)

3 doanh nghiệp giả lập × 20 ngày, khoảng 98 đơn/ngày.

| Kịch bản (T = 5) | % đơn giao đủ | % đơn giao đủ, ngày khan hàng | SL trung bình 5 ngày |
|---|---|---|---|
| FCFS | 0,746 | 0,677 | 0,726 |
| RE-NOCM | 0,774 | 0,710 | 0,755 |
| GO | 0,794 | 0,736 | 0,769 |
| RE | **0,899** | **0,872** | **0,857** |

- Thứ tự RE > GO > RE-NOCM > FCFS **trùng với bài báo**, và chênh lệch lớn hơn vào ngày khan hàng.
- Khoảng tin cậy bootstrap 95% của chênh lệch (bài báo không có): RE − FCFS = +0,150 [0,138; 0,163]; RE − GO = +0,103 [0,093; 0,113]; GO − RE-NOCM = +0,019 [0,016; 0,021]. Đây là cái giá của việc chia iteration (`T5`).
- **Phát hiện mới:** tối ưu theo *số đơn* hy sinh kênh FOOD (đơn lớn). Fill rate của FOOD là 0,14 với FCFS, 0,09 với GO, 0,08 với RE-NOCM. Chỉ nhờ OCM, RE mới kéo FOOD lên khoảng 0,57–0,61.

### TN3 – Các mở rộng của nhóm

| TN | Câu hỏi | Kết quả |
|---|---|---|
| 3a | MILP như in hay MILP đã sửa? | Trên testbed: fill rate 0,684 → 0,699; service level tối thiểu thực tế 0,526 → 0,598; bản sửa **không bao giờ kém hơn** về service level thực tế; CPU 6,3 → 2,0 giây. Trong mô phỏng, chênh lệch nhỏ (≤ 1 điểm). |
| 3b | SATP có giá trị không? Có nên dùng p̂ riêng cho từng sản phẩm? | SATP tăng fill rate của RE-NOCM từ 0,761 lên khoảng 0,788 (+2,7 điểm). p̂ theo sản phẩm và hệ số β cho chênh lệch < 1 điểm, nằm trong mức nhiễu. Đổi lại, SATP giảm fill rate của FOOD. |
| 3c | Nên chạy bao nhiêu iteration mỗi ngày? | RE tăng từ 0,867 (T = 2) lên khoảng 0,90 và bão hòa từ T ≈ 4. RE-NOCM giảm nhẹ khi T tăng (càng chia nhỏ càng giống FCFS). Khớp nhận định về đánh đổi ở §4. |
| 3d | Độ nhạy theo α | Với MILP đã sửa, fill rate gần như không đổi (GO 0,805–0,807). Nguyên nhân: L bị "khóa" bởi khách có đơn không thể đáp ứng (min SL = 0), nên số hạng max–min mất tác dụng. |
| 3e | Giá trị của OCM và của tầm nhìn tồn kho | Với p_accept = 0,7, khi có tầm nhìn tổng thể fill rate là 0,902, không có tầm nhìn là 0,850 (+5,2 điểm), tức là tránh chào trùng hàng thay thế. Giá trị của OCM tăng tuyến tính theo mức sẵn lòng của khách. |
| 3f | Hàm mục tiêu trọng số theo khối lượng | GO: fill rate của FOOD tăng gấp đôi (0,099 → 0,198), HORECA giảm (0,875 → 0,832), tổng thể giảm (0,805 → 0,775). RE: FOOD 0,578 → 0,624, tổng thể gần như không đổi. Đây là đánh đổi công bằng giữa hai kênh. |

## 8. Khác biệt so với bài báo (cần nêu trong tiểu luận)

- **Hình 5 (fill rate theo stock coverage):** với FSC, fill rate của nhóm bằng 1,00 ở mọi cỡ đơn, trong khi trong bài báo fill rate giảm theo cỡ đơn. Nguyên nhân là cách hiểu khác về bộ sinh freshness (mục 4, ý 2): phần tồn kho "bị hạn chế" của nhóm luôn khớp yêu cầu của một dòng đơn cụ thể, nên khi tồn kho đầy đủ thì mọi đơn đều phục vụ được.
- **Thời gian CPU** đo trên máy 2 nhân, có nhiều tiến trình chạy song song, giới hạn 30 giây (bài báo dùng 600 giây), số lần lặp ít hơn (1–2 so với 10). Chỉ nên so sánh **xu hướng**.
- **Mô phỏng doanh nghiệp** là giả định của nhóm. Mức tuyệt đối của các chỉ số không đại diện cho HEINEKEN Spain; bài báo cũng đã dịch chuyển số liệu thật.

## 9. Gợi ý dùng kết quả cho từng chương tiểu luận

| Chương | Kết quả nên dùng |
|---|---|
| 2. Research Methodology | `op/model.py` (công thức), `op/sim.py` (quy trình OPP + OCM), mục 3–4 của README |
| 3. Experiments and Results | TN1 (Bảng T1, T2c, Hình exp1_*), TN2 (T4, T5, Hình exp2) |
| 4. Evaluation | Mục 5–6 (lỗi MILP, lỗi CBC), T1c–T1e, T6b, Hình exp3a, phát hiện về kênh FOOD |
| 5. Extensions / Future directions | TN3b–3f (Hình exp3b–exp3f) |
