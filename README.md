# DSS cho bài toán Order Promising: tái lập và mở rộng

Đây là mã nguồn nhóm dùng cho tiểu luận phân tích bài báo:

> Framinan, J. M., Guerrero, F., Perez-Gonzalez, P., & Toscano, S. (2024). *Matching inventory and demand in a Fast Moving Consumer Goods company: A Decision Support System*. Computers & Industrial Engineering, 194, 110377. https://doi.org/10.1016/j.cie.2024.110377

Bài báo dùng dữ liệu của HEINEKEN Spain, vốn không được công bố. Vì vậy toàn bộ thực nghiệm trong repo chạy trên dữ liệu nhân tạo, sinh theo mô tả testbed ở mục 5.1 và mô tả doanh nghiệp ở mục 3 của bài. Các con số dưới đây chỉ dùng để so xu hướng với bài báo, không so trực tiếp với số liệu của công ty.

## 1. Cấu trúc thư mục

```
dss_op/
├── op/
│   ├── model.py            # MILP (1)-(7), gồm bản in trong bài và bản đã sửa
│   ├── testbed.py          # sinh instance theo mục 5.1 (SOZ/MOZ/BOZ, HH/LH, ...)
│   └── sim.py              # mô phỏng nhiều ngày: FCFS, GO, RE-NOCM, RE (SATP, OCM)
├── experiments/
│   ├── exp1_testbed.py     # TN1: sàng lọc ANOVA, lưới CPU, so sánh hai bản MILP
│   ├── exp_sim.py          # TN2 (so sánh kịch bản) và TN3a-3f (mở rộng)
│   └── analyze.py          # xuất bảng CSV, hình PNG và results/SUMMARY.md
├── tests/test_all.py       # kiểm thử ràng buộc và vài ví dụ có nghiệm biết trước
├── results/                # kết quả thô (*.csv), tables/, figures/, SUMMARY.md
├── logs/
└── run_pipeline.sh         # chạy lại toàn bộ, khoảng 2 giờ trên máy 2 nhân
```

## 2. Cài đặt và chạy

```bash
pip install -r requirements.txt     # PuLP 2.9 đã kèm sẵn CBC
python tests/test_all.py            # kiểm thử, mất khoảng 1 phút
./run_pipeline.sh                   # chạy toàn bộ thực nghiệm
python experiments/analyze.py       # chỉ vẽ lại bảng và hình từ results/*.csv
```

Solver là Coin-OR/Cbc, giống bài báo. Có thể chạy riêng từng thực nghiệm, ví dụ `python experiments/exp_sim.py exp2 --seeds 3 --days 20`.

## 3. Đối chiếu bài báo với mã nguồn

| Bài báo | Mã nguồn |
|---|---|
| Mục 4, Hình 1: vòng lặp green / orange / red | `sim.py`, hàm `Simulator._run_day` (nhánh iterative) |
| Mục 4.1, Hình 2: OPP 5 bước (ATP, SATP, MILP lần 1, UATP, MILP lần 2) | `sim.py`, cùng hàm trên |
| Mục 4.1.1: SATPⱼ(t) = ATPⱼ − d̂ⱼ·p̂(t), d̂ là trung bình động 5 ngày | `sim.py`, `pending_ratio_curves` và `_p_hat` |
| Mục 4.1.2: MILP (1)-(7) | `model.py`, `solve_order_promising` |
| Mục 4.2: OCM, thương lượng lại các đơn red | `sim.py`, đoạn `# OCM` |
| Mục 5.1: testbed, Bảng 1-3, Hình 3-5 | `testbed.py`, `exp1_testbed.py` |
| Mục 5.2: tiền xử lý, loại các đơn chắc chắn không đáp ứng được | `sim.py`, đoạn `# pre-processing` |
| Mục 5.2: các kịch bản RE, FCFS, GO, RE-NOCM, Hình 6-8 | `exp_sim.py exp2` |
| Mục 6: hướng nghiên cứu (SATP theo sản phẩm, ...) | `exp_sim.py 3b`, `3f` |

## 4. Những chỗ bài báo không nói rõ và cách nhóm xử lý

1. Ràng buộc (6) chỉ áp dụng cho những khách có đơn trong instance. Nếu áp cho mọi khách thì L sẽ bị chặn bởi cả những khách hôm đó không đặt hàng.
2. Freshness coverage (HFC/MFC): bài báo viết rằng phần tồn kho bị hạn chế được gán tuổi theo yêu cầu của từng dòng đơn, "bắt đầu từ đơn khắt khe nhất". Tuy nhiên batch khớp với yêu cầu khắt khe nhất thì đơn nào cũng dùng được, nên làm như vậy thực chất không hạn chế gì. Nhóm duyệt các dòng đơn theo thứ tự ngẫu nhiên.
3. SATP theo batch: bài báo định nghĩa SATP theo sản phẩm, trong khi MILP làm việc theo batch. Nhóm giảm đều mọi batch của sản phẩm j theo tỷ lệ SATPⱼ/ATPⱼ. ATP ở đây là tồn kho chưa cam kết tại thời điểm chạy. Nếu lấy ATP đầu ngày thì phần đã cam kết ở các iteration trước bị tính hai lần.
4. Với đơn được phục vụ, lượng xuất kho đúng bằng oᵢⱼ và lấy từ batch cũ trước (FEFO), trong số các batch mà MILP đã gán.
5. Doanh nghiệp giả lập (`WorldConfig`) có 60 SKU (20 dòng bia, mỗi dòng 3 quy cách; các quy cách cùng dòng có thể thay thế nhau khi thương lượng), 200 khách HORECA (2-8 dòng/đơn, 1-12 thùng/dòng, hạn tuổi 40-90 ngày) và 15 khách FOOD (6-12 dòng/đơn, 10-60 thùng/dòng, hạn tuổi 30-60 ngày). Trung bình khoảng 98 đơn/ngày. 35% số ngày bị khan hàng, khi đó tồn kho chỉ bằng 65-85% nhu cầu kỳ vọng. 3% sản phẩm gặp sự cố sản xuất. Một phần đơn đã có sẵn lúc 8:00 do khách đặt từ tối hôm trước.
6. OCM: khách đồng ý sửa đơn với xác suất `p_accept` = 0,7. Với mỗi dòng gây thiếu hàng, bộ phận CMT đề xuất quy cách khác cùng dòng bia nếu còn đủ hàng đạt hạn tuổi, nếu không thì bỏ dòng đó. Đơn đã sửa được đưa vào iteration tiếp theo; đơn còn red ở iteration cuối thì bị từ chối. Khi có tầm nhìn tồn kho chung, lượng đã chào cho khách trước được trừ ra trước khi chào cho khách sau.
7. Các chỉ số:
   - `fill_rate`: tỷ lệ đơn được giao đủ. Đơn đã sửa vẫn tính là được phục vụ, giống kịch bản RE trong bài.
   - `avg_sl_D`: service level trung bình 5 ngày của khách.
   - `pct_cust_full`: tỷ lệ khách có tất cả đơn được giao đủ.
   - `avg_sl_strict`: chỉ tính các đơn vị đúng sản phẩm khách đặt, không tính hàng thay thế.

## 5. Lỗi của mô hình MILP như được in trong bài

Khi cài đúng công thức (1)-(7) như trong bài, nhóm thấy hai kẽ hở (kiểm chứng trong `tests/test_all.py::test_paper_model_flaw` và bảng `T1c`):

- Tᵢⱼ không có cận trên. Ràng buộc (3) chỉ yêu cầu Tᵢⱼ ≥ oᵢⱼ(1−Yᵢⱼ), nên mô hình có thể cấp nhiều hơn lượng khách đặt.
- Đơn không được phục vụ (Pᵢ = 1) vẫn có thể được gán hàng qua Xᵢₗ.

Trong cả hai trường hợp, lượng hàng thừa vẫn được cộng vào service level ở ràng buộc (6). Kết quả là L bị đẩy lên cao, có lúc vượt 100%, và hàng bị phân cho những đơn sẽ không giao, làm giảm số đơn được phục vụ.

Bản sửa (`variant="corrected"`) thêm hai ràng buộc Tᵢⱼ ≤ oᵢⱼ và Tᵢⱼ ≤ oᵢⱼ(1−Pᵢ), đồng thời thay |J| bằng |Jᵢ| ở (4). Có thể phần mềm thực tế của nhóm tác giả không gặp lỗi này; nhận xét ở đây chỉ áp dụng cho công thức được in.

## 6. Lỗi của CBC

CBC 2.10.3 (bản đi kèm PuLP 2.9) đôi khi báo "Optimal" nhưng nghiệm trả về lại không khả thi, ví dụ có biến âm hoặc vượt tồn kho của batch. Lỗi xuất hiện khi bước tiền xử lý của CBC thất bại, log ghi "Postprocessed model is infeasible - possible tolerance issue". Nhóm gặp tình huống này ở khoảng 5% số lần giải.

Vì vậy `model.py` kiểm tra lại mọi nghiệm. Nếu nghiệm vi phạm ràng buộc, mô hình được giải lại với `preprocess off`; nếu vẫn không khả thi thì sửa nghiệm bằng cách tham lam, chỉ giữ những đơn còn đủ hàng. Số lần phải giải lại được ghi ở cột `resolved` / `n_resolved`. Test tương ứng là `tests/test_all.py::test_cbc_preprocessing_bug`.

Với một DSS chạy thật, đặc biệt khi đơn green được ERP xác nhận tự động, không nên dùng nghiệm của solver khi chưa kiểm tra lại.

## 7. Kết quả chính

Bảng đầy đủ nằm trong `results/SUMMARY.md` và `results/tables/`, hình nằm trong `results/figures/`.

### TN1: tái lập testbed mục 5.1 (Bảng 1-3, Hình 3-5)

Sàng lọc gồm 8 yếu tố, mỗi yếu tố 2 mức, lặp 2 lần, tổng cộng 512 instance giải bằng MILP như in trong bài.

- Kết quả ANOVA khớp với Bảng 1 của bài báo ở 16/24 cặp yếu tố - chỉ số (`T1_anova_screening.csv`).
- Trên cùng 256 instance, MILP đã sửa khớp 17/24, còn MILP như in khớp 14/24 (`T1d`). Nói cách khác, bản sửa cho kết quả gần bài báo hơn chính công thức trong bài.
- Bài báo kết luận α không ảnh hưởng. Với MILP như in, α ảnh hưởng rõ đến fill rate (0,554 khi α = 0,1 và 0,666 khi α = 0,9; p < 0,001): khi α nhỏ, mô hình có lợi khi đẩy L lên nên dồn hàng vào các đơn không giao. Với MILP đã sửa thì α không có ảnh hưởng (0,663 và 0,673; p = 0,555), đúng như bài báo (`T1e`).
- Kiểm tra MILP như in: 60% instance có hàng được gán cho đơn không giao, 61% có cấp dư, 31% có L > 1. L trung bình mô hình báo là 0,90 trong khi service level tối thiểu thực tế chỉ 0,52 (`T1c`).

Lưới CPU gồm 162 tổ hợp. Với MILP đã sửa, thời gian giải tăng khi freshness coverage giảm (FFC 4,2 giây, HFC 7,8 giây, MFC 9,8 giây), cùng xu hướng với Hình 4 của bài báo; thời gian cũng tăng theo số đơn và cỡ đơn như Hình 3. Với MILP như in, thời gian gần như không đổi (khoảng 13 giây) và 39% instance chạm giới hạn 30 giây, so với 10% ở bản sửa (`T2c`).

### TN2: tái lập mục 5.2 (so sánh 4 kịch bản, Hình 6-8)

3 doanh nghiệp giả lập, mỗi doanh nghiệp 20 ngày, khoảng 98 đơn/ngày.

| Kịch bản (T = 5) | Tỷ lệ đơn giao đủ | Tỷ lệ đơn giao đủ, ngày khan hàng | SL trung bình 5 ngày |
|---|---|---|---|
| FCFS | 0,746 | 0,677 | 0,726 |
| RE-NOCM | 0,774 | 0,710 | 0,755 |
| GO | 0,794 | 0,736 | 0,769 |
| RE | 0,899 | 0,872 | 0,857 |

Thứ tự RE > GO > RE-NOCM > FCFS giống bài báo, và khoảng cách giữa các kịch bản lớn hơn vào những ngày khan hàng. Nhóm tính thêm khoảng tin cậy bootstrap 95% cho chênh lệch (bài báo không có): RE − FCFS = +0,150 [0,138; 0,163], RE − GO = +0,103 [0,093; 0,113], GO − RE-NOCM = +0,019 [0,016; 0,021]. Chênh lệch GO − RE-NOCM chính là phần mất đi do chia ngày thành nhiều iteration (`T5`).

Một điểm bài báo không đề cập: khi mục tiêu là tối đa số đơn, kênh FOOD (đơn lớn) bị thiệt. Fill rate của FOOD chỉ là 0,14 với FCFS, 0,09 với GO và 0,08 với RE-NOCM. RE nâng được lên khoảng 0,57-0,61, chủ yếu nhờ OCM.

### TN3: các mở rộng

| TN | Câu hỏi | Kết quả |
|---|---|---|
| 3a | Dùng MILP như in hay MILP đã sửa? | Trên testbed, fill rate tăng từ 0,684 lên 0,699, service level tối thiểu thực tế từ 0,526 lên 0,598, và bản sửa không lần nào cho service level thực tế thấp hơn. CPU giảm từ 6,3 xuống 2,0 giây. Trong mô phỏng, chênh lệch không quá 1 điểm phần trăm. |
| 3b | SATP có tác dụng không, và có nên ước lượng p̂ riêng cho từng sản phẩm? | SATP tăng fill rate của RE-NOCM từ 0,761 lên khoảng 0,788. Dùng p̂ theo sản phẩm hay thêm hệ số β chỉ thay đổi dưới 1 điểm, không vượt mức nhiễu. SATP làm giảm fill rate của FOOD. |
| 3c | Mỗi ngày nên chạy bao nhiêu iteration? | RE tăng từ 0,867 (T = 2) lên khoảng 0,90 và gần như không đổi từ T = 4 trở đi. RE-NOCM giảm nhẹ khi T tăng vì càng chia nhỏ càng gần với FCFS, phù hợp với nhận xét về đánh đổi ở mục 4 của bài. |
| 3d | Fill rate nhạy thế nào với α? | Với MILP đã sửa, gần như không đổi (GO từ 0,805 đến 0,807). Lý do là L luôn bằng 0 vì có khách có đơn không thể đáp ứng, nên thành phần max-min trong hàm mục tiêu không còn tác dụng. |
| 3e | OCM và tầm nhìn tồn kho chung đóng góp bao nhiêu? | Với p_accept = 0,7, fill rate là 0,902 khi có tầm nhìn chung và 0,850 khi không có, chênh 5,2 điểm do tránh được việc chào cùng một lượng hàng thay thế cho nhiều khách. Lợi ích của OCM tăng gần tuyến tính theo mức sẵn lòng sửa đơn của khách. |
| 3f | Nếu trọng số mục tiêu theo khối lượng thay vì số đơn? | Với GO, fill rate của FOOD tăng gấp đôi (0,099 lên 0,198), HORECA giảm (0,875 xuống 0,832), tổng thể giảm (0,805 xuống 0,775). Với RE, FOOD tăng từ 0,578 lên 0,624 còn tổng thể gần như giữ nguyên. Đây là đánh đổi giữa hai kênh. |

## 8. Khác biệt so với bài báo

- Hình 5 (fill rate theo stock coverage): ở mức FSC, fill rate của nhóm bằng 1,00 với mọi cỡ đơn, còn trong bài báo fill rate giảm khi cỡ đơn tăng. Khác biệt này đến từ cách sinh dữ liệu freshness (mục 4, ý 2): phần tồn kho bị hạn chế luôn khớp với yêu cầu của một dòng đơn cụ thể, nên khi đủ hàng thì đơn nào cũng được phục vụ.
- Thời gian CPU được đo trên máy 2 nhân với nhiều tiến trình chạy song song, giới hạn 30 giây (bài báo dùng 600 giây) và số lần lặp ít hơn (1-2 lần, bài báo 10 lần). Vì vậy chỉ so sánh xu hướng.
- Doanh nghiệp mô phỏng dựa trên giả định của nhóm, nên giá trị tuyệt đối của các chỉ số không phản ánh HEINEKEN Spain. Bản thân bài báo cũng đã biến đổi số liệu thật trước khi công bố.
