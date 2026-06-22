---
name: code-analysis
description: Phân tích chi tiết một hoặc nhiều folder code do người dùng chỉ định — đọc từng file, xác định vai trò của các class và các function (chức năng, input/output, luồng gọi), rồi generate ra file markdown trình bày kết quả. Dùng skill này khi người dùng yêu cầu "phân tích folder/thư mục này", "phân tích class và function trong code", "đọc hiểu codebase rồi xuất file phân tích", "giải thích vai trò các class trong module X", hoặc muốn một báo cáo chi tiết về cách một phần codebase hoạt động — ngay cả khi họ không dùng đúng từ "skill" hay "analysis".
---

# Code Function Analysis

Phân tích chi tiết các folder code do người dùng chỉ định: vai trò của từng
class, từng function, và mối quan hệ giữa chúng — sau đó xuất kết quả ra một
file markdown. Chỉ đọc và báo cáo — **không sửa code nguồn**.

## Quy trình

### Bước 1 — Xác định phạm vi

Người dùng chỉ định một hoặc nhiều folder. Nếu họ chỉ nói tên project/repo
chung mà không rõ folder, hỏi lại folder cụ thể (vì mục tiêu của skill này
là phân tích sâu, không phù hợp để quét tràn cả repo lớn).

### Bước 2 — Khảo sát cấu trúc folder

Liệt kê toàn bộ file trong folder (và sub-folder nếu có) trước khi đọc chi
tiết, để có bức tranh tổng thể: bao nhiêu file, loại file gì (class-based,
util, config, test...). Bỏ qua file test/generated/build output trừ khi
được yêu cầu phân tích luôn.

### Bước 3 — Đọc và phân loại theo từng file

Với mỗi file trong phạm vi:
1. Đọc toàn bộ nội dung file.
2. Liệt kê các **class** (hoặc struct/interface có method đi kèm, tuỳ ngôn
   ngữ) định nghĩa trong file.
3. Liệt kê các **function độc lập** (không thuộc class nào) định nghĩa
   trong file.

### Bước 4 — Phân tích chi tiết từng class

Với mỗi class quan trọng (ưu tiên class được export/public trước, class nội
bộ/private có thể phân tích ngắn hơn), điền theo template:

```markdown
#### Class: `TenClass` — `file.ext:line`
- **Vai trò / trách nhiệm:** mô tả class này đại diện cho cái gì, giải quyết
  vấn đề gì trong hệ thống.
- **Kế thừa / implement:** class cha, interface implement (nếu có).
- **Thuộc tính chính:** các field/property quan trọng và ý nghĩa của chúng.
- **Method quan trọng:**
  - `methodA(input) -> output`: chức năng, side-effect nếu có
  - `methodB(...)`: ...
- **Phụ thuộc vào:** class/module khác mà nó dùng (constructor injection,
  import, gọi trực tiếp...).
- **Được sử dụng ở đâu:** nơi class này được khởi tạo/sử dụng (trong cùng
  folder hoặc nơi khác nếu xác định được qua grep).
```

### Bước 5 — Phân tích chi tiết từng function độc lập

Với mỗi function không thuộc class, điền theo template:

```markdown
#### Function: `tenHam(...)` — `file.ext:line`
- **Chức năng:** mô tả ngắn hàm này làm gì.
- **Input / Output:** kiểu dữ liệu và ý nghĩa từng tham số, giá trị trả về.
- **Side-effect:** I/O, gọi network/DB, mutate state ngoài (nếu có).
- **Gọi tới:** các function/class khác mà nó gọi.
- **Được gọi từ:** nơi function này được dùng (nếu xác định được).
```


### Bước 6 — Tổng kết vai trò của folder

Viết phần tổng quan: folder này đóng vai trò gì trong kiến trúc tổng thể của
project (ví dụ: "đây là layer service xử lý nghiệp vụ đơn hàng, nhận request
từ controller, gọi xuống repository để truy vấn DB"). Nếu xác định được nơi
folder này được import/sử dụng từ bên ngoài, ghi chú lại để làm rõ vai trò.

### Bước 7 — Generate file markdown kết quả

Viết toàn bộ kết quả ra file `.md` theo cấu trúc:

```markdown
# Phân tích folder: <tên-folder>

## 1. Tổng quan
- Vai trò của folder trong project
- Danh sách file đã phân tích

## 2. Phân tích theo file

### `file1.ext`
#### Class: `TenClassA`
...
#### Function: `tenHamA`
...

### `file2.ext`
...

## 3. Mối quan hệ giữa các thành phần
(danh sách hoặc sơ đồ Mermaid)

## 4. Nhận xét / Đề xuất (nếu có)
- Điểm cần lưu ý: code smell, class quá nhiều trách nhiệm, thiếu xử lý lỗi...
- Đề xuất cải thiện — chỉ nêu gợi ý, không tự áp dụng vào code.
```

## Quy tắc đặt tên & vị trí file output

- Một folder được phân tích → một file kết quả, đặt tên
  `<tên-folder>-ANALYSIS.md`, lưu ngay trong folder đó (hoặc nơi người dùng
  chỉ định).
- Phân tích nhiều folder trong một lần yêu cầu → mặc định tạo **một file
  riêng cho mỗi folder** để dễ đọc; nếu người dùng muốn gộp chung một file,
  làm theo yêu cầu đó.
- **Không tự ý ghi đè `README.md`** nếu file đó đã có nội dung khác (README
  thường là tài liệu giới thiệu project, không phải báo cáo phân tích). Nếu
  người dùng nói rõ muốn ghi vào `README.md`, xác nhận trước khi ghi đè, hoặc
  đề xuất thêm phần phân tích vào cuối README hiện tại.
- Nếu file kết quả đã tồn tại từ lần phân tích trước, hỏi người dùng muốn ghi
  đè hay tạo bản mới.

## Nguyên tắc

- Đây là yêu cầu phân tích **chi tiết** — ưu tiên độ đầy đủ hơn là rút gọn,
  nhưng vẫn viết súc tích trong từng mục, tránh diễn giải lại code dòng-by-
  dòng.
- Với folder có rất nhiều file, đọc và viết kết quả tuần tự theo từng file
  (đọc file → phân tích → viết phần tương ứng vào file `.md`) thay vì giữ
  toàn bộ trong đầu rồi viết một lần, để tránh quá tải context.
- Dependency bên ngoài project (thư viện npm/pip, external API) chỉ ghi nhận
  là "external dependency", không cần đọc source của nó.
- Không tự sửa code nguồn — chỉ đọc và báo cáo.
- Sau khi tạo file, báo ngắn gọn cho người dùng: đã phân tích bao nhiêu file,
  bao nhiêu class, bao nhiêu function, và file kết quả nằm ở đâu.
