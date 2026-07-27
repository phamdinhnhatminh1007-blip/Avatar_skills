# Avatar_skills — Hand Magic ✋🔥⚡🌪️🧊🪨

Điều khiển **phép thuật tứ đại nguyên tố** (lấy cảm hứng từ *Avatar*) bằng
**cử chỉ tay** trước webcam, theo thời gian thực. Giơ đúng thế tay là tung
"chiêu" tương ứng với hiệu ứng 3D — phun lửa, sét giáng từ trời, khiên đá,
lốc xoáy, tường băng — ngay trên hình ảnh camera của bạn.

Dự án ghép 3 mảng: **thị giác máy tính** (MediaPipe lấy điểm mốc bàn tay),
**học máy** (scikit-learn phân loại cử chỉ), và **đồ họa 3D** (Ursina/Panda3D
dựng hiệu ứng, shader, particle, bloom). Hỗ trợ **cả hai tay** cùng lúc.

## Bộ chiêu

| Cử chỉ | Nguyên tố | Hiệu ứng |
|--------|-----------|----------|
| ✋ Xòe bàn tay | Hỏa 🔥 | Phun luồng lửa nhọn, nóng và hỗn loạn về phía màn hình |
| ✊ Nắm đấm | Thổ 🪨 | Dựng **khiên đá** sần sùi, giữ mãi khi còn nắm đấm, sụp đổ khi thả |
| ☝️ Ngón trỏ | Sét ⚡ | **Sét giáng từ trời** xuống đất, để lại vệt sáng dư âm |
| 👍 Ngón cái | Khí 🌪️ | **Lốc xoáy** bao quanh tay, giữ mãi khi còn giơ, tan khi thả |
| 🤟 Thế Spider-Man | Thủy 🧊 | **Tường băng nhọn** đâm lên rồi bị kéo chìm xuống đất |

Điều khiển khác: phím `b` bật/tắt hiệu ứng glow (bloom) · `Esc` để thoát.

## Luồng hoạt động

```
Webcam ─► MediaPipe (21 landmarks) ─► ML model (nhận diện chiêu) ─► Hiệu ứng 3D (Ursina)
```

Mỗi khung hình: đọc webcam → lấy 21 điểm mốc bàn tay → chuẩn hóa thành vector →
model đoán ra chiêu → `main.py` gọi đúng module hiệu ứng để tung chiêu đó.

## Cài đặt

Yêu cầu Python 3.10+ và webcam. Nên dùng môi trường ảo:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Chạy

```bash
python main.py          # app chính (nhận diện + hiệu ứng)
```

File phụ trợ:

```bash
python gesture_demo.py  # chỉ nhận diện + in tên chiêu (không hiệu ứng), để test model
python collect_data.py  # thu dữ liệu cử chỉ mới
python train_model.py   # train lại model
```

## Cấu trúc file

Kiến trúc tách bạch: `main.py` lo phần "khung", **mỗi nguyên tố là một module
hiệu ứng riêng**, ghép lại qua một bảng `SKILLS` duy nhất.

| File | Vai trò |
|------|---------|
| `main.py` | **File chính.** Webcam, MediaPipe, ML nhận diện chiêu, dựng scene 3D, và **điều phối** chiêu → hiệu ứng qua bảng `SKILLS`. |
| `Fireball3D.py` | Hiệu ứng **Hỏa** — luồng phun lửa, khói, tàn lửa và shader glow. Cũng chứa `bloom_shader` dùng chung. |
| `earth.py` | Hiệu ứng **Thổ** — khiên đá (dựng lên, giữ, sụp đổ), texture đá, bụi. |
| `lightning.py` | Hiệu ứng **Sét** — tia sét zigzag chớp giật, vệt sáng dư âm. |
| `air.py` | Hiệu ứng **Khí** — lốc xoáy hình phễu quanh tay, tiếng gió loop. |
| `ice.py` | Hiệu ứng **Thủy** — cụm tinh thể sắc mọc hỗn loạn theo nhiều lớp chiều sâu. |
| `gesture_utils.py` | Dùng chung: danh sách `GESTURES` và `normalize_landmarks` (chuẩn hóa 21 điểm → vector 42 chiều). |
| `collect_data.py` | Thu dữ liệu cử chỉ, ghi vào `gestures.csv`. |
| `train_model.py` | Train `MLPClassifier` từ `gestures.csv` → `gesture_model.joblib`. |
| `gesture_demo.py` | Demo nhận diện thuần (không hiệu ứng), tiện test độ chính xác model. |
| `hand_landmarker.task` | Model MediaPipe phát hiện điểm mốc tay (bắt buộc). |
| `gestures.csv` / `gesture_model.joblib` | Dữ liệu và model đã train (file sinh ra). |
| `sounds/` | Âm thanh chiêu (`fireball.wav`, `thunder.wav`, `wind.wav`, `earth.wav`, `ice.wav`) — WAV PCM hoặc OGG. |

## Bảng chiêu (`SKILLS`) trong `main.py`

Toàn bộ ánh xạ cử chỉ → hiệu ứng gói gọn ở một chỗ:

```python
SKILLS = {
    0: Fireball3D.cast,    # Xòe bàn tay  -> luồng phun lửa (Hỏa)
    1: earth.cast,         # Nắm đấm       -> khiên đá (Thổ)
    3: lightning.cast,     # Ngón trỏ      -> sét (Sét)
    4: air.cast,           # Ngón cái      -> lốc xoáy (Khí)
    5: ice.cast,           # Spider-Man -> tường băng (Thủy)
    # 2: Ngón giữa         -> chưa gán
}
```

Vòng lặp chính chỉ việc: đoán chiêu → `SKILLS.get(gesture)` → gọi. Mỗi module
tự lo hiệu ứng, cooldown và âm thanh của riêng nó.

---

# 👉 Hướng dẫn thêm chiêu mới

Model đã nhận diện sẵn **6 cử chỉ** (index 0–5, xem `GESTURES` trong
`gesture_utils.py`). Gắn hiệu ứng cho các cử chỉ này **KHÔNG cần train lại** —
chỉ viết code.

**Quy trình:**

1. Viết một module hiệu ứng mới (xem `lightning.py` / `ice.py` làm mẫu). Mỗi
   hiệu ứng là một class kế thừa `Entity` có `update()` chạy mỗi frame rồi
   `destroy(self)`. Module cần một hàm:

   ```python
   def cast(lm, hand_to_world, hand_id=""):
       pos = hand_to_world(lm[9].x, lm[9].y)   # vị trí lòng bàn tay trong 3D
       ...                                     # tự lo cooldown + spawn hiệu ứng
   ```

2. Trong `main.py`: `import <module>` rồi thêm **một dòng** vào `SKILLS`:

   ```python
   SKILLS = { ..., 2: <module>.cast }          # ví dụ gán cho ngón giữa
   ```

Xong. Không cần đụng vào vòng lặp chính. Hai tay tự động hoạt động (cooldown
và trạng thái đã theo `hand_id`).

**Mẹo hiệu ứng:**

- Trục **z âm** = bay về phía màn hình (càng gần camera càng to nhờ phối cảnh).
  `SPAWN_Z = 16` là mặt phẳng hiệu ứng xuất hiện.
- Muốn hiệu ứng **phát sáng**: cho màu thật sáng (gần trắng) — `bloom_shader`
  chỉ bắt sáng phần vượt ngưỡng. Nền webcam đã bị giảm sáng nên không bị glow.
- Muốn **chấn động màn hình** (cảm giác lực): gọi `camera.shake(duration, magnitude)`.
- **Luôn `destroy(self)`** khi hiệu ứng hết đời, kẻo entity phình lên gây tụt FPS.

## Khi nào PHẢI train lại model

Chỉ khi **thêm một cử chỉ hoàn toàn mới** (chưa có trong `GESTURES`), hoặc model
hay nhận nhầm và bạn muốn thu thêm mẫu:

1. Sửa danh sách `GESTURES` trong `gesture_utils.py` (tối đa 10 chiêu, ứng phím 0–9).
2. `python collect_data.py` — giơ cử chỉ, **giữ phím số** tương ứng để ghi mẫu
   (~200–300 mẫu/chiêu, đổi góc/khoảng cách/tay trái–phải).
3. `python train_model.py` — train lại, xuất `gesture_model.joblib`.
4. Chạy lại `main.py`.

> Lưu ý: cùng một model dùng được cho **cả hai tay** — code thử cả bản gốc và
> bản lật gương của bàn tay, nên không cần train riêng tay trái/phải.

## Âm thanh

Mỗi chiêu tự phát âm thanh nếu có file tương ứng trong `sounds/`
(`fireball.wav`, `thunder.wav`, `earth.wav`, `wind.wav`, `ice.wav`). Thiếu file
thì bỏ qua, không lỗi. **Phải là WAV PCM hoặc OGG thật** — đổi đuôi file MP3
thành `.wav` sẽ không phát được (Panda3D báo "not a valid RIFF file").

## Tham số tinh chỉnh nhanh

| Tham số | File | Ý nghĩa |
|---------|------|---------|
| `CONF_THRESHOLD` (0.8) | `main.py` | Dưới ngưỡng này coi như không rõ chiêu. |
| `SMOOTH_FRAMES` (7) | `main.py` | Số frame để bình chọn (làm mượt, chống nhấp nháy). |
| `BG_BRIGHTNESS` (0.85) | `main.py` | Độ sáng webcam (phải < ngưỡng bloom để nền không bị glow). |
| `FlameStream.EMIT_INTERVAL` | `Fireball3D.py` | Mật độ lưỡi lửa trong luồng phun. |
| `timeout` | `earth.py` / `air.py` | Bỏ tay bao lâu thì khiên/lốc tan (chiêu duy trì). |
| `count` / `spacing` | `ice.py` | Số cột băng và độ rộng tường băng. |
| `threshold / intensity / blur_size` | `main.py` | Thông số bloom (glow). |

## Ghi chú kỹ thuật

- Webcam mở bất đồng bộ bằng **DirectShow** (`cv2.CAP_DSHOW`) trong lúc model và
  scene đang tải; frame nền 540p và frame nhận diện 512p để giảm thời gian chờ.
- Thoát bằng `os._exit(0)` để tránh lỗi destructor của MediaPipe trên Python 3.13.
- `gestures.csv` và `gesture_model.joblib` là file sinh ra; có thể không commit
  `.joblib` (người khác clone về tự `train_model.py`).
- `.vscode/` chứa đường dẫn `.venv` theo máy cá nhân — cân nhắc thêm vào `.gitignore`.
