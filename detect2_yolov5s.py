import torch
import os
import cv2
import shutil
import pathlib
from pathlib import Path

# ===== 路徑相容設定（適用於 Windows 系統）=====
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

x = "1.0"  # ← 這裡替換成你要的資料夾名稱

# 自動套用變數 x 到路徑中
base_path = r"E:\gpa_photo"
image_folder = fr"{base_path}\{x}\images"
label_folder = fr"{base_path}\{x}\labels"

target_image_folder = fr"{base_path}\{x}\images3"
target_label_folder = fr"{base_path}\{x}\labels3"

# ===== 使用者參數 =====
image_folder = fr"{base_path}\{x}"               # 圖片資料夾
output_folder = fr"{base_path}\{x}\output_labels_v5"       # 標註輸出資料夾
model_path = 'runs\\train\\exp7\\weights\\best.pt' # 模型權重
save_confidence = False                           # 是否儲存信心度 True  False  
confidence_threshold = 0.3                         # 信心度閾值

# ===== 繪圖相關設定 =====
draw_boxes = True                           # 是否繪製標註框
font_scale = 0.6                            # 字體大小（可自行調整）
box_color = (0, 255, 0)                     # 框的顏色（綠色）
text_color = (0, 0, 255)                    # 文字顏色（紅色）
drawn_image_output = fr"{base_path}\{x}\output_images_v5"      # 繪製圖儲存資料夾

# 每個類別保留的最高 N 筆偵測
max_detections_per_class = {
    0: 2,
    1: 2,
    2: 1,
    3: 1,
    4: 1,
    5: 1,
    6: 1
}

# ===== 初始化信心度統計 =====
class_confidences = {}
class_counts = {}

# ===== 載入 YOLOv5 模型 =====
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=False)
model.conf = confidence_threshold

# ===== 清空並重建輸出資料夾 =====
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
os.makedirs(output_folder)
if os.path.exists(drawn_image_output):
    shutil.rmtree(drawn_image_output)
os.makedirs(drawn_image_output)

# ===== 輸出 classes.txt，只含類別 ID =====
classes_txt_path = os.path.join(output_folder, "classes.txt")
with open(classes_txt_path, 'w') as f:
    for cls_id in sorted(max_detections_per_class.keys()):
        f.write(f"{cls_id}\n")

# ===== 開始處理每張圖片 =====
for filename in os.listdir(image_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(image_folder, filename)
        img = cv2.imread(image_path)
        h_img, w_img = img.shape[:2]

        results = model(image_path)
        detections = results.xywh[0].cpu().numpy()  # [x_center, y_center, width, height, conf, class]

        # 分類別儲存偵測框
        class_detections = {i: [] for i in range(7)}
        for det in detections:
            x, y, w, h, conf, cls = det
            cls = int(cls)
            if cls in class_detections and conf >= confidence_threshold:
                class_detections[cls].append((conf, x, y, w, h))

        # 處理各類別的 top N 偵測
        output_lines = []
        for cls, dets in class_detections.items():
            dets.sort(reverse=True)  # 按信心度高到低排序
            top_dets = dets[:max_detections_per_class.get(cls, 0)]
            for conf, x, y, w, h in top_dets:
                # 正規化座標
                x_norm = x / w_img
                y_norm = y / h_img
                w_norm = w / w_img
                h_norm = h / h_img

                if save_confidence:
                    line = f"{cls} {x_norm:.6f} {y_norm:.6f} {w_norm:.6f} {h_norm:.6f} {conf:.6f}"
                else:
                    line = f"{cls} {x_norm:.6f} {y_norm:.6f} {w_norm:.6f} {h_norm:.6f}"
                output_lines.append(line)

                # 計算信心度統計
                if cls not in class_confidences:
                    class_confidences[cls] = 0
                    class_counts[cls] = 0
                class_confidences[cls] += conf
                class_counts[cls] += 1

        # 儲存 YOLO 格式的標註 .txt
        txt_filename = os.path.splitext(filename)[0] + '.txt'
        txt_path = os.path.join(output_folder, txt_filename)
        with open(txt_path, 'w') as f:
            f.write('\n'.join(output_lines))
    if draw_boxes:
        os.makedirs(drawn_image_output, exist_ok=True)
        drawn_img = img.copy()
        for cls, dets in class_detections.items():
            dets.sort(reverse=True)
            top_dets = dets[:max_detections_per_class.get(cls, 0)]
            for conf, x, y, w, h in top_dets:
                # 邊界框座標（左上與右下角）
                x1 = int((x - w / 2))
                y1 = int((y - h / 2))
                x2 = int((x + w / 2))
                y2 = int((y + h / 2))

                # 畫邊界框
                cv2.rectangle(drawn_img, (x1, y1), (x2, y2), box_color, 2)

                # 標籤文字
                label = f"{cls} {conf:.2f}"
                (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
                cv2.rectangle(drawn_img, (x1, y1 - text_h - 4), (x1 + text_w, y1), box_color, -1)
                cv2.putText(drawn_img, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1)

        # 儲存畫框圖像
        output_img_path = os.path.join(drawn_image_output, filename)
        cv2.imwrite(output_img_path, drawn_img)

# ===== 顯示每個類別的平均信心度 =====
print(f"\n✅ 設定偵測信心度下限：{confidence_threshold}")
print("📊 每個類別的平均信心度：")

lines = [f"✅ 設定偵測信心度下限：{confidence_threshold}"]

for class_id in sorted(class_confidences.keys()):
    total_conf = class_confidences[class_id]
    count = class_counts[class_id]
    avg_conf = total_conf / count if count > 0 else 0
    line = f"類別 {class_id}: {avg_conf:.3f}（{count} 個偵測）"
    print(f" - {line}")
    lines.append(line)

# 寫入 conf_average.txt
output_path = os.path.join(output_folder, 'conf_average.txt')
with open(output_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print("\n✅ 偵測與統計完成，標註已儲存至：", output_folder)
