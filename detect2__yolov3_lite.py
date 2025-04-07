import cv2
import os
import glob
import numpy as np
from collections import defaultdict
import shutil
# ===== 你可以修改的參數區 =====
cfg_path = r"E:\0309v3lite\GP-YOLOv3-Lite.cfg"
weights_path = r"E:\0309v3lite\GP-YOLOv3-Lite_final.weights"
names_path = r"E:\0309v3lite\obj.names"
input_folder = r"E:\gpa_photo\0.5"
output_image_folder = r"E:\gpa_photo\0.5\output_images_v3"
output_label_folder = r"E:\gpa_photo\0.5\output_labels_v3"
confidence_threshold = 0.3
font_scale = 0.5
font_thickness = 1
# =================================

if os.path.exists(output_label_folder):
    shutil.rmtree(output_label_folder)
os.makedirs(output_label_folder)

if os.path.exists(output_image_folder):
    shutil.rmtree(output_image_folder)
os.makedirs(output_image_folder)

# 讀取 class 名稱
with open(names_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# 建立輸出資料夾
os.makedirs(output_image_folder, exist_ok=True)
os.makedirs(output_label_folder, exist_ok=True)

# 將 classes.txt 寫入標註資料夾
with open(os.path.join(output_label_folder, 'classes.txt'), 'w') as f:
    f.write('\n'.join(classes))

# 載入網路
net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

# 獲得輸出層名稱
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# 統計每類別的信心值
class_conf_stats = defaultdict(list)

# 取得所有圖片
image_paths = glob.glob(os.path.join(input_folder, '*.*'))

for img_path in image_paths:
    img = cv2.imread(img_path)
    height, width = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = int(np.argmax(scores))
            confidence = scores[class_id]
            if confidence > confidence_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, 0.4)

    # 輸出標註檔
    label_filename = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
    label_path = os.path.join(output_label_folder, label_filename)

    with open(label_path, 'w') as f:
        if len(indices) > 0:
            for i in indices:
                i = i[0] if isinstance(i, (list, tuple, np.ndarray)) else i
                x, y, w, h = boxes[i]
                cx = (x + w / 2) / width
                cy = (y + h / 2) / height
                nw = w / width
                nh = h / height
                f.write(f"{class_ids[i]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

                # 畫框和文字
                color = (0, 255, 0)
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                label = f"{classes[class_ids[i]]} {confidences[i]:.2f}"
                cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness)

                # 統計信心值
                class_conf_stats[class_ids[i]].append(confidences[i])

    out_img_path = os.path.join(output_image_folder, os.path.basename(img_path))
    cv2.imwrite(out_img_path, img)
    
print(f"\n✅ 設定偵測信心度下限：{confidence_threshold}")
print("📊 每類別的平均信心值：")

lines = [f"✅ 設定偵測信心度下限：{confidence_threshold}"]  # 加入第一行
for class_id, conf_list in class_conf_stats.items():
    avg_conf = sum(conf_list) / len(conf_list)
    line = f"{classes[class_id]}: {avg_conf:.3f} ({len(conf_list)} 個偵測)"
    print(f" - {line}")
    lines.append(line)

# 寫入檔案
output_path = os.path.join(output_label_folder, 'conf_average.txt')
with open(output_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print("\n✅ 偵測完成！標註圖片與標註檔已輸出。")
