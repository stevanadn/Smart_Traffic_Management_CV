from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import numpy as np

# --- 1. MUAT SEMUA MODEL & GAMBAR ---

# Muat Model A (Detektor Kendaraan & Orang)
vehicle_model = YOLO('VEHICLE-DETECTION-MODEL.pt')

# Muat Model B (Model deteksi helmet)
helmet_model = YOLO('28SEPT-HELMET-TO27-BEST-083-057.pt')

# Path gambar yang akan diuji
image_path = 'cobaan.png'

# Muat gambar dan konversi ke RGB
frame = cv2.imread(image_path)
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


# --- 2. FUNGSI BANTUAN UNTUK MENGHITUNG IoU ---
def calculate_iou(boxA, boxB):
    xA = np.maximum(boxA[0], boxB[0])
    yA = np.maximum(boxA[1], boxB[1])
    xB = np.minimum(boxA[2], boxB[2])
    yB = np.minimum(boxA[3], boxB[3])

    interArea = np.maximum(0, xB - xA) * np.maximum(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


# --- 3. JALANKAN MODEL A (DETEKTOR KENDARAAN) ---
vehicle_results = vehicle_model(frame_rgb)

# Simpan deteksi motor dan orang untuk diproses
persons = []
motorcycles = []
detected_other_vehicles = []
other_vehicles = ['car', 'bus', 'truck']

for r in vehicle_results:
    for box in r.boxes:
        class_id = int(box.cls[0])
        class_name = vehicle_model.names[class_id]

        if class_name == 'person':
            persons.append(box.xyxy[0].cpu().numpy())


        elif class_name == 'motorcycle':
            motorcycles.append(box.xyxy[0].cpu().numpy())
            # Langsung gambar kotak untuk motor
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (255, 165, 0), 2)  # Warna oranye
            cv2.putText(frame_rgb, 'Motorcycle', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 2)
        elif class_name in other_vehicles:
            detected_other_vehicles.append(box.xyxy[0].cpu().numpy())
            # Langsung gambar kotak untuk kendaraan lain
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Warna biru
            cv2.putText(frame_rgb, 'Vehicle', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# --- 4. LOGIKA PENGHUBUNG & INFERENSI MODEL B (SPESIALIS HELM) ---
IOU_THRESHOLD = 0.1  # Ambang batas IoU untuk asosiasi

for person_box in persons:
    is_rider = False
    for motor_box in motorcycles:
        iou = calculate_iou(person_box, motor_box)
        if iou > IOU_THRESHOLD:
            is_rider = True
            break  # Hentikan pencarian jika sudah terasosiasi dengan satu motor

    px1, py1, px2, py2 = map(int, person_box)

    if is_rider:
        # Potong area kepala (40% bagian atas dari kotak orang)
        head_y_end = py1 + int((py2 - py1) * 0.35)
        head_region = frame_rgb[py1:head_y_end, px1:px2]

        if head_region.shape[0] > 0 and head_region.shape[1] > 0:
            # Jalankan Model B pada potongan kepala
            helmet_results = helmet_model(head_region)

            # Ambil hasil deteksi helm
            for hr in helmet_results:
                if len(hr.boxes) > 0:
                    helmet_class_id = int(hr.boxes[0].cls[0])
                    helmet_class_name = helmet_model.names[helmet_class_id]

                    # Tentukan warna label berdasarkan hasil
                    label = helmet_class_name
                    if label == 'Helmet':
                        color = (0, 255, 0)  # Hijau untuk 'helmet'
                    if label == 'No_helmet':
                        color = (200, 0, 0)  # Merah untuk 'no_helmet'

                    # Gambar kotak di sekitar orang dan label hasil deteksi helm
                    cv2.rectangle(frame_rgb, (px1, py1), (px2, py2), color, 2)
                    cv2.putText(frame_rgb, label, (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    break  # Hanya ambil deteksi pertama

total_vehicles = len(motorcycles) + len(detected_other_vehicles)

# Logika : 2 detik per kendaraan, dengan batas minimum dan maksimum
base_green_time = 10  # Detik minimum lampu hijau
green_time_per_vehicle = 2  # Tambahan detik per kendaraan
max_green_time = 45  # Detik maksimum lampu hijau

green_duration = base_green_time + (total_vehicles * green_time_per_vehicle)
# Pastikan durasi tidak melebihi batas maksimum
green_duration = min(green_duration, max_green_time)

# Durasi lampu merah bisa dibuat tetap atau disesuaikan
red_duration = 30  # Durasi tetap untuk lampu merah

# Tampilkan informasi di pojok kanan atas
h, w, _ = frame_rgb.shape
info_position_y = 40
cv2.putText(frame_rgb, f'Jumlah Kendaraan: {total_vehicles}', (w - 450, info_position_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(frame_rgb, f'Durasi Hijau: {green_duration} detik', (w - 450, info_position_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
cv2.putText(frame_rgb, f'Durasi Merah: {red_duration} detik', (w - 450, info_position_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 0, 0), 2, cv2.LINE_AA)

plt.figure(figsize=(16, 10))
plt.imshow(frame_rgb)
plt.title('TRAFFIC VIOLATION DETECTION ')
plt.axis('off')
plt.show()