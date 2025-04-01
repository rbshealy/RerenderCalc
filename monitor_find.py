import os
import cv2
import pandas as pd
import torch
from ultralytics import YOLO
import math

def monitor_find(path="C:/Dev/RerenderCalc/output", save=True, output_csv="monitor_positions.csv"):
    model = YOLO("yolov8x.pt")
    print("🔧 YOLOv8x | Left-prioritized screen selection (top 3 leftmost candidates)")

    output_data = []

    for image_name in sorted(os.listdir(path)):
        if not image_name.lower().endswith(('.jpg', '.jpeg', '.png')) or not image_name.startswith("frame_"):
            continue

        image_path = os.path.join(path, image_name)

        try:
            results = model(image_path, verbose=False)[0]
            boxes = results.boxes
            class_names = results.names

            candidates = []

            for box in boxes:
                cls = int(box.cls[0])
                label = class_names[cls].lower()
                confidence = box.conf[0].item()

                if confidence < 0.3:
                    continue

                if any(term in label for term in ["tv", "screen", "monitor"]):
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    area = (x2 - x1) * (y2 - y1)
                    tr_x, tr_y = x2, y1
                    candidates.append({
                        "tr": (tr_x, tr_y),
                        "x1": x1,
                        "area": area
                    })

            if not candidates:
                raise ValueError("No valid screen-like object found")

            # Sort candidates by leftmost x1, then pick largest area among top 3
            candidates = sorted(candidates, key=lambda c: c["x1"])
            top_left_candidates = candidates[:3]
            best = max(top_left_candidates, key=lambda c: c["area"])

            tr_x, tr_y = best["tr"]

        except Exception as e:
            print(f"⚠️ Failed to process {image_name}: {e}")
            tr_x, tr_y = -1, -1

        output_data.append({
            'filename': image_name,
            'top_right_x': tr_x,
            'top_right_y': tr_y
        })

    df = pd.DataFrame(output_data)
    df["frame"] = df["filename"].str.extract(r"frame_(\d+)", expand=False).astype(int)
    df = df.sort_values(by="frame")

    if save:
        df.to_csv(output_csv, index=False)
        print(f"✅ Done! Saved to '{output_csv}'")

    return df


def monitor_find_and_label(path="C:/Dev/RerenderCalc/output", save=True, output_csv="monitor_positions.csv"):
    model = YOLO("yolov8x.pt")
    print("🔧 YOLOv8x | Left-prioritized selection + annotated image output")

    edited_dir = os.path.join(path, "edited_images")
    os.makedirs(edited_dir, exist_ok=True)

    output_data = []

    for image_name in os.listdir(path):
        if not image_name.lower().endswith(('.jpg', '.jpeg', '.png')) or not image_name.startswith("frame_"):
            continue

        image_path = os.path.join(path, image_name)
        image = cv2.imread(image_path)

        try:
            results = model(image_path, verbose=False)[0]
            boxes = results.boxes
            class_names = results.names

            candidates = []

            for box in boxes:
                cls = int(box.cls[0])
                label = class_names[cls].lower()
                confidence = box.conf[0].item()

                if confidence < 0.3:
                    continue

                if any(term in label for term in ["tv", "screen", "monitor"]):
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    area = (x2 - x1) * (y2 - y1)
                    tr_x, tr_y = x2, y1
                    candidates.append({
                        "tr": (tr_x, tr_y),
                        "x1": x1,
                        "area": area,
                        "box": (x1, y1, x2, y2),
                        "label": label,
                        "confidence": confidence
                    })

            if not candidates:
                raise ValueError("No valid screen-like object found")

            # Sort leftmost first, then choose largest from top 3
            candidates = sorted(candidates, key=lambda c: c["x1"])
            top_left_candidates = candidates[:3]
            best = max(top_left_candidates, key=lambda c: c["area"])

            tr_x, tr_y = best["tr"]
            x1, y1, x2, y2 = best["box"]
            label_text = f"{best['label']} ({best['confidence']:.2f})"

            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Save annotated image
            save_path = os.path.join(edited_dir, image_name)
            cv2.imwrite(save_path, image)

        except Exception as e:
            print(f"⚠️ Failed to process {image_name}: {e}")
            tr_x, tr_y = -1, -1

        output_data.append({
            'filename': image_name,
            'top_right_x': tr_x,
            'top_right_y': tr_y
        })

    df = pd.DataFrame(output_data)
    df["frame"] = df["filename"].str.extract(r"frame_(\d+)", expand=False).astype(int)
    df = df.sort_values(by="frame")

    if save:
        df.to_csv(output_csv, index=False)
        print(f"✅ CSV saved to '{output_csv}'")
        print(f"🖼️ Annotated images saved to '{edited_dir}'")

    return df

def pix_dist(f1,f2):
    return math.sqrt((f1["top_right_x"] - f2["top_right_x"])**2 + (f1["top_right_y"] - f2["top_right_y"])**2)

def monitor_find_adv(path="C:/Dev/RerenderCalc/output", save=True, output_csv="monitor_positions.csv",delta=10):
    model = YOLO("yolov8x.pt")
    model.to('cuda')
    print("GPU available:", torch.cuda.is_available())  # Should be True
    print("Model device:", next(model.model.parameters()).device)
    print("🔧 YOLOv8x | Left-prioritized screen selection (top 3 leftmost candidates)")

    output_data = []
    prev = {}

    for image_name in sorted(os.listdir(path)):
        if not image_name.lower().endswith(('.jpg', '.jpeg', '.png')) or not image_name.startswith("frame_"):
            continue
        render = True

        print(image_name)

        image_path = os.path.join(path, image_name)

        try:
            results = model(image_path, verbose=False)[0]
            boxes = results.boxes
            class_names = results.names

            candidates = []

            for box in boxes:
                cls = int(box.cls[0])
                label = class_names[cls].lower()
                confidence = box.conf[0].item()

                if confidence < 0.3:
                    continue

                if any(term in label for term in ["tv", "screen", "monitor"]):
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    area = (x2 - x1) * (y2 - y1)
                    tr_x, tr_y = x2, y1
                    candidates.append({
                        "tr": (tr_x, tr_y),
                        "x1": x1,
                        "area": area
                    })

            if not candidates:
                raise ValueError("No valid screen-like object found")

            # Sort candidates by leftmost x1, then pick largest area among top 3
            candidates = sorted(candidates, key=lambda c: c["x1"])
            top_left_candidates = candidates[:3]
            best = max(top_left_candidates, key=lambda c: c["area"])

            tr_x, tr_y = best["tr"]

        except Exception as e:
            print(f"⚠️ Failed to process {image_name}: {e}")
            tr_x, tr_y, render = -1, -1, False

        if not prev:
            prev = {
            'filename': image_name,
            'top_right_x': tr_x,
            'top_right_y': tr_y,
            're-render': True
        }
            output_data.append(prev)
            continue

        curr = {
            'filename': image_name,
            'top_right_x': tr_x,
            'top_right_y': tr_y,
            're-render': None
        }

        if pix_dist(prev,curr) >= delta:
            curr['re-render'] = render
            prev = curr
        else:
            curr['re-render'] = False

        output_data.append(curr)


    df = pd.DataFrame(output_data)
    #df["frame"] = df["filename"].str.extract(r"frame_(\d+)", expand=False).astype(int)
    #df = df.sort_values(by="frame")

    if save:
        df.to_csv(output_csv, index=False)
        print(f"✅ Done! Saved to '{output_csv}'")

    return df
