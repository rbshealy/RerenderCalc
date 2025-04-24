import face_alignment
from skimage import io
import pandas as pd
import os
import torch
import math

def pix_dist(f1,f2):
    return math.sqrt((f1["nose_x"] - f2["nose_x"])**2 + (f1["nose_y"] - f2["nose_y"])**2)


def nose_find(path ="C:\Dev\RerenderCalc\output",save = True):
    # Auto-select GPU if available, fallback to CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 Using device: {device}")

    # Initialize the face alignment model
    fa = face_alignment.FaceAlignment('2D', device=device)

    # Folder containing your images
    image_folder = path  # <-- Replace with actual path
    output_data = []

    for image_name in os.listdir(image_folder):
        if not image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        image_path = os.path.join(image_folder, image_name)
        try:
            input_image = io.imread(image_path)
            landmarks = fa.get_landmarks(input_image)
        except Exception as e:
            print(f"⚠️ Failed to process {image_name}: {e}")
            landmarks = None

        if landmarks:
            nose_tip = landmarks[0][30]  # Landmark 30 is nose tip
            output_data.append({
                'filename': image_name,
                'nose_x': float(nose_tip[0]),
                'nose_y': float(nose_tip[1])
            })
        else:
            print(f"🚫 No face detected in {image_name}")
            output_data.append({
                'filename': image_name,
                'nose_x': -1,
                'nose_y': -1
            })

    # Save results to CSV
    df = pd.DataFrame(output_data)
    if save:
        df.to_csv("nose_coordinates_face_alignment.csv", index=False)
        print("✅ Done! Saved to 'nose_coordinates_face_alignment.csv'")

    return df

def nose_find_adv(path ="C:\Dev\RerenderCalc\output",save = True,output_csv="nose_positions.csv", thres = 10):
    # Auto-select GPU if available, fallback to CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 Using device: {device}")

    # Initialize the face alignment model
    fa = face_alignment.FaceAlignment('2D', device=device)

    # Folder containing your images
    image_folder = path  # <-- Replace with actual path
    output_data = []
    last = None

    for image_name in os.listdir(image_folder):
        if not image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        image_path = os.path.join(image_folder, image_name)
        try:
            input_image = io.imread(image_path)
            landmarks = fa.get_landmarks(input_image)
        except Exception as e:
            print(f"⚠️ Failed to process {image_name}: {e}")
            landmarks = None

        if landmarks and landmarks[0][30][0] >= 0 and landmarks[0][30][1] >= 0:
            nose_tip = landmarks[0][30]  # Landmark 30 is nose tip
            if last is None:
                last = {
                'filename': image_name,
                'nose_x': float(nose_tip[0]),
                'nose_y': float(nose_tip[1]),
                're-render': True,
            }
                output_data.append(last)
            else:
                thisdata = {
                    'filename': image_name,
                    'nose_x': float(nose_tip[0]),
                    'nose_y': float(nose_tip[1]),
                }
                thisdata['re-render'] = pix_dist(last, thisdata) >= thres
                output_data.append(thisdata)
                if thisdata['re-render']:
                    last = thisdata
        else:
            print(f"🚫 No face detected in {image_name}")
            output_data.append({
                'filename': image_name,
                'nose_x': -1,
                'nose_y': -1,
                're-render': False
            })

    # Save results to CSV
    df = pd.DataFrame(output_data)
    if save:
        df.to_csv(output_csv, index=False)
        print(f"✅ Done! Saved to {path}")

    return df

