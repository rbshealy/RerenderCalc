import cv2
import os

from monitor_find import monitor_find, monitor_find_and_label, monitor_find_adv
from nose_find import nose_find

def sample_frames(video_path, output_dir, freq):
    """Converts a video to frames and saves them in the output directory."""

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening video file")
        return

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    frame_count = 0
    count = 0
    while frame_count < int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)

        # Read a frame from the video
        ret, frame = cap.read()

        if not ret:
            break

        # Save the frame as an image
        frame_path = os.path.join(output_dir, f"frame_{frame_count:04d}.png")
        cv2.imwrite(frame_path, frame)

        frame_count += freq
        count += 1

    cap.release()
    print(f"Successfully extracted {count} frames from {video_path} to {output_dir}")


def sample_frames_2(video_path, output_dir, freq):
    """Converts a video to frames and saves them in the output directory."""

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening video file")
        return

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    frame_count = freq
    count = 0
    while frame_count < int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)

        # Read a frame from the video
        ret, frame = cap.read()
        ret, frame2 = cap.read()

        if not ret:
            break

        # Save the frame as an image
        frame_path = os.path.join(output_dir, f"frame_{frame_count:04d}.png")
        cv2.imwrite(frame_path, frame)
        frame_path = os.path.join(output_dir, f"frame_{frame_count+1:04d}.png")
        cv2.imwrite(frame_path, frame2)

        frame_count += freq
        count += 1

    cap.release()
    print(f"Successfully extracted {count * 2} frames from {video_path} to {output_dir}")


def clear_dir(dir):
    for filename in os.listdir(dir):
        file_path = os.path.join(dir, filename)
        if filename.endswith('.png') and os.path.isfile(file_path):
            os.remove(file_path)
            #print(f"Deleted: {file_path}")

if __name__ == "__main__":
    #sample_frames_2("C:\Dev\RerenderCalc\eye_gaze.mp4","C:\Dev\RerenderCalc\output",125)
    #nose_find()

    #sample_frames("C:\Dev\RerenderCalc\eye_gaze_9279.mp4","C:\Dev\RerenderCalc\output",1)
    monitor_find_adv(output_csv="monitor_positions_10.csv",delta=10)
    monitor_find_adv(output_csv="monitor_positions_20.csv",delta=20)
    monitor_find_adv(output_csv="monitor_positions_30.csv",delta=30)
