import cv2
import os

def sample_frames(video_path, output_dir, freq):
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
    sample_frames("C:\Dev\RerenderCalc\eye_gaze.mp4","C:\Dev\RerenderCalc\output")
