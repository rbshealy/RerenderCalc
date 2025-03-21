
import tensorflow as tf
import pandas as pd
import numpy as np

#source code adapted from https://github.com/facebookresearch/projectaria_tools/blob/main/core/mps/EyeGazeReader.h#L40

ipd_meters = 0.063


def get_gaze_intersection_point(left_yaw_rads: float, right_yaw_rads: float, pitch_rads: float) -> np.ndarray:
    half_ipd = ipd_meters / 2.0
    intersection_x = half_ipd * (np.tan(left_yaw_rads) + np.tan(right_yaw_rads)) / \
                     (np.tan(right_yaw_rads) - np.tan(left_yaw_rads))
    intersection_z = ipd_meters / (np.tan(right_yaw_rads) - np.tan(left_yaw_rads))
    intersection_y = intersection_z * np.tan(pitch_rads)

    return np.array([intersection_x, intersection_y, intersection_z])

def get_gaze_intersection_point_tf(left_yaw_rads, right_yaw_rads, pitch_rads):
    """
    Computes the gaze intersection points for a batch of inputs using TensorFlow.

    Parameters:
        left_yaw_rads (Tensor): Left eye yaw angles in radians (shape: [N])
        right_yaw_rads (Tensor): Right eye yaw angles in radians (shape: [N])
        pitch_rads (Tensor): Pitch angles in radians (shape: [N])
        ipd_meters (Tensor): Interpupillary distances in meters (shape: [N])

    Returns:
        Tensor: Intersection points with shape [N, 3]
    """

    half_ipd = ipd_meters / 2.0

    tan_left = tf.math.tan(left_yaw_rads)
    tan_right = tf.math.tan(right_yaw_rads)
    tan_pitch = tf.math.tan(pitch_rads)

    denominator = tan_right - tan_left

    intersection_x = half_ipd * (tan_left + tan_right) / denominator
    intersection_z = ipd_meters / denominator
    intersection_y = intersection_z * tan_pitch

    return tf.stack([intersection_x, intersection_y, intersection_z], axis=1)

"""
N = 18000  # Number of rows
left_yaw_rads_np = np.random.uniform(-0.5, 0.5, size=N).astype(np.float32)
right_yaw_rads_np = np.random.uniform(-0.5, 0.5, size=N).astype(np.float32)
pitch_rads_np = np.random.uniform(-0.5, 0.5, size=N).astype(np.float32)
ipd_meters_np = np.full(N, 0.065, dtype=np.float32)  # Constant IPD (e.g., 65mm)

# Convert to TensorFlow tensors
left_yaw_rads_tf = tf.convert_to_tensor(left_yaw_rads_np)
right_yaw_rads_tf = tf.convert_to_tensor(right_yaw_rads_np)
pitch_rads_tf = tf.convert_to_tensor(pitch_rads_np)
ipd_meters_tf = tf.convert_to_tensor(ipd_meters_np)

# Compute intersection points
intersection_points = get_gaze_intersection_point_tf(left_yaw_rads_tf, right_yaw_rads_tf, pitch_rads_tf)

# Convert back to NumPy if needed
intersection_points_np = intersection_points.numpy()

print(intersection_points_np.shape)  # Should be (18000, 3)
"""

if __name__ == "__main__":
    df = pd.read_csv("./data/general_eye_gaze.csv",header=0)
    nparr = df[["left_yaw_rads_cpf", "right_yaw_rads_cpf", "pitch_rads_cpf"]].to_numpy()
    tsor = tf.convert_to_tensor(nparr, dtype = tf.float32)
    print(tf.shape(tsor))
    vecs = get_gaze_intersection_point_tf(tsor[:,0],tsor[:,1],tsor[:,2])
    print(vecs)


