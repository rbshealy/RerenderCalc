
import tensorflow as tf
import pandas as pd
import numpy as np

#source code adapted from https://github.com/facebookresearch/projectaria_tools/blob/main/core/mps/EyeGazeReader.h#L40

ipd_meters = 0.063

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

if __name__ == "__main__":
    """
    df = pd.read_csv("./data/general_eye_gaze.csv",header=0)
    nparr = df[["left_yaw_rads_cpf", "right_yaw_rads_cpf", "pitch_rads_cpf"]].to_numpy()
    tsor = tf.convert_to_tensor(nparr, dtype = tf.float32)
    print(tf.shape(tsor))
    vecs = get_gaze_intersection_point_tf(tsor[:,0],tsor[:,1],tsor[:,2])
    print(vecs)


