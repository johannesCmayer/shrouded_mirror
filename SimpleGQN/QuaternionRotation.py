import numpy as np

def y_rotation(quaternion, rot):
    l =  list((np.sin(rot/2) * quaternion)[:3])
    l.append(np.cos(rot/2))
    return np.array(normalize_quarternion(l))


def wrap_around_quaternion_rotation(normal_quaternion_iterable):
    if normal_quaternion_iterable[3] > 0.99:
        normal_quaternion_iterable = np.array([0, -0.98, 0, 0.011])
    if normal_quaternion_iterable[3] < -0.99:
        normal_quaternion_iterable = np.array([0, 0.011, 0, 0.98])
    return normal_quaternion_iterable


def normalize_quarternion(quaternion_iterable):
    quat_mag = 0
    for e in quaternion_iterable:
        quat_mag += np.power(e, 2)
    quat_mag = np.sqrt(quat_mag)
    normalized_quat = []
    for e in quaternion_iterable:
        normalized_quat.append(e / quat_mag)
    # TODO extract the wraparound behaviour to somewhere else
    return wrap_around_quaternion_rotation(np.array(normalized_quat))


if __name__ == '__main__':
    quat = np.array([0,1,0,0])
    while(True):
        quat = y_rotation(quat, 20)
        print(quat)