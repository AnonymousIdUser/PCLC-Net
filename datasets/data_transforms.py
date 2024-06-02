import numpy as np
import torch
import transforms3d
from scipy.spatial.transform import Rotation as R
from sklearn import cluster

def generate_random_Rt() -> (np.ndarray,  np.ndarray,  np.ndarray):
    rot = R.random()
    displacement = np.random.randn(3) # the x/y/z displacement from original point
    displacement[0:2] *= 20
    displacement[2] *= 5
    
    return rot.as_matrix(), rot.as_quat(), displacement

class Compose(object):
    def __init__(self, transforms):
        self.transformers = []
        for tr in transforms:
            transformer = eval(tr['callback'])
            parameters = tr['parameters'] if 'parameters' in tr else None
            self.transformers.append({
                'callback': transformer(parameters),
                'objects': tr['objects']
            })  # yapf: disable

    def __call__(self, data):
        for tr in self.transformers:
            transform = tr['callback']
            objects = tr['objects']
            rnd_value = np.random.uniform(0, 1)
            if transform.__class__ in [NormalizeObjectPose]:
                data = transform(data)
            elif transform.__class__ in [RandomRotateAndDisplacementPoints, FixPoseRotateAndDisplacementPoints, RandomRotatePoints]:
                data = transform(data, objects)
            else:
                for k, v in data.items():
                    if k in objects and k in data:
                        if transform.__class__ in [
                            RandomMirrorPoints
                        ]:
                            data[k] = transform(v, rnd_value)
                        else:
                            data[k] = transform(v)

        return data

class ToTensor(object):
    def __init__(self, parameters):
        pass

    def __call__(self, arr):
        shape = arr.shape
        if len(shape) == 3:    # RGB/Depth Images
            arr = arr.transpose(2, 0, 1)

        # Ref: https://discuss.pytorch.org/t/torch-from-numpy-not-support-negative-strides/3663/2
        return torch.from_numpy(arr.copy()).float()


class RandomSamplePoints(object):
    def __init__(self, parameters):
        self.n_points = parameters['n_points']

    def __call__(self, ptcloud):
        choice = np.random.permutation(ptcloud.shape[0])
        ptcloud = ptcloud[choice[:self.n_points]]

        if ptcloud.shape[0] < self.n_points:
            zeros = np.zeros((self.n_points - ptcloud.shape[0], 3))
            ptcloud = np.concatenate([ptcloud, zeros])

        return ptcloud

class UpSamplePoints(object):
    def __init__(self, parameters):
        self.n_points = parameters['n_points']

    def __call__(self, ptcloud):
        curr = ptcloud.shape[0]
        need = self.n_points - curr

        if need < 0:
            return ptcloud[np.random.permutation(self.n_points)]

        while curr <= need:
            ptcloud = np.tile(ptcloud, (2, 1))
            need -= curr
            curr *= 2

        choice = np.random.permutation(need)
        ptcloud = np.concatenate((ptcloud, ptcloud[choice]))

        return ptcloud


class KMeansClusterDrop(object):
    def __init__(self, parameters):
        pass

    def __call__(self, ptcloud):
        if ptcloud.shape[0] < 8:
            return ptcloud
        kmeans = cluster.KMeans(n_clusters=8, random_state=42, n_init=10, init="k-means++")
        # print(ptcloud.shape)
        kmeans.fit(ptcloud)  
        patches = kmeans.labels_ # get labels of points
        labels = np.unique(patches)
        drop_num = np.random.randint(0, 3)
        if drop_num == 0:
            return ptcloud
        # print("drop num is:", drop_num)
        keep_labels = list(np.random.choice(labels, 8 - drop_num, replace=False))
        index = np.argwhere(np.isin(patches, keep_labels))
        new_points = ptcloud[index]
        new_points = new_points.reshape(len(new_points), -1)

        return new_points
    


class RandomRotateAndDisplacementPoints(object):
    def __init__(self, parameters):
        pass

    def __call__(self, data, objects):
        rotation_matrix, rotation_quat, displacement = generate_random_Rt()
        for k in objects:
            ptcloud = data[k]
            data[k] = np.dot(rotation_matrix, ptcloud.T).T + displacement
        
        return data
    

class RandomRotatePoints(object):
    def __init__(self, parameters):
        pass

    def __call__(self, data, objects):
        rotation_matrix, rotation_quat, displacement = generate_random_Rt()
        for k in objects:
            ptcloud = data[k]
            data[k] = np.dot(rotation_matrix, ptcloud.T).T
        
        return data

    
class FixPoseRotateAndDisplacementPoints(object):
    def __init__(self, parameters):
        pass

    def __call__(self, data, objects):
        rotation_matrix = np.array([[0.8660254038, 0.3535533906, 0.3535533906], [-0.5, 0.612372431992212, 0.612372431992212], [0, -0.7071067812, 0.7071067812]])
        displacement = np.array([0.5, 0.5, 0.5])
        for k in objects:
            ptcloud = data[k]
            data[k] = np.dot(rotation_matrix, ptcloud.T).T + displacement
        
        return data


class RandomMirrorPoints(object):
    def __init__(self, parameters):
        pass

    def __call__(self, ptcloud, rnd_value):
        trfm_mat = transforms3d.zooms.zfdir2mat(1)
        trfm_mat_x = np.dot(transforms3d.zooms.zfdir2mat(-1, [1, 0, 0]), trfm_mat)
        trfm_mat_z = np.dot(transforms3d.zooms.zfdir2mat(-1, [0, 0, 1]), trfm_mat)
        if rnd_value <= 0.25:
            trfm_mat = np.dot(trfm_mat_x, trfm_mat)
            trfm_mat = np.dot(trfm_mat_z, trfm_mat)
        elif rnd_value > 0.25 and rnd_value <= 0.5:    # lgtm [py/redundant-comparison]
            trfm_mat = np.dot(trfm_mat_x, trfm_mat)
        elif rnd_value > 0.5 and rnd_value <= 0.75:
            trfm_mat = np.dot(trfm_mat_z, trfm_mat)

        ptcloud[:, :3] = np.dot(ptcloud[:, :3], trfm_mat.T)
        return ptcloud


class NormalizeObjectPose(object):
    def __init__(self, parameters):
        input_keys = parameters['input_keys']
        self.ptcloud_key = input_keys['ptcloud']
        self.bbox_key = input_keys['bbox']

    def __call__(self, data):
        ptcloud = data[self.ptcloud_key]
        bbox = data[self.bbox_key]

        # Calculate center, rotation and scale
        # References:
        # - https://github.com/wentaoyuan/pcn/blob/master/test_kitti.py#L40-L52
        center = (bbox.min(0) + bbox.max(0)) / 2
        bbox -= center
        yaw = np.arctan2(bbox[3, 1] - bbox[0, 1], bbox[3, 0] - bbox[0, 0])
        rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        bbox = np.dot(bbox, rotation)
        scale = bbox[3, 0] - bbox[0, 0]
        bbox /= scale
        ptcloud = np.dot(ptcloud - center, rotation) / scale
        ptcloud = np.dot(ptcloud, [[1, 0, 0], [0, 0, 1], [0, 1, 0]])

        data[self.ptcloud_key] = ptcloud
        return data
