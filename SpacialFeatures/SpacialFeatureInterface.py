import abc


class SpacialFeatureInterface(metaclass=abc.ABCMeta):
    """
    Interface for all SpacialFeatures
    """

    @classmethod
    def __subclasshook__(cls, subclass):
        return hasattr(subclass, "apply") and callable(subclass.apply) or NotImplemented

    @abc.abstractmethod
    def apply(self, point_cloud):
        """apply feature to point cloud"""
        raise NotImplementedError
