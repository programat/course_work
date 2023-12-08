from abc import ABC, abstractmethod

import numpy as np


class AngleCalculationStrategy(ABC):
    @abstractmethod
    def calculate_angle(self, p1, p2, ref_pt):
        pass


class Angle2DCalculation(AngleCalculationStrategy):
    def calculate_angle(self, p1, p2, ref_pt=np.array([0, 0])):
        try:
            p1_ref = p1 - ref_pt
            p2_ref = p2 - ref_pt

            cos_theta = (np.dot(p1_ref, p2_ref)) / (1.0 * np.linalg.norm(p1_ref) * np.linalg.norm(p2_ref))
            theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))

            try:
                degree = int(180 / np.pi) * theta
            except RuntimeWarning:
                return 0

            return int(degree)
        except Exception as exc:
            print(exc)
            return 0
