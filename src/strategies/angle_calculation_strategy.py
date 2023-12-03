from abc import ABC, abstractmethod

class AngleCalculationStrategy(ABC):
    @abstractmethod
    def calculate_angles(self, landmarks):
        pass

class Angle2DCalculation(AngleCalculationStrategy):
    def calculate_angles(self, landmarks):

        pass