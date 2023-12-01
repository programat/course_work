from abc import ABC, abstractmethod

class AngleCalculationStrategy(ABC):
    @abstractmethod
    def calculate_angles(self, landmarks):
        pass

class DefaultAngleCalculation(AngleCalculationStrategy):
    def calculate_angles(self, landmarks):
        # Реализация вычисления углов по умолчанию
        pass