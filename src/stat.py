import numpy as np
import scipy as sp
import random


class HalfPointTriangularDist:
    def __init__(self, bins: np.ndarray, values: np.ndarray):
        self.bins = bins
        self.values = values
    
    def y_values(self):
        """Ay = v -> y = inv(A) * v"""
        m = (self.bins[:-1] + self.bins[1:]) / 2
        A = np.zeros([7, 7])

        for i in range(7):
            if i == 0:
                A[0, 0] = 2 + (m[1] - self.bins[1]) / (m[1] - m[0])
                A[0, 1] = (self.bins[1] - m[0]) / (m[1] - m[0])
            elif i == 6:
                A[6, 5] = (m[6] - self.bins[6]) / (m[6] - m[5])
                A[6, 6] = 2 + (self.bins[6] - m[5]) / (m[6] - m[5])
            else:
                A[i, i-1] = (m[i] - self.bins[i]) / (m[i] - m[i-1])
                A[i, i] = 2 + (self.bins[i] - m[i-1]) / (m[i] - m[i-1]) + (m[i+1] - self.bins[i+1]) / (m[i+1] - m[i])
                A[i, i+1] = (self.bins[i+1] - m[i]) / (m[i+1] - m[i])

        v = 4 * self.values

        y = np.dot(np.linalg.inv(A), v) # -> x좌표 m[i]에 대응되는 y값
        return y

    def random_values(self, k):
        """
        어느 구간인지 / 왼쪽 절반인지 오른쪽 절반인지 / uniform인지 triangular인지
        """
        m = (self.bins[:-1] + self.bins[1:]) / 2
        y = self.y_values()
        random_values = list()

        bin_cnt = len(self.bins) - 1
        random_bins = random.choices(
            population=np.arange(bin_cnt), 
            weights=self.values * np.diff(self.bins), 
            k=k
        )

        for i in range(bin_cnt):
            if i == 0:
                left_height = 0
                right_height = (y[i] * (m[i+1] - self.bins[i+1]) + y[i+1] * (self.bins[i+1] - m[i])) / (m[i+1] - m[i])
            elif i == bin_cnt - 1:
                left_height = (y[i-1] * (m[i] - self.bins[i]) + y[i] * (self.bins[i] - m[i-1])) / (m[i] - m[i-1])
                right_height = 0
            else:
                left_height = (y[i-1] * (m[i] - self.bins[i]) + y[i] * (self.bins[i] - m[i-1])) / (m[i] - m[i-1])
                right_height = (y[i] * (m[i+1] - self.bins[i+1]) + y[i+1] * (self.bins[i+1] - m[i])) / (m[i+1] - m[i])
            
            random_lr = random.choices(
                population=["left", "right"], 
                weights=[y[i] + left_height, y[i] + right_height], 
                k=len([x for x in random_bins if x == i])
            )

            random_left_ut = random.choices(
                population=["uniform", "triangular"], 
                weights=[min([y[i], left_height]), abs(y[i] - left_height) / 2], 
                k=len([x for x in random_lr if x == "left"])
            )
            random_values += np.random.uniform(self.bins[i], m[i], len([x for x in random_left_ut if x == "uniform"])).tolist()
            peak_left = self.bins[i] if left_height >= y[i] else m[i]
            random_values += np.random.triangular(
                self.bins[i], peak_left, m[i], len([x for x in random_left_ut if x == "triangular"])
            ).tolist()

            random_right_ut = random.choices(
                population=["uniform", "triangular"], 
                weights=[min([y[i], right_height]), abs(y[i] - right_height) / 2], 
                k=len([x for x in random_lr if x == "right"])
            )
            random_values += np.random.uniform(m[i], self.bins[i+1], len([x for x in random_right_ut if x == "uniform"])).tolist()
            peak_right = m[i] if y[i] >= right_height else self.bins[i+1]
            random_values += np.random.triangular(
                m[i], peak_right, self.bins[i+1], len([x for x in random_right_ut if x == "triangular"])
            ).tolist()
        return random_values


class ExponentialDistApprox:
    def __init__(self, bins: np.ndarray, values: np.ndarray) -> None:
        self.bins = bins
        self.values = values

    def avg_time_if_exp_dist(self, lam):
        avg_time_list = []
        for i in range(len(self.bins)-1):
            nmr = self.bins[i] - self.bins[i+1] * np.exp(-lam * (self.bins[i+1] - self.bins[i]))
            dnm = 1 - np.exp(-lam * (self.bins[i+1] - self.bins[i]))
            avg_time = 1 / lam + nmr / dnm
            avg_time_list.append(avg_time)
        areas = self.values * np.diff(self.bins)
        return sum(np.array(avg_time_list) * areas) / sum(areas)
    
    def find_lambda(self):
        def equation(lam):
            return self.avg_time_if_exp_dist(lam) - 1 / lam
        sol = sp.optimize.newton(equation, 0.5)
        return sol


class StairDist:
    def __init__(self, bins, ratio) -> None:
        self.bins = bins
        self.ratio = ratio

    def random_values(self, k):
        bin_random = random.choices(
            population=self.bins[:-1], 
            weights=self.ratio, 
            k=k
        )
        random_values = list()
        for x in bin_random:
            i = np.where(self.bins == x)[0][0]
            rv = random.uniform(self.bins[i], self.bins[i+1])
            random_values.append(rv)
        return random_values


def main():
    in_room_duration_ratio = np.array([59.2, 26.5, 8.0, 2.8, 1.7, 1.3, 0.6])
    bins = np.array([0, 2, 4, 6, 8, 12, 24, 48])
    eda = ExponentialDistApprox(bins, in_room_duration_ratio / np.diff(bins))
    lam = eda.find_lambda()
    return

if __name__ == "__main__":
    main()