from typing import List
import math
import os
from matplotlib.axes import Axes


def calculate_distance(lat1, lon1, lat2, lon2):
    y_diff = (lat2 - lat1) * 111.32  # 위도 1도는 약 111.32 km    
    avg_lat = math.radians((lat1 + lat2) / 2)  # 평균 위도를 라디안으로 변환
    x_diff = (lon2 - lon1) * 111.32 * math.cos(avg_lat)  # 경도 차이를 거리로 변환 (km)
    return math.sqrt(x_diff**2 + y_diff**2)


class Logger:
    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir
        self.logs = list()

    def log_append(self, log, show=True):
        self.logs.append(log)
        if show:
            print(log)
    
    def save(self):
        log_path = f"{self.output_dir}/event_log.txt"
        with open(log_path, "a+") as f:
            for log in self.logs:
                f.writelines(log + "\n")


def find_project_root(path=".", marker="src"):
    current_path = os.path.abspath(path)
    while current_path != os.path.dirname(current_path):
        if marker in os.listdir(current_path):
            return current_path
        current_path = os.path.dirname(current_path)
    return None


class StrConverter:
    @staticmethod
    def time2str(time_day_scale):
        days = math.floor(time_day_scale)
        
        hours_fraction = time_day_scale - days
        hours = math.floor(hours_fraction * 24)
        
        minutes_fraction = (hours_fraction * 24) - hours
        minutes = math.floor(minutes_fraction * 60)
        
        seconds_fraction = (minutes_fraction * 60) - minutes
        seconds = math.floor(seconds_fraction * 60)
        
        return f"{days:02d}일 {hours:02d}:{minutes:02d}:{seconds:02d}"

    @staticmethod
    def region2str(region: List[str]):
        return f"{region[0]} {region[1]} {region[2]}"


class VizToolkit:
    @staticmethod
    def vertices_regular_polygon(x, y, r, n) -> List[List[float]]:
        vertices = list()
        if n % 2 == 1:
            start_angle = -90
        elif n == 4:
            start_angle = -45
        else:
            start_angle = 0
        
        for i in range(n):
            angle = math.radians(i * (360 / n) + start_angle)
            vertex_x = x + r * math.cos(angle)
            vertex_y = y + r * math.sin(angle)
            vertices.append([vertex_x, vertex_y])
        return vertices

    @staticmethod
    def interpolate_color(start_color, end_color, val):
        """
        두 색상(start_color, end_color)과 val을 사용하여 색상 interpolation, 
        val은 0 ~ 1 사이의 값
        """
        start_red = int(start_color[1:3], 16)
        start_green = int(start_color[3:5], 16)
        start_blue = int(start_color[5:7], 16)

        end_red = int(end_color[1:3], 16)
        end_green = int(end_color[3:5], 16)
        end_blue = int(end_color[5:7], 16)

        red = int(start_red + (end_red - start_red) * val)
        green = int(start_green + (end_green - start_green) * val)
        blue = int(start_blue + (end_blue - start_blue) * val)

        return f"#{red:02x}{green:02x}{blue:02x}"
    

class PlotContext:
    def __init__(self, ax: Axes):
        self.ax = ax

    def __enter__(self) -> Axes:
        return self.ax

    def __exit__(self, exc_type, exc_val, exc_tb):
        return