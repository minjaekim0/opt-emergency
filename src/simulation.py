from typing import List, Tuple, Dict, Union
import random
from datetime import datetime
from copy import deepcopy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import simpy
import tkinter as tk
from PIL import ImageGrab
import pickle

import sys
import os
sys.path.append(os.path.abspath(".."))
from src.utils import Logger, StrConverter, VizToolkit, find_project_root, PlotContext
from src.stat import ExponentialDistApprox
from src.preprocessing import RegionFilter, bed_num_imputation, WeightedAverageDistribution

sns.set_style("darkgrid")
plt.rcParams["font.family"] = "Nanum Gothic"
plt.rcParams["font.size"] = 6
plt.rcParams['axes.unicode_minus'] = False


class SimulationExecuter: 
    def __init__(
        self, 
        duration, 
        patient_cnt, 
        target_sido,
        ktas_level_distribution, 
        occupancy_duration_distribution, 
        pop_df, 
        er_df, 
        nearest_er_from_region_df, 
        travel_time_between_er_df,
        random_seed=42,
        additional_indicators=True,
        visualize=True, 
        show_arrow=False,
        show_log=True, 
        show_bar=True,
        save_data=True,
    ):  
        self.SIMULATION_START_TIME = datetime.now().strftime("%y%m%d_%H%M%S")
        if save_data:
            PROJECT_ROOT_PATH = find_project_root()
            self.output_dir = f"{PROJECT_ROOT_PATH}/output/{self.SIMULATION_START_TIME}/"
            os.mkdir(self.output_dir)
            self.logger = Logger(output_dir=self.output_dir)

        random.seed(random_seed)
        np.random.seed(random_seed)

        self.duration = duration
        self.patient_cnt = patient_cnt
        self.target_sido = target_sido
        self.ktas_level_distribution = ktas_level_distribution
        self.occupancy_duration_distribution = occupancy_duration_distribution
        self.pop_df = pop_df
        self.er_df = er_df.copy()
        self.nearest_er_from_region_df = nearest_er_from_region_df
        self.additional_indicators = additional_indicators
        self.visualize = visualize
        self.show_arrow = show_arrow
        self.show_log = show_log
        self.show_bar = show_bar
        self.save_data = save_data

        self.er_df = self.er_df[self.er_df["hperyn"] > 0]
        self._check_if_alienated_region_exists()

        nearest_er_from_region_df["region"] = nearest_er_from_region_df.apply(
            lambda x: f"{x['시도']}_{x['시군구']}_{x['읍면동']}".replace(" ", "_"), axis=1
        )
        cols = ["region"]
        for pf in ["1st", "2nd", "3rd", "center"]:
            cols += [f"nearest_er_hpid_{pf}", f"nearest_er_travel_time_{pf}"]
        self.nearest_er_from_region_mapping = \
            nearest_er_from_region_df[cols].set_index("region").to_dict("index")
        
        self.er_type_mapping = dict()
        for _, (hpid, er_type) in self.er_df[["hpid", "의료기관분류"]].iterrows():
            self.er_type_mapping[hpid] = er_type
        
        self.travel_time_between_er_mapping = dict()
        for _, (hpid1, hpid2, travel_time, _) in travel_time_between_er_df.iterrows():
            self.travel_time_between_er_mapping[f"{hpid1}_{hpid2}"] = travel_time

        self.env = simpy.Environment()

        self.er_resources = dict()
        self.er_trackers = dict()
        for row in self.er_df.itertuples():
            hpid = getattr(row, "hpid")
            num_beds = getattr(row, "hperyn")
            self.er_resources[hpid] = simpy.PriorityResource(env=self.env, capacity=num_beds)
            self.er_trackers[hpid] = ERTracker(hpid, num_beds, self.env)
                
        self.patient_trackers = {i: PatientTracker(i) for i in range(self.patient_cnt)}

        self.avg_occupancy_duration = None
        self.sampling = dict()
        self._sampling()

        assert not (additional_indicators is False and visualize is True)
        if additional_indicators:
            self.indc = IndicatorCalculator(
                self.env, target_sido, 
                self.er_resources, self.er_trackers, self.patient_trackers, 
                pop_df, self.er_df
            )
            if visualize:
                assert not (visualize is False and show_arrow is True)
                self.viz = SimulationVisualizer(
                    self.env, self.indc, target_sido, 
                    self.er_resources, self.er_trackers, self.patient_trackers, 
                    pop_df, self.er_df, self.avg_occupancy_duration, show_arrow
                )
    
    def _check_if_alienated_region_exists(self):
        """
        특정 지역의 1st, 2nd, 3rd, center 응급의료기관 4가지 모두 병상 수가 0일 경우
        해당 지역은 응급실 접근성이 매우 떨어지게 되는 상황임.
        따라서 적절치 않은 input으로 보고 error 발생시킴.
        """
        for row in self.nearest_er_from_region_df.itertuples():
            postfixes = ["1st", "2nd", "3rd", "center"]
            bools = list()
            for pf in postfixes:
                bools.append(getattr(row, "nearest_er_hpid_" + pf) not in self.er_df["hpid"].tolist())
            if all(bools):
                region = getattr(row, "시도") + "_" + getattr(row, "시군구") + "_" + getattr(row, "읍면동")
                raise Exception(f"{region}에서 1st, 2nd, 3rd, center 응급의료기관 모두 병상 수 = 0")
    
    def _sampling(self):
        # 환자 발생 시간 sampling; 단위: day
        patient_cnt_buffer = int(self.patient_cnt * 1.1)  # 시뮬레이션 마지막 즈음 환자 발생 안하는 현상 방지
        self.occurrence_time_intervals = [
            random.expovariate(self.patient_cnt) * self.duration for _ in range(patient_cnt_buffer)
        ]

        # 환자 발생 지역 sampling
        pop_df_copy = self.pop_df.copy()
        pop_df_copy["proba"] = pop_df_copy["인구수"] / pop_df_copy["인구수"].sum()
        region_list = [
            Region(getattr(row, "시도"), getattr(row, "시군구"), getattr(row, "읍면동"))
            for row in pop_df_copy.itertuples()
        ]
        occurred_regions = random.choices(
            population=region_list,
            weights=pop_df_copy["proba"].tolist(),
            k=self.patient_cnt
        )
        
        # ktas 단계 sampling
        ktas_levels = random.choices(
            population=["ktas12", "ktas3", "ktas45"], 
            weights=self.ktas_level_distribution, 
            k=self.patient_cnt
        )

        # 환자 응급실 재실 시간 sampling
        occupancy_duration_bins = np.array([0, 2, 4, 6, 8, 12, 24, 48]) / 24  # day 단위
        occupancy_durations = dict()
        self.avg_occupancy_duration = dict()

        for ktas, dist in self.occupancy_duration_distribution.items():
            exponential_dist_approx = ExponentialDistApprox(
                bins=occupancy_duration_bins, 
                values=dist / np.diff(occupancy_duration_bins)
            )
            lam = exponential_dist_approx.find_lambda()
            self.avg_occupancy_duration[ktas] = 1 / lam
            occupancy_durations[ktas] = np.random.exponential(1 / lam, ktas_levels.count(ktas)).tolist()
        self.avg_occupancy_duration_all = sum(
            np.array(list(self.avg_occupancy_duration.values())) * self.ktas_level_distribution
        ) / sum(self.ktas_level_distribution)
        
        # summarize
        occupancy_durations_copy = deepcopy(occupancy_durations)
        for i in range(self.patient_cnt):
            self.sampling[i] = {
                "occurrence_region": occurred_regions[i],
                "ktas_level": ktas_levels[i],
            }
            self.sampling[i]["occupancy_duration"] = occupancy_durations_copy[ktas_levels[i]].pop()
        del occupancy_durations_copy
        
    def run(self):
        """시뮬레이션 실행"""
        self.env.process(self.patient_occur())
        if self.additional_indicators:
            self.env.process(self.indc.keep_updating())
        if self.visualize:
            self.env.process(self.viz.keep_plotting())
        
        end_time_smoother = 1 / (24 * 60)
        self.env.run(until=self.duration + end_time_smoother)
        
        print()
        if self.additional_indicators and self.save_data:
            self._save_data()
            self.logger.save()
        if self.visualize:
            self.viz.main.mainloop()
        
    def _save_data(self):
        avg_occupancy_rate_each_hpid_df = pd.DataFrame(index=self.er_df.hpid, columns=["avg_occupancy_rate"])
        for h, vals in self.indc.each_occupancy_rate_record.items():
            avg_occupancy_rate_each_hpid_df.loc[h, "avg_occupancy_rate"] = np.mean(vals)
        avg_occupancy_rate_each_hpid_df.to_csv(self.output_dir + f"avg_occupancy_rate_each_hpid.csv")

        with open(self.output_dir + f"time_to_start_treatment.pickle", "wb") as f:
            pickle.dump(self.indc.time_to_start_treatment_data, f)

        if self.visualize:
            display_scaling_factor = 2
            x1 = self.viz.main.winfo_rootx() * display_scaling_factor
            y1 = self.viz.main.winfo_rooty() * display_scaling_factor
            x2 = x1 + self.viz.main.winfo_width() * display_scaling_factor
            y2 = y1 + self.viz.main.winfo_height() * display_scaling_factor
            ImageGrab.grab().crop((x1, y1, x2, y2)).save(self.output_dir + f"canvas_image.png")
    
    def patient_occur(self):
        """신규 환자가 계속해서 발생"""

        patient_id = 0
        while patient_id < self.patient_cnt - 1:
            time_interval = self.occurrence_time_intervals.pop()
            yield self.env.timeout(time_interval)

            region = self.sampling[patient_id]["occurrence_region"]
            self.patient_trackers[patient_id].occur(self.env.now, region)

            if self.save_data:
                log = f"{StrConverter.time2str(self.env.now)}: " + \
                    f"환자 {patient_id} 발생; {region}"
                self.logger.log_append(log, show=self.show_log)            

            goal_hpid, goal_travel_time = self._select_er(region)
            self.env.process(self.transport_patient(region, goal_hpid, patient_id, goal_travel_time, by_ktas=False))
            patient_id += 1

            assert not (self.show_log is True and self.show_bar is True)
            if self.show_log is False and self.show_bar is True:
                time_elapsed = datetime.now() - datetime.strptime(self.SIMULATION_START_TIME, '%y%m%d_%H%M%S')
                print(f"\r[{time_elapsed}] {patient_id + 1} / {self.patient_cnt} ", end="")
            
    def _select_er(self, region, max_time_diff=15, center_weight=5) -> Tuple[str, float]:
        """
        nearest er 3개 중에서, 이동시간이 최소시간과 10분 차이 이내인 병원들 중에서 랜덤으로 선택
        권역/지역응급센터 -> 선택 가중치 3배
        """
        postfixes = ["1st", "2nd", "3rd", "center"]
        info_dict = dict()
        
        for pf in postfixes:
            hpid = self.nearest_er_from_region_mapping[str(region)][f"nearest_er_hpid_{pf}"]
            if hpid not in self.er_df["hpid"].tolist():
                continue
            er_type = self.er_type_mapping[hpid]
            travel_time = self.nearest_er_from_region_mapping[str(region)][f"nearest_er_travel_time_{pf}"]
            info_dict[pf] = {"hpid": hpid, "er_type": er_type, "travel_time": travel_time}
        min_travel_time = min([x["travel_time"] for x in info_dict.values()])
        
        er_select_weights = list()
        for info in info_dict.values():
            travel_time_diff = info["travel_time"] - min_travel_time
            weight = max(0, (max_time_diff - travel_time_diff) / max_time_diff)
            if info["er_type"] in ["권역응급의료센터", "지역응급의료센터"]:
                weight *= center_weight
            er_select_weights.append(weight)

        selected_pf = random.choices(population=list(info_dict.keys()), weights=er_select_weights, k=1)[0]
        goal_hpid_ = info_dict[selected_pf]["hpid"]

        transport_etc_time = 15 / (60 * 24)
        goal_travel_time_ = info_dict[selected_pf]["travel_time"] / (24 * 60) + transport_etc_time

        return goal_hpid_, goal_travel_time_

    def transport_patient(self, region, hpid_to, patient_id, travel_time, by_ktas=False):
        """이송: 지역 -> 병원"""

        yield self.env.timeout(travel_time)
        self.env.process(self._check_availability(hpid_to, patient_id, by_ktas))

        if self.save_data:
            log = f"{StrConverter.time2str(self.env.now)}: " + \
                f"환자 {patient_id} 이송; {region} -> {hpid_to}"
            self.logger.log_append(log, show=self.show_log)

        if self.visualize and self.show_arrow:
            self.viz.arrow_graphics.new_arrow_from_region(region, hpid_to, self.env.now)

    def transfer_patient(self, hpid_from, hpid_to, patient_id, travel_time, by_ktas=False):
        """전원: 병원 -> 병원"""

        yield self.env.timeout(travel_time)
        self.env.process(self._check_availability(hpid_to, patient_id, by_ktas))

        if self.save_data:
            log = f"{StrConverter.time2str(self.env.now)}: " + \
                f"환자 {patient_id} 전원; {hpid_from} -> {hpid_to}"
            self.logger.log_append(log, show=self.show_log)
        
        self.patient_trackers[patient_id].transfer(by_ktas)
        
        if self.visualize and self.show_arrow:
            self.viz.arrow_graphics.new_arrow_from_er(hpid_from, hpid_to, self.env.now)

    def _check_availability(self, hpid, patient_id, by_ktas=False):
        """병원에서 환자 수용 가능한지 판단"""

        yield self.env.timeout(0)

        er_type = self.er_type_mapping[hpid]
        ktas_level = self.sampling[patient_id]["ktas_level"]
        transfer_etc_time = 20 / (60 * 24)
        
        # ktas 의한 우선순위 해당하여 전원을 온 경우: 무조건 수용
        if by_ktas:
            self.env.process(self.receive_patient(hpid, patient_id))

        # ktas 단계 상 현재 병원에서 치료 불가능하면 무조건적 전원
        elif self._is_available_to_cover(er_type, ktas_level) is False:
            goal_er_type = self._available_er_type(ktas_level)
            best_higher_hpid = self._near_hpid_best_to_transfer(hpid, goal_er_type=goal_er_type, must_transfer=True)

            travel_time = self.travel_time_between_er_mapping[f"{hpid}_{best_higher_hpid}"] / (24 * 60) + transfer_etc_time
            self.env.process(
                self.transfer_patient(hpid, best_higher_hpid, patient_id, travel_time, by_ktas=True)
            )

        # capa 꽉 찼을 때: 근처 병원 queue 확인 후, 전원시키는 게 나을지 이동시간 고려하여 판단
        elif self._is_full_capa(hpid):
            goal_er_type = self._available_er_type(ktas_level)
            better_hpid = self._near_hpid_best_to_transfer(hpid, goal_er_type=goal_er_type, must_transfer=False)

            if better_hpid is not None:
                travel_time = self.travel_time_between_er_mapping[f"{hpid}_{better_hpid}"] / (24 * 60) + transfer_etc_time
                self.env.process(
                    self.transfer_patient(hpid, better_hpid, patient_id, travel_time, by_ktas=False)
                )
            else:
                self.env.process(self.receive_patient(hpid, patient_id))
        else:
            self.env.process(self.receive_patient(hpid, patient_id))

    def receive_patient(self, hpid, patient_id):
        """환자 수용"""

        yield self.env.timeout(0)

        self.patient_trackers[patient_id].arrive(self.env.now)

        if self.visualize:
            self.viz.er_graphics.update_cnt(hpid)

        if self.save_data:
            log = f"{StrConverter.time2str(self.env.now)}: 환자 {patient_id} 수용; {hpid}"
            self.logger.log_append(log, show=self.show_log)

        self.env.process(self.treat_patient(hpid, patient_id))

    def treat_patient(self, hpid, patient_id):
        """환자 치료"""

        req = self.er_resources[hpid].request(self._ktas2priority(self.sampling[patient_id]["ktas_level"]))
        yield req

        occupancy_duration = self.sampling[patient_id]["occupancy_duration"]
        self.er_trackers[hpid].start_treatment(patient_id, occupancy_duration)
        self.patient_trackers[patient_id].start_treatment(self.env.now, occupancy_duration)

        if self.save_data:
            log = f"{StrConverter.time2str(self.env.now)}: 환자 {patient_id} 치료; {hpid}"
            self.logger.log_append(log, show=self.show_log)

        yield self.env.timeout(occupancy_duration)
        
        self.er_resources[hpid].release(req)
        self.patient_trackers[patient_id].complete_treatment(self.env.now)
        self.er_trackers[hpid].complete_treatment(patient_id)

        if self.visualize:
            self.viz.er_graphics.update_cnt(hpid)

        if self.save_data:
            log = f"{StrConverter.time2str(self.env.now)}: 환자 {patient_id} 퇴원; {hpid}"
            self.logger.log_append(log, show=self.show_log)

    @staticmethod
    def _ktas2priority(ktas_level):
        if ktas_level == "ktas12": 
            return 0
        elif ktas_level == "ktas3": 
            return 1
        else:  # ktas_level == "ktas45"
            return 2
    
    @staticmethod
    def _available_er_type(ktas_level):
        if ktas_level == "ktas12":
            return ["권역응급의료센터", "지역응급의료센터"]
        else:
            return [
                "권역응급의료센터", "지역응급의료센터", 
                "지역응급의료기관", "응급의료기관 외의 의료기관(응급의료시설)"
            ]
    
    def _is_available_to_cover(self, er_type, ktas_level):
        return er_type in self._available_er_type(ktas_level)
    
    def _is_full_capa(self, hpid):
        return self.er_resources[hpid].count == self.er_resources[hpid].capacity
    
    def _near_ers_from_er(self, hpid, N=5, goal_er_type=None) -> Dict[str, float]:
        if type(goal_er_type) == list:
            target_hpids = [k for k, v in self.er_type_mapping.items() if v in goal_er_type]
        elif goal_er_type:
            target_hpids = [k for k, v in self.er_type_mapping.items() if v == goal_er_type]
        else:
            target_hpids = [k for k in self.er_type_mapping.keys()]
        target_hpids = [target for target in target_hpids if target != hpid]
        
        return_cnt = min(N, len(target_hpids))
        target_hpid_travel_times = [
            self.travel_time_between_er_mapping[f"{hpid}_{target}"] / (24 * 60) 
            for target in target_hpids
        ]
        min_time_indices = np.argsort(target_hpid_travel_times)[:return_cnt]
        out = {target_hpids[i]: target_hpid_travel_times[i] for i in min_time_indices}
        return out

    def _near_hpid_best_to_transfer(self, hpid, goal_er_type, must_transfer) -> Union[None, str]:
        hpid_travel_time_mapping = self._near_ers_from_er(hpid, goal_er_type=goal_er_type)
        
        # 구급차에 환자 태우고 내리는 등 기타 소요시간 입퇴원 각각 10분씩 총 20분
        transfer_etc_time = 20 / (60 * 24)
        expected_total_time = dict()
        for h, travel_time in hpid_travel_time_mapping.items():
            if not self._is_full_capa(h):
                waiting_time = 0
            elif len(self.er_resources[h].queue) == 0:
                waiting_time = min([
                    complete_time - self.env.now 
                    for complete_time in self.er_trackers[h].current_patient_treatment_complete_time.values()
                ])
            else:
                patient_cnt = self.er_resources[h].capacity + len(self.er_resources[h].queue)
                alpha = patient_cnt - self.er_resources[h].capacity + 1
                beta = self.avg_occupancy_duration_all / self.er_resources[h].capacity
                waiting_time = alpha * beta  # gamma distribution's expected value

            expected_total_time[h] = travel_time + waiting_time + transfer_etc_time

        min_expected_total_time = min(expected_total_time.values())
        min_expected_total_time_hpid = min(expected_total_time, key=expected_total_time.get)

        if must_transfer:
            return min_expected_total_time_hpid
        else:
            current_expected_waiting_time = self.er_trackers[hpid].expected_remaining_time() + \
                self.avg_occupancy_duration_all * (len(self.er_resources[hpid].queue) / self.er_resources[hpid].capacity)
            
            if min_expected_total_time < current_expected_waiting_time:
                return min_expected_total_time_hpid
            else:
                return None


class Region:
    def __init__(self, sido, sigungu, eupmyeondong):
        self.sido = sido
        self.sigungu = sigungu
        self.eupmyeondong = eupmyeondong

    def __str__(self):
        out = f"{self.sido}_{self.sigungu}_{self.eupmyeondong}"
        out = out.replace(" ", "_")
        return out


class ERTracker:
    """응급실의 진료 상황 트래킹"""

    def __init__(self, hpid, capacity, env):
        self.hpid = hpid
        self.capacity = capacity
        self.env = env
        self.current_patient_treatment_start_time = dict()
        self.current_patient_treatment_complete_time = dict()
        self.treated_patient_cnt = 0

    def start_treatment(self, patient_id, treatment_duration):
        self.current_patient_treatment_start_time[patient_id] = self.env.now
        self.current_patient_treatment_complete_time[patient_id] = self.env.now + treatment_duration

    def complete_treatment(self, patient_id):
        del self.current_patient_treatment_start_time[patient_id]
        del self.current_patient_treatment_complete_time[patient_id]
        self.treated_patient_cnt += 1
    
    def expected_remaining_time(self):
        if self.capacity > len(self.current_patient_treatment_start_time):
            return 0
        else:
            current_patient_time_remaining = {
                patient_id: complete_time - self.env.now
                for patient_id, complete_time in self.current_patient_treatment_complete_time.items()
            }
            min_time_remaining = min(current_patient_time_remaining.values())
            return min_time_remaining


class PatientTracker:
    """환자의 진료 상황 트래킹"""

    def __init__(self, patient_id):
        self.patient_id = patient_id
        
        self.occurred_time = None
        self.occured_region = Region(None, None, None)
        self.arrival_time = None
        self.start_treatment_time = None
        self.complete_treatment_time = None

        self.transporting_duration = None
        self.waiting_duration = None
        self.treatment_duration = None

        self.occurred = False
        self.ktas_transferred = False
        self.normal_transferred = False
        self.transferred = self.ktas_transferred or self.normal_transferred
        self.arrived = False
        self.started_treatment = False
        self.treated = False
    
    def occur(self, t, region):
        self.occurred_time = t
        self.occured_region = region
        self.occurred = True

    def transfer(self, by_ktas=False):
        if by_ktas is True:
            self.ktas_transferred = True
        else:
            self.normal_transferred = True
    
    def arrive(self, t):
        self.arrival_time = t
        self.transporting_duration = t - self.occurred_time
        self.arrived = True
    
    def start_treatment(self, t, treatment_duration):
        self.start_treatment_time = t
        self.waiting_duration = t - self.arrival_time
        self.started_treatment = True
        self.treatment_duration = treatment_duration
    
    def complete_treatment(self, t):
        self.complete_treatment_time = t
        self.treated = True


class IndicatorCalculator:
    def __init__(
        self, 
        env, 
        target_sido, 
        er_resources: Dict[str, simpy.Resource], 
        er_trackers: Dict[str, ERTracker],
        patient_trackers: Dict[int, PatientTracker], 
        pop_df, 
        er_df
    ):
        self.env = env
        self.target_sido = target_sido
        self.er_resources = er_resources
        self.er_trackers = er_trackers
        self.patient_trackers = patient_trackers
        self.pop_df = pop_df
        self.er_df = er_df

        self.er_types_simple = ["센터", "else"]
        self.time_to_start_treatment_data = {sido: list() for sido in target_sido}
        self.each_occupancy_rate_record = {hpid: list() for hpid in self.er_trackers.keys()}
        self.occupancy_rate_data = {
            sido: {er_type: list() for er_type in self.er_types_simple}
            for sido in target_sido
        }
        self.normal_transference_rate_record = {sido: list() for sido in target_sido}
        self.ktas_transference_rate_record = {sido: list() for sido in target_sido}
        self.queue_len_data = {
            sido: {er_type: list() for er_type in self.er_types_simple}
            for sido in target_sido
        }

        self.hpid_mapping = self.er_df[["hpid", "시도", "의료기관분류"]].set_index("hpid").to_dict("index")

    def keep_updating(self):
        while True:
            yield self.env.timeout(1 / 24)  # 1시간 단위 plot
            self._update()

    def _update(self):
        # 0.
        for sido in self.target_sido:
            self.time_to_start_treatment_data[sido] = list()
            for p in self.patient_trackers.values():
                if p.started_treatment is True and p.occured_region.sido == sido:
                    self.time_to_start_treatment_data[sido].append(
                        (p.start_treatment_time - p.occurred_time) * (24 * 60)
                    )

        # 1.
        for hpid, er in self.er_resources.items():
            self.each_occupancy_rate_record[hpid].append(er.count / er.capacity * 100)

        for sido in self.target_sido:
            for er_type in self.er_types_simple:
                self.occupancy_rate_data[sido][er_type] = list()

                for hpid, vals in self.each_occupancy_rate_record.items():
                    sido_hpid, er_type_hpid = self.hpid_mapping[hpid].values()

                    if sido_hpid == sido:
                        if er_type == "센터":
                            if er_type_hpid[-2:] == "센터":
                                self.occupancy_rate_data[sido][er_type].append(np.mean(vals))
                        else:
                            if er_type_hpid[-2:] != "센터":
                                self.occupancy_rate_data[sido][er_type].append(np.mean(vals))

        # 2.
        total_patient_cnt = sum([p.occurred for p in self.patient_trackers.values()])
        if total_patient_cnt > 0:
            for sido in self.target_sido:
                normal_transferred_cnt = sum([
                    p.normal_transferred for p in self.patient_trackers.values()
                    if p.occured_region.sido == sido
                ])
                normal_transference_rate = normal_transferred_cnt / total_patient_cnt * 100  # percent
                self.normal_transference_rate_record[sido].append(normal_transference_rate)

                ktas_transferred_cnt = sum([
                    p.ktas_transferred for p in self.patient_trackers.values()
                    if p.occured_region.sido == sido
                ])
                ktas_transference_rate = ktas_transferred_cnt / total_patient_cnt * 100  # percent
                self.ktas_transference_rate_record[sido].append(ktas_transference_rate)
        else:
            for sido in self.target_sido:
                self.normal_transference_rate_record[sido].append(None)
                self.ktas_transference_rate_record[sido].append(None)

        # 3.
        for sido in self.target_sido:
            for er_type in self.er_types_simple:
                self.queue_len_data[sido][er_type] = list()

                for hpid, resoruce in self.er_resources.items():
                    sido_hpid, er_type_hpid = self.hpid_mapping[hpid].values()

                    if sido_hpid == sido:
                        if er_type == "센터":
                            if er_type_hpid[-2:] == "센터":
                                self.queue_len_data[sido][er_type].append(len(resoruce.queue))
                        else:
                            if er_type_hpid[-2:] != "센터":
                                self.queue_len_data[sido][er_type].append(len(resoruce.queue))


class SimulationVisualizer:
    def __init__(
        self, 
        env, 
        indicator_calculator: IndicatorCalculator,
        target_sido,
        er_resources: Dict[str, simpy.Resource], 
        er_trackers: Dict[str, ERTracker],
        patient_trackers: Dict[int, PatientTracker], 
        pop_df, 
        er_df, 
        avg_occupancy_duration, 
        show_arrow
    ):
        self.env = env
        self.indicator_calculator = indicator_calculator
        self.target_sido = target_sido
        self.er_resources = er_resources
        self.er_trackers = er_trackers
        self.patient_trackers = patient_trackers
        self.pop_df = pop_df
        self.er_df = er_df
        self.avg_occupancy_duration = avg_occupancy_duration
        self.show_arrow = show_arrow

        self.main = tk.Tk()
        self.main.title("ER Simulation")
        self.main.config(bg="white")
        self.canvas = tk.Canvas(self.main, width=600, height=600, bg="white")
        self.canvas.pack(side=tk.LEFT, expand=False)
        self.main.after(10, None)  # ERGraphics 내에서 canvas의 height, width 측정 위해 필요

        fig, self.axes = plt.subplots(4, len(self.target_sido), dpi=80)
        fig.subplots_adjust(wspace=0.5, hspace=0.7, top=0.9, bottom=0.1, left=0.15, right=0.95)

        self.data_plot = FigureCanvasTkAgg(fig, master=self.main)
        self.data_plot.get_tk_widget().config(width=350*len(self.target_sido), height=1000)
        self.data_plot.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        end_lon = [pop_df["경도"].min(), pop_df["경도"].max()]
        end_lat = [pop_df["위도"].min(), pop_df["위도"].max()]
        self.pop_graphics = PopulationGraphics(self.canvas, pop_df, end_lon, end_lat)
        self.er_graphics = ERGraphics(self.canvas, self.er_resources, er_df, end_lon, end_lat)
        if show_arrow:
            self.arrow_graphics = ArrowGraphics(self.canvas, er_df, pop_df, end_lon, end_lat)

        self.time_list = list()
        self.time_text = self.canvas.create_text(
            50, 50, text=f"Time = {StrConverter.time2str(self.env.now)}", 
            anchor=tk.NW, fill="black"
        )

    def keep_plotting(self):
        while True:
            yield self.env.timeout(1 / 24)  # 1시간 단위 plot
            self._update_time()

            if self.show_arrow:
                self.arrow_graphics.update_visibility(self.env.now)
                self.er_graphics.draw()
            self.canvas.update()

            elapsed_hours = round(self.env.now * 24)
            if elapsed_hours % 6 == 0:
                self._plot()

    def _update_time(self):
        self.canvas.delete(self.time_text)
        self.time_text = self.canvas.create_text(
            50, 50, text=f"Time = {StrConverter.time2str(self.env.now)}", 
            anchor=tk.NW, fill="black"
        )
        self.time_list.append(self.env.now)
    
    def _plot(self):
        for col, sido in enumerate(self.target_sido):
            with PlotContext(self.axes[0, col]) as ax:
                ax.cla()
                sns.histplot(self.indicator_calculator.time_to_start_treatment_data[sido], bins=np.arange(0, 95, 5), ax=ax)
                # sns.kdeplot(self.time_to_start_treatment_data, clip=(0, 100), color=color, linewidth=1, ax=ax)
                ax.axvline(x=30, color="red", linewidth=1, linestyle=":")
                ax.set_title(sido + "\n")
                ax.set_xlabel("Time to start treatment (min.)")
                ax.set_ylabel("Patient cnt")
                
            with PlotContext(self.axes[1, col]) as ax:
                ax.cla()
                for er_type in self.indicator_calculator.er_types_simple:
                    sns.histplot(
                        self.indicator_calculator.occupancy_rate_data[sido][er_type], bins=np.arange(0, 110, 10),
                        linewidth=0, alpha=0.3, label=er_type, ax=ax
                    )
                ax.set_xlabel("Occupancy rate (%)")
                ax.set_ylabel("ER cnt")
                ax.legend()

            with PlotContext(self.axes[2, col]) as ax:
                ax.cla()
                ax.plot(
                    self.time_list, self.indicator_calculator.ktas_transference_rate_record[sido], 
                    linewidth=1, label="by ktas"
                )
                ax.plot(
                    self.time_list, self.indicator_calculator.normal_transference_rate_record[sido], 
                    linewidth=1, linestyle=":", label="full capa"
                )
                ax.set_xlabel("Time (day)")
                ax.set_ylabel("Transference rate (%)")
                ax.legend()

            with PlotContext(self.axes[3, col]) as ax:
                ax.cla()
                for er_type in self.indicator_calculator.er_types_simple:
                    sns.histplot(
                        self.indicator_calculator.queue_len_data[sido][er_type], bins=np.arange(0, 10, 1),
                        linewidth=0, alpha=0.3, label=er_type, ax=ax
                    )
                ax.set_xlabel("queue length")
                ax.set_ylabel("ER cnt")
                ax.legend()

        self.data_plot.draw()


class ERGraphics:
    MARGIN = 0.1
    
    def __init__(
        self, 
        canvas, 
        er_resources: Dict[str, simpy.Resource], 
        er_df, 
        end_lon: List[float], 
        end_lat: List[float], 
        show_name=False
    ):
        assert {"기관명", "의료기관분류", "경도", "위도"}.issubset(set(er_df.columns))
        self.canvas = canvas
        self.er_resources = er_resources
        self.er_df = er_df.copy()
        self.end_lon = end_lon
        self.end_lat = end_lat
        self.show_name = show_name
        
        self.draw(initial=True)
        self.canvas.update()

    def draw(self, initial=False):
        if initial is False:
            for hpid in self.er_df["hpid"].tolist():
                self.canvas.delete(hpid)
                
        er_type_order = [
            "권역응급의료센터", 
            "지역응급의료센터", 
            "지역응급의료기관", 
            "응급의료기관 외의 의료기관(응급의료시설)"
        ]
        self.er_df["의료기관분류"] = pd.Categorical(self.er_df["의료기관분류"], er_type_order)
        self.er_df.sort_values("의료기관분류", ascending=False, inplace=True)

        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        for row in self.er_df.itertuples():
            hpid = getattr(row, "hpid")
            er_name = getattr(row, "기관명")
            er_type = getattr(row, "의료기관분류")
            lon = getattr(row, "경도")
            lat = getattr(row, "위도")

            x = canvas_width * (
                self.MARGIN + (1 - 2 * self.MARGIN) * (lon - self.end_lon[0]) / (self.end_lon[1] - self.end_lon[0])
            )
            y = canvas_height * (
                self.MARGIN + (1 - 2 * self.MARGIN) * (1 - (lat - self.end_lat[0]) / (self.end_lat[1] - self.end_lat[0]))
            )
            
            if er_type == "권역응급의료센터":
                n = 6
            elif er_type == "지역응급의료센터":
                n = 5
            elif er_type == "지역응급의료기관":
                n = 4
            else:
                n = 3
            vertices = VizToolkit.vertices_regular_polygon(x, y, r=7, n=n)

            if initial:
                color = "#00FF00"  # green
            else:
                color = self._interpolated_color(hpid)
            self.canvas.create_polygon(vertices, outline="black", fill=color, tag=hpid)

            if self.show_name:
                self.canvas.create_text(x+10, y, anchor="nw", text=er_name, fill="black", font=("NanumGothic", 6))
        
    def update_cnt(self, hpid):
        color = self._interpolated_color(hpid)
        self.canvas.itemconfigure(hpid, fill=color)
    
    def _interpolated_color(self, hpid):
        cnt = self.er_resources[hpid].count
        capa = self.er_resources[hpid].capacity
        color = VizToolkit.interpolate_color("00FF00", "FF0000", cnt / capa)
        return color


class PopulationGraphics:
    MARGIN = 0.1
    
    def __init__(
        self, 
        canvas, 
        pop_df, 
        end_lon: List[float], 
        end_lat: List[float], 
        show_name=False
    ):
        assert {"읍면동", "인구수", "경도", "위도"}.issubset(set(pop_df.columns))
        
        self.canvas = canvas
        self.pop_df = pop_df.copy()
        self.end_lon = end_lon
        self.end_lat = end_lat
        self.show_name = show_name
        
        self.draw()
        self.canvas.update()

    def draw(self, initial=False):
        if initial is False:
            for row in self.pop_df.itertuples():
                full_name = f"{getattr(row, '시도')}_{getattr(row, '시군구')}_{getattr(row, '읍면동')}"
                self.canvas.delete(full_name)

        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        for row in self.pop_df.itertuples():
            lon = getattr(row, "경도")
            lat = getattr(row, "위도")
            dong_name = getattr(row, "읍면동")
            full_name = f"{getattr(row, '시도')}_{getattr(row, '시군구')}_{getattr(row, '읍면동')}"

            x = canvas_width * (
                self.MARGIN + (1 - 2 * self.MARGIN) * (lon - self.end_lon[0]) / (self.end_lon[1] - self.end_lon[0])
            )
            y = canvas_height * (
                self.MARGIN + (1 - 2 * self.MARGIN) * (1 - (lat - self.end_lat[0]) / (self.end_lat[1] - self.end_lat[0]))
            )
            r = np.sqrt(getattr(row, "인구수") / self.pop_df.인구수.max() * 300)

            self.canvas.create_oval(x-r, y-r, x+r, y+r, outline="gray", fill="white", tag=full_name)
            
            if self.show_name:
                self.canvas.create_text(x+5, y, anchor="nw", text=dong_name, fill="black", font=("NanumGothic", 6))


class ArrowGraphics:
    MARGIN = 0.1
    KEEPING_ARROW_TIME = 6 / 24
    INITIAL_ARROW_THICKNESS = {"transfer": 4, "transport": 2}

    def __init__(
        self, 
        canvas, 
        er_df, 
        pop_df, 
        end_lon: List[float], 
        end_lat: List[float]
    ):
        self.canvas = canvas
        self.er_df = er_df
        self.pop_df = pop_df
        self.end_lon = end_lon
        self.end_lat = end_lat
        self.created_time_mapping = dict()

    def new_arrow_from_region(self, region_from: Region, hpid_to, now: float):
        lon1 = self.pop_df[
            (self.pop_df["시도"] == region_from.sido) &
            (self.pop_df["시군구"] == region_from.sigungu) &
            (self.pop_df["읍면동"] == region_from.eupmyeondong)
        ]["경도"].iloc[0]
        lat1 = self.pop_df[
            (self.pop_df["시도"] == region_from.sido) &
            (self.pop_df["시군구"] == region_from.sigungu) &
            (self.pop_df["읍면동"] == region_from.eupmyeondong)
        ]["위도"].iloc[0]
        x1 = self._lon2x(lon1)
        y1 = self._lat2y(lat1)
        x2 = self._lon2x(self.er_df[self.er_df["hpid"] == hpid_to]["경도"].iloc[0])
        y2 = self._lat2y(self.er_df[self.er_df["hpid"] == hpid_to]["위도"].iloc[0])
        tag = f"transport__{region_from}__{hpid_to}"
        
        self.canvas.create_line(
            x1, y1, x2, y2, arrow=tk.LAST, 
            width=self.INITIAL_ARROW_THICKNESS["transport"], fill="#666666", tag=tag
        )
        self.created_time_mapping[tag] = now

    def new_arrow_from_er(self, hpid_from, hpid_to, now: float):
        x1 = self._lon2x(self.er_df[self.er_df["hpid"] == hpid_from]["경도"].iloc[0])
        y1 = self._lat2y(self.er_df[self.er_df["hpid"] == hpid_from]["위도"].iloc[0])
        x2 = self._lon2x(self.er_df[self.er_df["hpid"] == hpid_to]["경도"].iloc[0])
        y2 = self._lat2y(self.er_df[self.er_df["hpid"] == hpid_to]["위도"].iloc[0])
        tag = f"transfer__{hpid_from}__{hpid_to}"

        self.canvas.create_line(
            x1, y1, x2, y2, arrow=tk.LAST, 
            width=self.INITIAL_ARROW_THICKNESS["transfer"], 
            dash=(5, 5), fill="#DD7700", tag=tag
        )
        self.created_time_mapping[tag] = now

    def update_visibility(self, now):
        removal_list = list()
        for tag, created_time in self.created_time_mapping.items():
            movement_type = tag.split("__")[0]
            elapsed_time = now - created_time
            width = self.INITIAL_ARROW_THICKNESS[movement_type] * (1 - elapsed_time / (3 / 24))
            if width > 0:
                self.canvas.itemconfigure(tag, width=width)
            else:
                self.canvas.delete(tag)

                removal_list.append(tag)
        for tag in removal_list:
            del self.created_time_mapping[tag]

    def _lon2x(self, lon):
        return self.canvas.winfo_width() * (
            self.MARGIN + (1 - 2 * self.MARGIN) * 
            (lon - self.end_lon[0]) / (self.end_lon[1] - self.end_lon[0])
        )

    def _lat2y(self, lat):
        return self.canvas.winfo_height() * (
            self.MARGIN + (1 - 2 * self.MARGIN) * 
            (1 - (lat - self.end_lat[0]) / (self.end_lat[1] - self.end_lat[0]))
        )


def main():
    emergency_df = pd.read_csv("../data/processed/emergency_df.csv")
    population_df = pd.read_csv("../data/processed/community_population_df.csv")
    nearest_er_df = pd.read_csv("../data/processed/nearest_er_tk_daytime.csv")
    travel_time_between_er_df = pd.read_csv("../data/processed/travel_time_between_er_tk_daytime.csv")
    ktas_level_ratio_df = pd.read_csv("../data/processed/ktas_level_ratio.csv").astype({"ktas": str})
    occupancy_duration_df = pd.read_csv("../data/processed/occupancy_duration_ktas_corr.csv")
    target_sido = ["대구", "경북"]
    sigungu_excluded = ["울릉군"]
    
    rf = RegionFilter(emergency_df, population_df, travel_time_between_er_df, target_sido, sigungu_excluded)
    emergency_df = rf.emergency_df_filtering()
    population_df = rf.population_df_filtering()
    travel_time_between_er_df = rf.travel_time_between_er_df_filtering()
    bed_num_imputation(emergency_df)
    wa = WeightedAverageDistribution(ktas_level_ratio_df, occupancy_duration_df, population_df, target_sido)
    ktas_level_distribution = wa.ktas_level_ratio()
    occupancy_duration_distribution = wa.occupancy_duration()

    params = {'A2700014': -3,
        'A1300018': -10,
        'A2700018': -6,
        'A2700001': -8,
        'A2700005': -10,
        'A1303386': -10,
        'A2700015': -7,
        'A1300005': -10,
        'A1300076': -7,
        'A1300007': -10,
        'A1300110': -10,
        'A1300034': -10,
        'A1300045': -9,
        'A2700011': -10,
        'A2702752': -4,
        'A2700009': -2,
        'A2702563': -9,
        'A1300011': -6,
        'A2700052': -5,
        'A2700038': -6,
        'A2700023': -4,
        'A2700070': -5,
        'A2700063': -4,
        'E2700554': -4,
        'A2700097': -10,
        'A2700071': -5,
        'A2700030': -2,
        'A2700020': -8,
        'A2700026': -2,
        'A2700066': -4}
    for k, v in params.items():
        emergency_df.loc[emergency_df.hpid == k, "hperyn"] += v

    DURATION = 30
    PATIENT_CNT_ONE_MONTH = 38000
    PATIENT_CNT = int(PATIENT_CNT_ONE_MONTH / 30 * DURATION)  # 38000 / 1month

    executer = SimulationExecuter(
        duration=DURATION, 
        patient_cnt=PATIENT_CNT,
        target_sido=target_sido,
        ktas_level_distribution=ktas_level_distribution,
        occupancy_duration_distribution=occupancy_duration_distribution,
        pop_df=population_df,
        er_df=emergency_df,
        nearest_er_from_region_df=nearest_er_df,
        travel_time_between_er_df=travel_time_between_er_df,
        additional_indicators=True,
        visualize=True,
        show_log=False,
        show_arrow=False
    )
    executer.run()


if __name__ == "__main__":
    main()
