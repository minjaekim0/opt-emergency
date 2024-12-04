import optuna
import numpy as np
import pandas as pd
import random
import pickle
from datetime import datetime
import shutil

import sys
import os
sys.path.append(os.path.abspath(".."))
from src.preprocessing import RegionFilter, bed_num_imputation, WeightedAverageDistribution
from src.simulation import SimulationExecuter
from src.utils import find_project_root, Logger


class GeneticAlgorithmOptimizer:
    def __init__(
        self, 
        decision_hpids, 
        duration, 
        patient_cnt, 
        sim_target_sido,
        opt_target_sido,
        ktas_level_distribution,
        occupancy_duration_distribution,
        pop_df,
        er_df,
        nearest_er_from_region_df, 
        travel_time_between_er_df,
        num_sim_iter,
        num_opt_iter,
        population_size,
        opt_mode,
        study_name,
    ):        
        self.simulation_kwargs_common = {
            "duration": duration, 
            "patient_cnt": patient_cnt, 
            "target_sido": sim_target_sido,
            "ktas_level_distribution": ktas_level_distribution, 
            "occupancy_duration_distribution": occupancy_duration_distribution, 
            "pop_df": pop_df, 
            "nearest_er_from_region_df": nearest_er_from_region_df, 
            "travel_time_between_er_df": travel_time_between_er_df,
            "additional_indicators": False,
            "visualize": False, 
            "show_arrow": False,
            "show_log": False, 
            "show_bar": True, 
            "save_data": False, 
        }
        self.opt_target_sido = opt_target_sido
        self.er_df = er_df
        self.num_sim_iter = num_sim_iter
        self.num_opt_iter = num_opt_iter
        self.population_size = population_size
        assert opt_mode in ["min_cost", "max_tri"]
        self.opt_mode = opt_mode
        self.study_name = study_name

        self.time_to_start_treatment_orig = None
        self.best_trial_num = None

        self.decision_hpids = decision_hpids

        self.sql_storage = f"sqlite:///../optuna_output/{study_name}/sql_storage.db"
        PROJECT_ROOT_PATH = find_project_root()
        self.output_dir = f"{PROJECT_ROOT_PATH}/optuna_output/{study_name}/"
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self):
        sampler = optuna.samplers.NSGAIISampler(constraints_func=self._constraint)

        if self.opt_mode == "min_cost":
            direction = "minimize"
        else:
            direction = "maximize"
        self.study = optuna.create_study(
            study_name="study", 
            storage=self.sql_storage, 
            load_if_exists=True,
            direction=direction,
            sampler=sampler
        )

        completed_trials = len([t for t in self.study.trials if t.datetime_complete is not None])

        if 0 < completed_trials < self.num_opt_iter:
            raise Exception("Do not re-load, please delete db and re-run this code.")

        if completed_trials == 0:
            init_populations = self._init_populations()
            for pop in init_populations:
                self.study.enqueue_trial(pop)
        
        self.study.optimize(self._objective, n_trials=self.num_opt_iter)

        self._save_result()

    def _save_result(self):
        with open(f"{self.output_dir}/study.pickle", "wb") as f:
            pickle.dump(self.study, f)

    def _init_populations(self):
        if self.opt_mode == "min_cost":
            return self._init_populations_min_cost()
        else:
            return self._init_populations_max_tri()
    
    def _init_populations_min_cost(self):
        init_variation = 3
        populations = list()

        orig = {hpid: 0 for hpid in self.decision_hpids}
        populations.append(orig)
        
        # for i in range(len(self.decision_hpids)):
        #     deltas = [0 for _ in range(len(self.decision_hpids))]
        #     deltas[i] = -init_variation
        #     population = {hpid: d for hpid, d in zip(self.decision_hpids, deltas)}
        #     populations.append(population)

        # for i in range(len(self.decision_hpids)):
        #     selected = np.random.randint(0, len(self.decision_hpids), size=3)
        #     deltas = np.random.randint(-init_variation, 0, size=len(self.decision_hpids))
        #     deltas = [int(d) if j in selected else 0 for j, d in enumerate(deltas)]
        #     population = {hpid: d for hpid, d in zip(self.decision_hpids, deltas)}
        #     populations.append(population)

        return populations
    
    def _init_populations_max_tri(self):
        init_variation = 3
        populations = list()

        orig = {hpid: 0 for hpid in self.decision_hpids}
        populations.append(orig)

        return populations
    
    def _objective(self, trial: optuna.trial.Trial):
        delta_num_beds = list()
        for hpid in self.er_df["hpid"]:
            if hpid in self.decision_hpids:
                num_beds = self.er_df.loc[self.er_df["hpid"] == hpid, "hperyn"].values[0]
                if self.opt_mode == "min_cost":
                    delta_num_beds.append(trial.suggest_int(hpid, max(-10, -num_beds), 0))
                else:
                    upper = min(5, num_beds)
                    delta_num_beds.append(trial.suggest_int(hpid, -upper, upper))
            else:
                delta_num_beds.append(0)
        er_df_copy = self.er_df.copy()
        er_df_copy["hperyn"] = er_df_copy["hperyn"] + delta_num_beds
        tri_values = self._tri_values_from_simulation(er_df_copy["hperyn"])
        tri_avg = float(np.mean(tri_values))
        trial.set_user_attr("tri", tri_avg)
        
        num_beds = int(sum(er_df_copy["hperyn"]))
        trial.set_user_attr("num_beds", num_beds)

        num_beds_orig = self.study.trials[0].user_attrs["num_beds"]
        tri_orig = self.study.trials[0].user_attrs["tri"]

        if self.opt_mode == "min_cost":
            # 추후 "각 병원의 점유율 일정 수준 이하" constraint로 추가
            constraint_buffer = 0.001
            constraint = [tri_orig - tri_avg - constraint_buffer]
            objective = num_beds
            print(f"objective: num_beds = {objective}")
            print(f"constraint: -Δ(tri) - {constraint_buffer} = {constraint[0]:+f} -> {constraint[0] <= 0}")
        else:
            constraint_buffer = 5
            constraint = [num_beds - num_beds_orig - constraint_buffer]
            objective = tri_avg
            print(f"objective: tri = {objective}")
            print(f"constraint: Δ(num_beds) - {constraint_buffer} = {constraint[0]:+d} -> {constraint[0] <= 0}")

        print(f"tri_values = {tri_values}")
        print(f"num_beds = {num_beds}")
            
        trial.set_user_attr("constraint", constraint)
        return objective

    def _tri_values_from_simulation(self, num_beds_values):
        er_df_copy = self.er_df.copy()
        er_df_copy["hperyn"] = num_beds_values
        simulation_kwargs = self.simulation_kwargs_common
        simulation_kwargs["er_df"] = er_df_copy

        tri_values = list()
        for seed in range(self.num_sim_iter):
            simulation_kwargs["random_seed"] = seed
            sim = SimulationExecuter(**simulation_kwargs)
            sim.run()

            time_to_start_treatment_data = list()
            for p in sim.patient_trackers.values():
                if p.started_treatment is True and p.occured_region.sido == self.opt_target_sido:
                    time_to_start_treatment_data.append(
                        (p.start_treatment_time - p.occurred_time) * (24 * 60)
                    )
            time_to_start_treatment_data = np.array(time_to_start_treatment_data)
            tri = sum(time_to_start_treatment_data <= 30) / len(time_to_start_treatment_data)
            tri_values.append(float(tri))
        return tri_values

    @staticmethod
    def _constraint(trial):
        return trial.user_attrs["constraint"]


def main():
    emergency_df = pd.read_csv("../data/processed/emergency_df.csv")
    population_df = pd.read_csv("../data/processed/community_population_df.csv")
    nearest_er_df = pd.read_csv("../data/processed/nearest_er_tk_daytime.csv")
    travel_time_between_er_df = pd.read_csv("../data/processed/travel_time_between_er_tk_daytime.csv")
    ktas_level_ratio_df = pd.read_csv("../data/processed/ktas_level_ratio.csv").astype({"ktas": str})
    occupancy_duration_df = pd.read_csv("../data/processed/occupancy_duration_ktas_corr.csv")
    sim_target_sido = ["대구", "경북"]
    opt_target_sido = "경북"
    sigungu_excluded = ["울릉군"]
    
    rf = RegionFilter(emergency_df, population_df, travel_time_between_er_df, sim_target_sido, sigungu_excluded)
    emergency_df = rf.emergency_df_filtering()
    population_df = rf.population_df_filtering()
    travel_time_between_er_df = rf.travel_time_between_er_df_filtering()
    bed_num_imputation(emergency_df)
    wa = WeightedAverageDistribution(ktas_level_ratio_df, occupancy_duration_df, population_df, sim_target_sido)
    ktas_level_distribution = wa.ktas_level_ratio()
    occupancy_duration_distribution = wa.occupancy_duration()

    DURATION = 7
    PATIENT_CNT_ONE_MONTH = 38000
    PATIENT_CNT = int(PATIENT_CNT_ONE_MONTH / 30 * DURATION)  # 38000 / 1month
    
    decision_hpids = [
        'A2700014', 'A2700007', 'A2700016', 'A2700003', 'A2700013', 'A2700004', 'A2700012', 'A2700006', 'A2700002', 'A2700018', 
        'A2700001', 'A2700005', 'A2700015', 'A2700011', 'A2700008', 'A2700017', 'A2702752', 'A2700019', 'A2700009', 'A2700036', 
        'A2702563', 'A2700031', 'A2700052', 'A2700038', 'A2700023', 'A2700070', 'A2700063', 'E2700553', 'E2700554', 'A2700097', 
        'A2700071', 'A2700030', 'A2700020', 'A2700026', 'A2700066'
    ]
    # decision_hpids = emergency_df["hpid"].tolist()
    ga = GeneticAlgorithmOptimizer(
        decision_hpids=decision_hpids,
        duration=DURATION, 
        patient_cnt=PATIENT_CNT,
        sim_target_sido=sim_target_sido,
        opt_target_sido=opt_target_sido,
        ktas_level_distribution=ktas_level_distribution,
        occupancy_duration_distribution=occupancy_duration_distribution,
        pop_df=population_df,
        er_df=emergency_df,
        nearest_er_from_region_df=nearest_er_df,
        travel_time_between_er_df=travel_time_between_er_df,
        num_sim_iter=5,
        num_opt_iter=1000,
        population_size=50,
        opt_mode="min_cost",
        study_name="min_cost__gyeongbuk_ers__no_init",
    )
    ga.run() 


if __name__ == "__main__":
    main()
