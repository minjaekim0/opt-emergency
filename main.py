
import numpy as np
import pandas as pd
from opt_emergency.simulation import SimulationExecuter


def main():
    emergency_df = pd.read_csv("data/processed/emergency_df.csv")
    population_df = pd.read_csv("data/processed/community_population_df.csv")
    emergency_tk = emergency_df[
        (emergency_df["시도"].isin(["대구", "경북"])) &
        (emergency_df["시군구"] != "울릉군")
    ]
    population_tk = population_df[
        (population_df["시도"].isin(["대구", "경북"])) &
        (population_df["시군구"] != "울릉군")
    ]
    nearest_er_tk = pd.read_csv("data/processed/nearest_er_tk.csv")
    travel_time_between_er_df = pd.read_csv("data/processed/travel_time_between_er_tk.csv")
    travel_time_between_er_df = travel_time_between_er_df[
        (travel_time_between_er_df.hpid1 != "E2700553") & (travel_time_between_er_df.hpid2 != "E2700553")
    ]  # 울릉군보건의료원 제외

    def _preprocessing():
        # 응급실 병상 수 전처리
        avg_hperyn = emergency_tk.groupby("의료기관분류").hperyn.mean().round()
        for typ, avg in avg_hperyn.items():
            emergency_tk.loc[emergency_tk["의료기관분류"] == typ, "hperyn"] = \
                emergency_tk.loc[emergency_tk["의료기관분류"] == typ, "hperyn"].fillna(avg).replace(0, avg)
    _preprocessing()

    population_data = {
        "gyeongbuk": population_tk[population_tk["시도"] == "경북"]["인구수"].sum(), 
        "daegu": population_tk[population_tk["시도"] == "대구"]["인구수"].sum()
    }
    ktas_level_data = {
        "gyeongbuk": np.array([5.3, 43.0, 51.5]),
        "daegu": np.array([11.2, 60.5, 28.3])
    }
    occupancy_duration_df = pd.read_csv("data/processed/occupancy_duration_ktas_corr.csv")

    # 경북에서 발생한 ktas=12 환자가 대구로 바로 이송됐을 가능성 존재
    # -> ktas_level, occupancy_duration은 인구수로 가중평균해서 사용
    ktas_level_distribution = np.sum(
        [ktas_level_data[k] * population_data[k] for k, v in population_data.items()], axis=0
    ) / sum(population_data.values())

    occupancy_duration_distribution = dict()
    for ktas in occupancy_duration_df["ktas"].drop_duplicates():
        occupancy_duration_weighted_sum = np.zeros(7)

        for sido, sido_kor in zip(["gyeongbuk", "daegu"], ["경북", "대구"]):
            occ_dur = occupancy_duration_df[
                (occupancy_duration_df["시도"] == sido_kor) & (occupancy_duration_df["ktas"] == ktas)
            ].to_numpy()[0][2:].astype(float)
            occupancy_duration_weighted_sum += occ_dur * population_data[sido]

        occupancy_duration_distribution[str(ktas)] = \
            occupancy_duration_weighted_sum / occupancy_duration_weighted_sum.sum() * 100
            
    DURATION = 30
    PATIENT_CNT_ONE_MONTH = 38000
    PATIENT_CNT = int(PATIENT_CNT_ONE_MONTH / 30 * DURATION)  # 38000 / 1month

    executer = SimulationExecuter(
        duration=DURATION, 
        patient_cnt=PATIENT_CNT,
        ktas_level_distribution=ktas_level_distribution,
        occupancy_duration_distribution=occupancy_duration_distribution,
        pop_df=population_tk,
        er_df=emergency_tk,
        nearest_er_from_region_df=nearest_er_tk,
        travel_time_between_er_df=travel_time_between_er_df,
        visualize=False,
        show_log=False,
        show_arrow=False
    )
    executer.run()


if __name__ == "__main__":
    main()