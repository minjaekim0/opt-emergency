import numpy as np
import pandas as pd


class RegionFilter:
    def __init__(self, emergency_df, population_df, travel_time_between_er_df, target_sido, sigungu_excluded):
        self._emergency_df = emergency_df
        self._population_df = population_df
        self._travel_time_between_er_df = travel_time_between_er_df
        self.target_sido = target_sido
        self.sigungu_excluded = sigungu_excluded
    
    def emergency_df_filtering(self):
        df = self._emergency_df[
            (self._emergency_df["시도"].isin(self.target_sido)) & 
            (~self._emergency_df["시군구"].isin(self.sigungu_excluded))
        ]
        return df

    def population_df_filtering(self):
        df = self._population_df[
            (self._population_df["시도"].isin(self.target_sido)) & 
            (~self._population_df["시군구"].isin(self.sigungu_excluded))
        ]
        return df

    def travel_time_between_er_df_filtering(self):
        hpid_excluded = self._emergency_df[
            self._emergency_df["시군구"].isin(self.sigungu_excluded)
        ]["hpid"].tolist()
        df = self._travel_time_between_er_df[
            (~self._travel_time_between_er_df["hpid1"].isin(hpid_excluded)) & 
            (~self._travel_time_between_er_df["hpid2"].isin(hpid_excluded))
        ]
        return df


def bed_num_imputation(emergency_df):
    avg_hperyn = emergency_df.groupby("의료기관분류").hperyn.mean().round()
    for typ, avg in avg_hperyn.items():
        emergency_df.loc[emergency_df["의료기관분류"] == typ, "hperyn"] = \
            emergency_df.loc[emergency_df["의료기관분류"] == typ, "hperyn"].fillna(avg).replace(0, avg)


class WeightedAverageDistribution:
    """
    예를 들어 경북에서 발생한 ktas=12 환자가 대구로 바로 이송됐을 가능성 존재
    -> ktas_level, occupancy_duration은 인구수로 가중평균해서 사용
    """
    
    def __init__(self, ktas_level_ratio_df, occupancy_duration_df, population_df, target_sido):
        self._ktas_level_ratio_df = ktas_level_ratio_df
        self._occupancy_duration_df = occupancy_duration_df
        self._population_data = {
            sido: population_df[population_df["시도"] == sido]["인구수"].sum() for sido in target_sido
        }
        self.target_sido = target_sido
    
    def ktas_level_ratio(self):
        out = list()
        for ktas_level in ["ktas12", "ktas3", "ktas45"]:
            weighted_avg = 0
            for sido, pop in self._population_data.items():
                weighted_avg += self._ktas_level_ratio_df[
                    (self._ktas_level_ratio_df["시도"] == sido) & (self._ktas_level_ratio_df["ktas"] == ktas_level)
                ]["ratio"].iloc[0] * pop
            weighted_avg /= sum(self._population_data.values())
            out.append(weighted_avg)
        return out
    
    def occupancy_duration(self):
        out = dict()
        for ktas in self._occupancy_duration_df["ktas"].drop_duplicates():
            occupancy_duration_weighted_sum = np.zeros(7)

            for sido in self.target_sido:
                occ_dur = self._occupancy_duration_df[
                    (self._occupancy_duration_df["시도"] == sido) & (self._occupancy_duration_df["ktas"] == ktas)
                ].to_numpy()[0][2:].astype(float)
                occupancy_duration_weighted_sum += occ_dur * self._population_data[sido]

            out[ktas] = \
                occupancy_duration_weighted_sum / occupancy_duration_weighted_sum.sum() * 100
        return out


def main():
    pass


if __name__ == "__main__":
    main()