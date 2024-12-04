from typing import Tuple, List, Dict, Union
import requests
import pandas as pd
import xmltodict
from datetime import datetime
import math

import sys
import os
sys.path.append(os.path.abspath(".."))
from private import PRIVATE

if __name__ == "__main__":
    from utils import calculate_distance
else:
    from .utils import calculate_distance


class NaverMap: 
    def _get_address_info(self, address: str) -> dict:
        """Geocoding api"""
        
        url = "https://naveropenapi.apigw.ntruss.com/map-geocode/v2/geocode"
        headers = {
            "X-NCP-APIGW-API-KEY-ID": PRIVATE.NCP_CLIENT_ID,
            "X-NCP-APIGW-API-KEY": PRIVATE.NCP_CLIENT_SECRET
        }
        params = {"query": address}
        response = requests.get(url, headers=headers, params=params)
        address_info = response.json()
        return address_info
    
    def _search_map(self, query: str) -> dict:
        """search api"""

        url = "https://openapi.naver.com/v1/search/local.json"
        headers = {
            "X-Naver-Client-Id": PRIVATE.NDEV_CLIENT_ID,
            "X-Naver-Client-Secret": PRIVATE.NDEV_CLIENT_SECRET
        }
        params = {"query": query}
        response = requests.get(url, headers=headers, params=params)
        out = response.json()
        if out["items"]:
            return out["items"][0]
        else:
            print(f"Cannot find query: {query}")
            return dict()
    
    def get_lon_lat(self, address: str) -> Tuple[float, float]:
        address_info = self._get_address_info(address=address)
        if address_info["addresses"]:
            lon = float(address_info["addresses"][0]["x"])
            lat = float(address_info["addresses"][0]["y"])
            return lon, lat
        else:
            print(f"Cannot find lon and lat of: {address}")
            return None, None
    
    def get_address(self, name: str) -> str:
        out = self._search_map(query=name)
        if out:
            return out["roadAddress"]
        else:
            return None
    
    def get_nearest_goal(self, pos_start: Tuple[float], pos_top3: List[Tuple[float, float]]):
        start = f"{pos_start[0]},{pos_start[1]}"
        goals = ":".join([f"{lon},{lat}" for lon, lat in pos_top3])
        summary = self._get_directions5_api_summary(start, goals)

        nearest_pos = summary["goal"]["location"]  # 소수점 7자리 절사됨
        nearest_idx = self._min_distance_idx(nearest_pos, pos_top3)

        travel_time_ms = summary["duration"]
        travel_time_minute = round(travel_time_ms / 1000 / 60, 4)

        return {"nearest_index": nearest_idx, "travel_time": travel_time_minute}

    def _min_distance_idx(self, pos, pos_cand):
        distance_squared = [
            (lon - pos[0]) ** 2 + (lat - pos[1]) ** 2 
            for lon, lat in pos_cand
        ]
        nearest_idx = distance_squared.index(min(distance_squared))
        return nearest_idx
    
    def get_travel_times(self, pos_start: Tuple[float], pos_goal_cand: List[Tuple[float, float]]):
        travel_times = list()
        for pos_goal in pos_goal_cand:
            t = self.get_travel_time(pos_start, pos_goal)
            travel_times.append(t)
        return travel_times
    
    def get_travel_time(self, pos_start, pos_goal):
        start = f"{pos_start[0]},{pos_start[1]}"
        goal = f"{pos_goal[0]},{pos_goal[1]}"
        summary = self._get_directions5_api_summary(start, goal)
        if summary is not None:
            travel_time_ms = summary["duration"]
            travel_time_minute = round(travel_time_ms / 1000 / 60, 4)
        else:
            travel_time_minute = 0
        return travel_time_minute
    
    def _get_directions5_api_summary(self, start, goals) -> Union[Dict, None]:
        url = "https://naveropenapi.apigw.ntruss.com/map-direction/v1/driving"
        headers = {
            "X-NCP-APIGW-API-KEY-ID": PRIVATE.NCP_CLIENT_ID,
            "X-NCP-APIGW-API-KEY": PRIVATE.NCP_CLIENT_SECRET
        }
        query = {
            "start": start, 
            "goal": goals, 
            "option": "trafast"  # trafast: 실시간 빠른 길 옵션
        }
        response = requests.get(url, headers=headers, params=query)

        try_cnt = 5
        while try_cnt > 0:
            if "route" in response.json().keys():
                summary = response.json()["route"]["trafast"][0]["summary"]
                return summary
            try_cnt -= 1
            print(start, goals)
        return None


class PublicDataPortal:
    def get_er_list_at_region(self, sido: str, sigungu: str) -> List[dict]:
        """https://www.data.go.kr/data/15000563/openapi.do"""

        url = "http://apis.data.go.kr/B552657/ErmctInfoInqireService/getEgytListInfoInqire"
        params = {
            "serviceKey": PRIVATE.PUBLIC_DATA_PORTAL_KEY,
            "Q0": sido, 
            "Q1": sigungu,
        }
        response = requests.get(url, params=params)
        er_xml = response.content
        er_list = xmltodict.parse(er_xml)["response"]["body"]["items"]["item"]
        if type(er_list) == dict: # 검색결과 1개
            er_list = [er_list]
        return er_list
    
    def get_er_info(self, er_hpid: str) -> dict:
        """https://www.data.go.kr/data/15000563/openapi.do"""

        url = "http://apis.data.go.kr/B552657/ErmctInfoInqireService/getEgytBassInfoInqire"
        params = {
            "serviceKey": PRIVATE.PUBLIC_DATA_PORTAL_KEY,
            "HPID": er_hpid,
        }
        response = requests.get(url, params=params)
        er_xml = response.content
        er_info = xmltodict.parse(er_xml)["response"]["body"]["items"]["item"]
        return er_info


class NearestEr:
    def __init__(self, er_df) -> None:
        assert {"경도", "위도"}.issubset(set(er_df.columns))
        self.er_df = er_df
        self.nm = NaverMap()

    def get_topN_straight_distance_er_address(self, lon, lat, N) -> pd.DataFrame:
        straight_distance = self.er_df.apply(
            lambda row: calculate_distance(lon, lat, row["경도"], row["위도"]), axis=1)
        if min(straight_distance) == 0:  # 자기자신 제외
            topN_idx = straight_distance.nsmallest(N+1).iloc[1:].index
        else:
            topN_idx = straight_distance.nsmallest(N).index
        topN_df = self.er_df.loc[topN_idx].reset_index(drop=True)
        topN_df = topN_df[["hpid", "기관명", "주소", "경도", "위도"]]
        topN_df["직선거리"] = straight_distance.loc[topN_idx].tolist()
        return topN_df
    
    def get_nearest_er_info(self, lon: float, lat: float) -> dict:
        start = (lon, lat)

        top3_df = self.get_topN_straight_distance_er_address(lon, lat, 3)
        pos_top3 = [
            (getattr(row, "경도"), getattr(row, "위도")) 
            for row in top3_df[["경도", "위도"]].itertuples()
        ]
        nearest_result = self.nm.get_nearest_goal(start, pos_top3)
        nearest_er_info = top3_df.iloc[nearest_result["nearest_index"]].to_dict()
        
        nearest_er_info["소요시간"] = nearest_result["travel_time"]
        nearest_er_info["측정시각"] = datetime.now().strftime("%Y%m%d_%H%M%S")
        return nearest_er_info
    
    def get_nearest_er_order_info(self, lon: float, lat: float, N: int) -> pd.DataFrame:
        N2 = math.ceil(N * 1.5)
        start = (lon, lat)

        topN2_df = self.get_topN_straight_distance_er_address(lon, lat, N2)
        pos_topN2 = [
            (getattr(row, "경도"), getattr(row, "위도")) 
            for row in topN2_df[["경도", "위도"]].itertuples()
        ]
        travel_times = self.nm.get_travel_times(start, pos_topN2)
        nearest_order_idx = sorted(range(len(travel_times)), key=lambda i: travel_times[i])
        topN2_df["소요시간"] = travel_times
        topN2_df["측정시각"] = datetime.now().strftime("%Y%m%d_%H%M%S")
        topN2_df = topN2_df.iloc[nearest_order_idx].reset_index(drop=True)
        return topN2_df.iloc[:N]
    

if __name__ == "__main__":
    emergency_df = pd.read_csv("../data/processed/emergency_df.csv")
    emergency_df_tk = emergency_df[emergency_df["시도"].isin(["대구", "경북"])]

    nm = NaverMap()
    hpids = sorted(emergency_df_tk.hpid.tolist())
    travel_times = []
    for hpid1 in hpids:
        for hpid2 in [x for x in hpids if x != hpid1]:
            pos_start = emergency_df_tk.loc[emergency_df_tk.hpid == hpid1, ["경도", "위도"]].values[0]
            pos_goal = emergency_df_tk.loc[emergency_df_tk.hpid == hpid2, ["경도", "위도"]].values[0]
            tt = nm.get_travel_time(pos_start, pos_goal)
            travel_times.append([hpid1, hpid2, tt])
    print(0)