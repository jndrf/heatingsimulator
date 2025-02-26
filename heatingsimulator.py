#!/usr/bin/env python3

"""
Tool to backtest heat pump performance on real temperature data
"""

import numpy as np
import pandas as pd
from collections.abc import Callable
from collections import namedtuple
from operator import attrgetter
from pathlib import Path

from scipy.interpolate import NearestNDInterpolator

PowerData = namedtuple(
    'PowerData',
    [
        'cop',  # Coefficient of Performance
        't_water',  # water temperature on hot side in °C
        't_air',  # outside air temperature in °C
    ],
)


class Heatload:
    def __init__(self, heatload: float, reftemp: float, cutoff: float = 15):
        """Heatload in kW at temperature reftemp in °C"""
        self._slope = heatload / (cutoff - reftemp)
        self._cutoff = cutoff

    def __call__(self, temp: float | pd.Series) -> float | pd.Series:
        """heatload in kW for a given temp in °C"""
        if type(temp) is pd.Series:
            ret = temp.where(temp < self._cutoff, self._cutoff)
            ret = self._slope * (self._cutoff - temp)
            return ret

        else:  # python scalar
            if temp >= self._cutoff:
                return 0
            return self._slope * (self._cutoff - temp)


class Heatpump:
    @staticmethod
    def theoretical_cop(t_air: float, t_water: float) -> float:
        """return theoretical COP for given air and water temperatures in °C"""
        return (t_water + 273) / (t_water - t_air)  # need to convert to Kelvin

    def _create_cop_ratio_grid(self, powerdata: list[PowerData]) -> Callable[[float, float], float]:
        """Create an interpolation function for actual over theoretical COP"""
        powerdata.sort(key=attrgetter('t_water', 't_air'))
        actual_cops = np.array([x.cop for x in powerdata])
        theor_cops = np.array([self.theoretical_cop(x.t_air, x.t_water) for x in powerdata])

        cop_ratio = actual_cops / theor_cops

        interpolator = NearestNDInterpolator(
            x=([x.t_air for x in powerdata], [x.t_water for x in powerdata]), y=cop_ratio
        )

        def interpolation_func(t_air: float, t_water: float) -> float:
            """Return the ratio of actual of theoretical COP for a given working point"""
            ret = interpolator((t_air, t_water))  # returns zero-dim np.array
            return ret

        return interpolation_func

    def __init__(self, powerdata: list[PowerData]):
        self.cop_ratio = self._create_cop_ratio_grid(powerdata)

    def electric_power_from_heat(self, heat: float, t_water: float, t_air: float):
        """calculate electric power in kW for a given heat output in kW"""
        real_cop = self.theoretical_cop(t_air, t_water) * self.cop_ratio(t_air, t_water)
        return heat / real_cop


def read_temperature_data(file: Path) -> pd.DataFrame:
    df = pd.read_csv(
        file, sep=';', header=0, usecols=['MESS_DATUM', 'TT_TU'], dtype={'TT_TU': float}
    )
    df['date'] = pd.to_datetime(df['MESS_DATUM'], format='%Y%m%d%H')
    df['timespan'] = df['date'].diff(1)
    df['hours'] = df['timespan'] / pd.Timedelta(hours=1)
    df.set_index('date')

    return df


if __name__ == '__main__':
    pdlist = [
        PowerData(4, 35, 2),
        PowerData(5.1, 35, 7),
        PowerData(3.2, 35, -7),
        PowerData(2.2, 55, -7),
    ]
    hp = Heatpump(pdlist)

    load = Heatload(10, -10)

    # air_temps = np.linspace(15, -15, 7)
    air_temps = [7, 2, -7]
    electric = [hp.electric_power_from_heat(load(x), 35, x) for x in air_temps]

    for temp, power in zip(air_temps, electric):
        print(
            f'{temp: >5} °C: {power:.1f} kW electric for {load(temp):.1f} kW thermal: a COP of {load(temp) / power:.1f}'
        )

    df = read_temperature_data('data/produkt_tu_stunde_19490101_20231231_01975.txt')
    df['heat_required'] = load(df['TT_TU'])
    df['electricity'] = (
        hp.electric_power_from_heat(df['heat_required'], 35, df['TT_TU']) * df['hours']
    )

    print(
        f'heat required: {df["heat_required"].sum():.0f} from {df["electricity"].sum():.0f} kW of electricity for a COP of {df["heat_required"].sum() / df["electricity"].sum():.2f}'
    )
