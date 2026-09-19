import numpy as np


ODOM_V_NOISE_PROPORTIONAL = 0.03
ODOM_OMEGA_NOISE_PROPORTIONAL = 0.15

ODOM_V_NOISE_FLOOR = 0.015
ODOM_OMEGA_NOISE_FLOOR = 0.05


class OdometrySensor:

    def __init__(self, seed: int = None):
        self._rng = np.random.default_rng(seed)

    def sense(self, v_true: float, omega_true: float) -> tuple:
        v_std = ODOM_V_NOISE_PROPORTIONAL * abs(v_true) + ODOM_V_NOISE_FLOOR
        omega_std = (ODOM_OMEGA_NOISE_PROPORTIONAL * abs(omega_true) +
                     ODOM_OMEGA_NOISE_FLOOR)

        v_measured = v_true + self._rng.normal(0.0, v_std)
        omega_measured = omega_true + self._rng.normal(0.0, omega_std)
        return v_measured, omega_measured
