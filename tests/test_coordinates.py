import numpy as np
import pandas as pd
from math import pi


def geoloc(self, attitude: np.ndarray, coords: np.ndarray):
    """geoloc Processes input data and performs geolocalization

    Parameters
    ----------
    attitude : np.ndarray
        Array of shape (N, 7), namely:
            0 <- Latitude [-pi/2, pi/2]
            1 <- Longitude [-pi, pi]
            2 <- Altitude
            3 <- Roll
            4 <- Pitch
            5 <- Yaw
            6 <- Heading
            7 <- Vground
    coords : np.ndarray
        Array of shape (N, 2), namely:
            0 <- x offset w.r.t camera center
            1 <- y offset w.r.t camera center
            both in the reference frame of the plane's body

    Returns
    ----------
    critical_indices : np.ndarray
        Array of indices of the critical points
    mask : np.ndarray
        Array of boolean values, True if the point is valid, False otherwise

    Usage
    ----------
    >>> # definition of attitude and coords
    >>> indices, mask = ds.geoloc(attitude, coords)
    >>> nn = nn[mask]  # filter the neural network output
    """

    # Use asserts to check
    assert (
        len(attitude.shape) == 2
    ), f"Log non valido > log passato {len(attitude.shape)}-D, atteso 2-D"
    assert (
        attitude.shape[1] == 8
    ), f"Log non valido > log passato {attitude.shape[1]} colonne, atteso 8 colonne"
    assert (
        attitude.dtype == float
    ), f"Log non valido > log passato {attitude.dtype}, atteso float"
    assert (
        len(coords.shape) == 2
    ), f"Coordinate non valide > coordinate passate {len(coords.shape)}-D, atteso 2-D"
    assert (
        coords.shape[1] == 2
    ), f"Coordinate non valide > coordinate passate {coords.shape[1]} colonne, atteso 2 colonne"
    assert (
        coords.dtype == float
    ), f"Coordinate non valide > coordinate passate {coords.dtype}, atteso float"
    assert (
        coords.shape[0] == attitude.shape[0]
    ), f"Coordinate non valide > coordinate passate {coords.shape[0]} righe, atteso {attitude.shape[0]} righe"
    assert not np.any(
        attitude[:, 2] < 0
    ), f"Altitudine non valida, agli indici {np.where(attitude[:,0]<0)}, valori {attitude[attitude[:,0]<0]}"
    assert not np.any(attitude[:, 0] < -pi / 2) and not np.any(
        attitude[:, 0] > pi / 2
    ), f"Latitudine non valida, agli indici {np.where(attitude[:,0]<-pi/2 | attitude[:,0]>pi/2)}, valori {attitude[attitude[:,0]<-pi/2 | attitude[:,0]>pi/2]}"
    assert not np.any(attitude[:, 2] < -pi) and not np.any(
        attitude[:, 1] > pi
    ), f"Longitudine non valida, agli indici {np.where(attitude[:,1]<-pi | attitude[:,2]>pi)}, valori {attitude[attitude[:,1]<-pi | attitude[:,1]>pi]}"

    N = attitude.shape[0]

    # scrivo su geoloc_df i dati in input
    # data concatena attitude e coords orizzontalmente
    data = np.hstack((attitude, coords))
    geoloc_row = pd.DataFrame(data, columns=self.geoloc_df.columns[:10])

    # geolocalizzazione con errore
    q_samples = (
        np.random.randn(
            N, self.configs.config_dict["decision_system"]["geoloc_monte_carlo"]
        )
        * self.configs.config_dict["decision_system"]["altitude_error"]
        + attitude[:, 2][:, None]
    )
    angles_samples = (
        np.random.randn(
            N, self.configs.config_dict["decision_system"]["geoloc_monte_carlo"], 3
        )
        * np.array(self.configs.config_dict["decision_system"]["rpy_error"])[
            None, None, :
        ]
        + attitude[:, 3:6][:, None, :]
    )

    phi_uav = angles_samples[:, :, 0]
    theta_uav = angles_samples[:, :, 1]
    psi_uav = angles_samples[:, :, 2]
    ahrs2ned = np.array(
        [
            [
                np.cos(theta_uav) * np.cos(psi_uav),
                np.sin(phi_uav) * np.sin(theta_uav) * np.cos(psi_uav)
                - np.cos(phi_uav) * np.sin(psi_uav),
                np.cos(phi_uav) * np.sin(theta_uav) * np.cos(psi_uav)
                + np.sin(phi_uav) * np.sin(psi_uav),
            ],
            [
                np.cos(theta_uav) * np.sin(psi_uav),
                np.sin(phi_uav) * np.sin(theta_uav) * np.sin(psi_uav)
                + np.cos(phi_uav) * np.cos(psi_uav),
                np.cos(phi_uav) * np.sin(theta_uav) * np.sin(psi_uav)
                - np.sin(phi_uav) * np.cos(psi_uav),
            ],
            [
                -np.sin(theta_uav),
                np.sin(phi_uav) * np.cos(theta_uav),
                np.cos(phi_uav) * np.cos(theta_uav),
            ],
        ]
    ).reshape(
        3, 3, N, self.configs.config_dict["decision_system"]["geoloc_monte_carlo"]
    )
    P_Fcam = np.hstack(
        (
            coords,
            np.full((N, 1), self.configs.config_dict["decision_system"]["ratio"]),
        )
    )
    P_Fxyz = np.einsum(
        "rc, nc -> nr", self.cam_error_matrix, P_Fcam
    )  # coordinate di P rispetto F in coordinate corpo dell'UAV
    P_Fahrs = np.einsum(
        "rc, nc -> nr", self.ahrs_error_matrix, P_Fxyz
    )  # coordinate di P rispetto F in coordinate corpo dell'ahrs
    P_Fned = np.einsum(
        "rcni, nc -> nir", ahrs2ned, P_Fahrs
    )  # coordinate di P rispetto F in coordinate NED
    F_UAVned = np.einsum(
        "rcni, c -> nir", ahrs2ned, self.F_UAVahrs
    )  # coordinate di F rispetto l'UAV in coordinate NED
    T_Oned = (
        F_UAVned[:, :, 0:2]
        + ((q_samples - F_UAVned[:, :, 2]) / P_Fned[:, :, 2])[:, :, None]
        * P_Fned[:, :, 0:2]
    )

    mu = np.mean(T_Oned, axis=1)
    T = T_Oned - mu[:, None, :]
    cov = (
        1
        / (self.configs.config_dict["decision_system"]["geoloc_monte_carlo"] - 1)
        * np.einsum("nir, nic -> nrc", T, T)
    )

    # salvo i risultati
    mu += self.gps2ne(attitude[:, :2], self.geofence_center)
    cov = (
        cov
        + (np.eye(2) * self.configs.config_dict["decision_system"]["ne_error"] ** 2)[
            None, :, :
        ]
    )

    # correggo mu con il bias dovuto al delay
    mu = (
        mu
        + self.configs.config_dict["decision_system"]["delay"]
        * attitude[:, 7][:, None]
        * np.array([np.cos(attitude[:, 6]), np.sin(attitude[:, 6])]).T
    )

    # filtro i dati validi
    mask = (mu - self.gps2ne(self.geofence[0], self.geofence_center)) @ self.gps2ne(
        np.array(self.geofence[1]),
        self.geofence[0],
    ) >= 0
    mask &= (mu - self.gps2ne(self.geofence[1], self.geofence_center)) @ self.gps2ne(
        np.array(self.geofence[2]),
        self.geofence[1],
    ) >= 0
    mask &= (mu - self.gps2ne(self.geofence[2], self.geofence_center)) @ self.gps2ne(
        np.array(self.geofence[3]),
        self.geofence[2],
    ) >= 0

    # riporto in geoloc_df i risultati quindi modifico le ultime N righe (già costruite all'inizio di questo metodo)
    geoloc_row["mu(n)"] = mu[:, 0]
    geoloc_row["mu(e)"] = mu[:, 1]
    geoloc_row["cov(n,n)"] = cov[:, 0, 0]
    geoloc_row["cov(n,e)"] = cov[:, 0, 1]
    geoloc_row["cov(e,n)"] = cov[:, 1, 0]
    geoloc_row["cov(e,e)"] = cov[:, 1, 1]
    geoloc_row["mask"] = mask
    self.geoloc_df = pd.concat(
        [
            geoloc_row.astype(self.geoloc_df.dtypes),
            self.geoloc_df.astype(geoloc_row.dtypes),
        ],
        ignore_index=True,
    )

    critical_indices = np.where(~mask)[0]
    mu = mu[mask]
    cov = cov[mask]
    self.mu.append(mu)
    self.cov_inv.append(np.linalg.inv(cov))

    return critical_indices, mask
