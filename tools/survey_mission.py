#!/usr/bin/env python3
"""
survey_mission.py
─────────────────
Calcola i waypoint di una survey da un poligono + camera params
e carica la missione su PX4 via pymavlink.

Testato con PX4 SITL (Gazebo). Avvia SITL prima di eseguire:
    make px4_sitl gazebo

Connessioni default PX4 SITL:
    - QGC  → udp:0.0.0.0:14550
    - Script → udp:0.0.0.0:14551  (così non confligge con QGC)
"""

import math
import time
from dataclasses import dataclass
from typing import List, Tuple

try:
    from pymavlink import mavutil
except ImportError:
    raise ImportError("Installa pymavlink: pip install pymavlink")


# ═══════════════════════════════════════════════════════════════════════════════
# Tipi
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class LatLon:
    lat: float
    lon: float


@dataclass
class CameraParams:
    sensor_width_mm: float
    sensor_height_mm: float
    image_width_px: float
    image_height_px: float
    focal_length_mm: float
    frontal_overlap: float  # 0-1
    side_overlap: float  # 0-1
    altitude_m: float
    landscape: bool = True


@dataclass
class Transect:
    """Coppia inizio/fine di una striscia di volo."""

    start: LatLon
    end: LatLon
    altitude_m: float


# ═══════════════════════════════════════════════════════════════════════════════
# CameraCalc (replica CameraCalc.cc)
# ═══════════════════════════════════════════════════════════════════════════════


def camera_calc(params: CameraParams) -> dict:
    image_density = (params.altitude_m * params.sensor_width_mm * 100.0) / (
        params.image_width_px * params.focal_length_mm
    )
    if params.landscape:
        fp_side = (params.image_width_px * image_density) / 100.0
        fp_frontal = (params.image_height_px * image_density) / 100.0
    else:
        fp_side = (params.image_height_px * image_density) / 100.0
        fp_frontal = (params.image_width_px * image_density) / 100.0

    return {
        "adjusted_footprint_side_m": fp_side * (1.0 - params.side_overlap),
        "adjusted_footprint_frontal_m": fp_frontal * (1.0 - params.frontal_overlap),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Geometria + survey grid (replica SurveyComplexItem._rebuildTransects())
# ═══════════════════════════════════════════════════════════════════════════════

EARTH_R = 6_371_000.0


def _to_xy(origin, point):
    lat0 = math.radians(origin.lat)
    x = math.radians(point.lon - origin.lon) * EARTH_R * math.cos(lat0)
    y = math.radians(point.lat - origin.lat) * EARTH_R
    return x, y


def _to_latlon(origin, x, y):
    lat0 = math.radians(origin.lat)
    return LatLon(
        origin.lat + math.degrees(y / EARTH_R),
        origin.lon + math.degrees(x / (EARTH_R * math.cos(lat0))),
    )


def _rotate(x, y, cx, cy, a):
    s, c = math.sin(a), math.cos(a)
    x -= cx
    y -= cy
    return c * x - s * y + cx, s * x + c * y + cy


def _intersections(x0, y0, x1, y1, poly):
    ts, dx, dy = [], x1 - x0, y1 - y0
    n = len(poly)
    for i in range(n):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % n]
        edx, edy = bx - ax, by - ay
        d = dx * edy - dy * edx
        if abs(d) < 1e-12:
            continue
        t = ((ax - x0) * edy - (ay - y0) * edx) / d
        u = ((ax - x0) * dy - (ay - y0) * dx) / d
        if -1e-9 <= t <= 1 + 1e-9 and -1e-9 <= u <= 1 + 1e-9:
            ts.append(max(0.0, min(1.0, t)))
    return sorted(set(round(t, 10) for t in ts))


def _in_poly(px, py, poly):
    inside, j = False, len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _clip(x0, y0, x1, y1, poly):
    ts = _intersections(x0, y0, x1, y1, poly)
    segs = []
    for i in range(0, len(ts) - 1, 2):
        ta, tb = ts[i], ts[i + 1]
        mx = x0 + (ta + tb) / 2 * (x1 - x0)
        my = y0 + (ta + tb) / 2 * (y1 - y0)
        if _in_poly(mx, my, poly):
            segs.append((x0 + ta * (x1 - x0), x0 + tb * (x1 - x0)))
    return segs


def build_transects(
    polygon: List[LatLon],
    camera: CameraParams,
    grid_angle_deg: float = 0.0,
) -> Tuple[List[Transect], dict]:
    """
    Genera i transect (inizio/fine striscia) dal poligono e dai parametri camera.
    Restituisce solo gli estremi — il trigger camera avviene in volo autonomamente.
    """
    calc = camera_calc(camera)
    step_side = calc["adjusted_footprint_side_m"]
    step_frontal = calc["adjusted_footprint_frontal_m"]

    origin = polygon[0]
    pts = [_to_xy(origin, p) for p in polygon]
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)

    a = math.radians(-grid_angle_deg)
    rot = [_rotate(x, y, cx, cy, a) for x, y in pts]

    xs = [p[0] for p in rot]
    ys = [p[1] for p in rot]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    transects = []
    y = y_min
    stripe = 0
    while y <= y_max + step_side:
        for xa, xb in _clip(x_min - 10, y, x_max + 10, y, rot):
            # Inverte il verso nelle strisce pari (serpentina)
            if stripe % 2 == 0:
                p_start = (xa, y)
                p_end = (xb, y)
            else:
                p_start = (xb, y)
                p_end = (xa, y)

            inv = math.radians(grid_angle_deg)
            sx, sy = _rotate(p_start[0], p_start[1], cx, cy, inv)
            ex, ey = _rotate(p_end[0], p_end[1], cx, cy, inv)

            transects.append(
                Transect(
                    start=_to_latlon(origin, sx, sy),
                    end=_to_latlon(origin, ex, ey),
                    altitude_m=camera.altitude_m,
                )
            )
            stripe += 1
        y += step_side

    return transects, {
        **calc,
        "n_transects": len(transects),
        "trigger_distance_m": step_frontal,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Upload missione su PX4 via pymavlink
# ═══════════════════════════════════════════════════════════════════════════════

MAV_FRAME_GLOBAL_RELATIVE_ALT = 3
MAV_FRAME_MISSION = 2  # per RTL e comandi senza coordinate


def build_mission_items(transects, trigger_dist_m, takeoff_alt=10.0):
    items = []

    def item(frame, cmd, lat, lon, alt, p1=0, p2=0, p3=0, p4=0):
        items.append(
            dict(
                frame=frame,
                command=cmd,
                param1=p1,
                param2=p2,
                param3=p3,
                param4=p4,
                lat=lat,
                lon=lon,
                alt=alt,
            )
        )

    # Takeoff — frame GLOBAL_RELATIVE_ALT, lat/lon della posizione di decollo
    item(
        MAV_FRAME_GLOBAL_RELATIVE_ALT,
        22,
        transects[0].start.lat,
        transects[0].start.lon,
        takeoff_alt,
        p1=0,
        p2=0,
        p3=0,
        p4=float("nan"),
    )

    for t in transects:
        # Inizio striscia
        item(
            MAV_FRAME_GLOBAL_RELATIVE_ALT,
            16,
            t.start.lat,
            t.start.lon,
            t.altitude_m,
            p1=0,
            p2=2,
            p3=0,
            p4=float("nan"),
        )
        # Fine striscia
        item(
            MAV_FRAME_GLOBAL_RELATIVE_ALT,
            16,
            t.end.lat,
            t.end.lon,
            t.altitude_m,
            p1=0,
            p2=2,
            p3=0,
            p4=float("nan"),
        )

    # RTL — frame MAV_FRAME_MISSION, coordinate 0
    item(MAV_FRAME_MISSION, 20, 0, 0, 0)

    return items


def upload_mission(mav, items: list, timeout: int = 30):
    """
    Carica la missione su PX4 via pymavlink.
    connection_string esempi:
        'udp:0.0.0.0:14551'   ← SITL (lascia 14550 a QGC)
        'tcp:127.0.0.1:5760'  ← MAVProxy TCP bridge
        '/dev/ttyACM0'        ← USB seriale
    """

    total = len(items)
    print(f"[→] Upload missione: {total} item...")

    # Invia MISSION_COUNT
    mav.mav.mission_count_send(
        mav.target_system,
        mav.target_component,
        total,
        mavutil.mavlink.MAV_MISSION_TYPE_MISSION,
    )

    # Risponde ai MISSION_REQUEST_INT uno alla volta
    uploaded = 0
    t_start = time.time()
    while uploaded < total:
        if time.time() - t_start > timeout:
            raise TimeoutError("Timeout durante l'upload della missione")

        msg = mav.recv_match(
            type=["MISSION_REQUEST_INT", "MISSION_REQUEST"], blocking=True, timeout=5
        )
        if msg is None:
            continue

        seq = msg.seq
        if seq >= total:
            break

        it = items[seq]
        mav.mav.mission_item_int_send(
            mav.target_system,
            mav.target_component,
            seq,
            it["frame"],
            it["command"],
            1 if seq == 0 else 0,  # current (primo item = current)
            1,  # autocontinue
            it["param1"],
            it["param2"],
            it["param3"],
            it["param4"],
            int(it["lat"] * 1e7),
            int(it["lon"] * 1e7),
            it["alt"],
            mavutil.mavlink.MAV_MISSION_TYPE_MISSION,
        )
        print(
            f"    [{seq+1}/{total}] cmd={it['command']} lat={it['lat']:.5f} lon={it['lon']:.5f}"
        )
        uploaded += 1

    # Aspetta MISSION_ACK
    ack = mav.recv_match(type="MISSION_ACK", blocking=True, timeout=timeout)
    if ack is None:
        raise TimeoutError("Nessun MISSION_ACK ricevuto")
    if ack.type != mavutil.mavlink.MAV_MISSION_ACCEPTED:
        raise RuntimeError(f"Mission upload fallito: MISSION_ACK type={ack.type}")

    print(f"[✓] Missione caricata con successo ({total} item)")
    return mav


def start_mission(mav, trigger_dist_m):
    # Arma
    mav.mav.command_long_send(
        mav.target_system, mav.target_component, 400, 0, 1, 0, 0, 0, 0, 0, 0
    )
    time.sleep(2)

    # Imposta trigger camera
    mav.mav.command_long_send(
        mav.target_system,
        mav.target_component,
        206,
        0,
        trigger_dist_m,
        0,
        1,
        0,
        0,
        0,
        0,
    )

    # Avvia missione
    mav.mav.command_long_send(
        mav.target_system, mav.target_component, 300, 0, 0, 0, 0, 0, 0, 0, 0
    )


def request_mission_list(mav):
    """
    Richiede la lista missione a PX4.
    QGC in ascolto vede i MISSION_ITEM_INT passare e aggiorna la sua view.
    """
    mav.mav.mission_request_list_send(
        mav.target_system,
        mav.target_component,
        mavutil.mavlink.MAV_MISSION_TYPE_MISSION,
    )
    # Aspetta MISSION_COUNT come conferma
    msg = mav.recv_match(type="MISSION_COUNT", blocking=True, timeout=5)
    if msg:
        print(f"[✓] PX4 conferma: {msg.count} item in missione")


def set_camera_trigger(mav, distance_m: float):
    """
    Attiva il trigger camera a distanza fissa.
    distance_m=0 per disabilitare.

    MAV_CMD_DO_SET_CAM_TRIGG_DIST (206):
        param1 = distanza in metri (0 = stop)
        param2 = shutter integration time (0 = ignore)
        param3 = 1 = trigger un singolo scatto immediato, 0 = no
    """
    mav.mav.command_long_send(
        mav.target_system,
        mav.target_component,
        206,  # MAV_CMD_DO_SET_CAM_TRIGG_DIST
        0,  # confirmation
        distance_m,  # param1: distanza trigger in metri
        0,  # param2: shutter time (ignore)
        1,  # param3: trigger subito uno scatto
        0,
        0,
        0,
        0,  # param4-7 non usati
    )
    print(f"[📷] Camera trigger impostato: ogni {distance_m}m")

    # Aspetta ACK da PX4
    ack = mav.recv_match(type="COMMAND_ACK", blocking=True, timeout=3)
    if ack and ack.command == 206:
        if ack.result == 0:
            print("[✓] Camera trigger confermato da PX4")
        else:
            print(f"[!] PX4 ha risposto con result={ack.result}")


def monitor_and_trigger(mav, mission_items, trigger_dist_m):
    #                        ↑ passala come parametro
    print("[i] In ascolto eventi missione...")
    while True:
        msg = mav.recv_match(
            type=["MISSION_ITEM_REACHED", "MISSION_CURRENT", "CAMERA_TRIGGER"],
            blocking=True,
            timeout=60,
        )
        if msg is None:
            break

        if msg.get_type() == "MISSION_ITEM_REACHED":
            print(f"[→] Waypoint {msg.seq} raggiunto")
            if msg.seq == 1:
                set_camera_trigger(mav, trigger_dist_m)

        elif msg.get_type() == "CAMERA_TRIGGER":
            print(f"[📷] Scatto #{msg.seq}")

        elif msg.get_type() == "MISSION_CURRENT":
            if msg.seq == len(mission_items) - 1:
                set_camera_trigger(mav, 0)
                break


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # ── Poligono di survey ────────────────────────────────────────────────────
    # Modifica con le coordinate reali della tua area
    polygon = [
        LatLon(35.061623, -118.157026),
        LatLon(35.059337, -118.158008),
        LatLon(35.061589, -118.160843),
        LatLon(35.059404, -118.163795),
    ]

    # ── Parametri camera ─────────────────────────────────────────────────────
    # Sony RX100 come esempio
    camera = CameraParams(
        sensor_width_mm=14.58,
        sensor_height_mm=8.31,
        image_width_px=5472,
        image_height_px=3648,
        focal_length_mm=16.0,
        frontal_overlap=0.80,  # 80%
        side_overlap=0.2,  # 20%
        altitude_m=50.0,
    )
    CONNECTION = "udp:localhost:14550"
    TIMEOUT = 30
    print(f"[→] Connessione a {CONNECTION}...")
    mav = mavutil.mavlink_connection(CONNECTION)
    mav.wait_heartbeat(timeout=TIMEOUT)
    print(
        f"[✓] Heartbeat ricevuto — system={mav.target_system} component={mav.target_component}"
    )

    # ── Calcolo survey ────────────────────────────────────────────────────────
    transects, info = build_transects(polygon, camera, grid_angle_deg=0.0)
    print("DEBUG camera_calc:", camera_calc(camera))
    print("DEBUG info:", info)
    print(f"[i] Transect calcolati: {info['n_transects']}")
    print(f"[i] Passo tra strisce: {info['adjusted_footprint_side_m']:.1f}m")
    print(f"[i] Trigger ogni:      {info['trigger_distance_m']:.1f}m")

    items = build_mission_items(transects, trigger_dist_m=info["trigger_distance_m"])
    print(f"[i] Mission item totali: {len(items)}")

    # ── Upload su PX4 SITL ────────────────────────────────────────────────────
    # QGC usa 14550 → qui usiamo 14551 per non confliggere
    # Cambia in '/dev/ttyACM0' o 'tcp:IP:5760' per drone reale

    mav = upload_mission(mav=mav, items=items)

    # Decommentare per avviare automaticamente la missione:
    start_mission(mav, trigger_dist_m=info["trigger_distance_m"])
    set_camera_trigger(mav, info["trigger_distance_m"])

    time.sleep(30)

    polygon = [
        LatLon(35.061623, -118.157026),
        LatLon(35.059337, -118.158008),
        LatLon(35.061589, -118.160843),
        LatLon(35.059404, -118.163795),
    ]

    transects, info = build_transects(polygon, camera, grid_angle_deg=0.0)
    items = build_mission_items(transects, trigger_dist_m=info["trigger_distance_m"])
    mav = upload_mission(mav=mav, items=items)
    start_mission(mav, trigger_dist_m=info["trigger_distance_m"])
    set_camera_trigger(mav, info["trigger_distance_m"])
