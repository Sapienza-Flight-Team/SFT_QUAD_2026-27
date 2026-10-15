[11 05 2026 15:12] Luca: #!/usr/bin/env python3
"""
download_mission.py
Scarica la missione corrente da PX4 via MAVLink, 
replicando la logica di QGC MissionManager.

Flusso (identico a QGC):
  1. Richiedi il conteggio item  → MISSION_REQUEST_LIST
  2. Per ogni item i             → MISSION_REQUEST_INT (o MISSION_REQUEST)
  3. Ricevi MISSION_ITEM_INT     → salva item
  4. Invia MISSION_ACK           → segnala completamento
"""

from pymavlink import mavutil
from dataclasses import dataclass, field
from typing import Optional
import time
import json


# ─────────────────────────────────────────
#  Data model — un item della missione
# ─────────────────────────────────────────

@dataclass
class MissionItem:
    seq: int
    frame: int            # MAV_FRAME (0=global, 3=global_rel_alt, ecc.)
    command: int          # MAV_CMD (16=WAYPOINT, 22=TAKEOFF, 21=LAND, ecc.)
    current: int          # 1 se è l'item attivo
    autocontinue: int
    param1: float
    param2: float
    param3: float
    param4: float
    x: float              # latitudine (gradi)
    y: float              # longitudine (gradi)
    z: float              # altitudine (m)
    mission_type: int = 0  # 0=MISSION, 1=FENCE, 2=RALLY

    def __str__(self):
        cmd_names = {
            16: "WAYPOINT", 17: "LOITER_UNLIM", 18: "LOITER_TURNS",
            19: "LOITER_TIME", 20: "RETURN_TO_LAUNCH", 21: "LAND",
            22: "TAKEOFF", 85: "LOITER_TO_ALT", 189: "DO_JUMP",
            178: "DO_CHANGE_SPEED", 201: "DO_SET_ROI",
        }
        cmd_str = cmd_names.get(self.command, f"CMD_{self.command}")
        return (f"  [{self.seq:>3}] {cmd_str:<22} "
                f"lat={self.x:>12.7f}  lon={self.y:>13.7f}  alt={self.z:>8.2f}m  "
                f"{'[CURRENT]' if self.current else ''}")


# ─────────────────────────────────────────
#  Classe principale — MissionDownloader
# ─────────────────────────────────────────

class MissionDownloader:
    """
    Scarica la missione da PX4/ArduPilot esattamente come fa QGC
    in src/MissionManager/PlanManager.cc (_requestList → _requestNextMissionItem).
    """

    def __init__(
        self,
        connection_string: str = "udp:127.0.0.1:14550",
        timeout: float = 5.0,
        retries: int = 3,
    ):
        self.connection_string = connection_string
        self.timeout = timeout
        self.retries = retries
        self.mav: Optional[mavutil.mavfile] = None

    # ── Connessione ───────────────────────────────────────────────────────────

    def connect(self) -> None:
        print(f"[Mission] Connessione a {self.connection_string} ...")
        self.mav = mavutil.mavlink_connection(self.connection_string)
        self.mav.wait_heartbeat()
        print(f"[Mission] Heartbeat OK  "
              f"(sys={self.mav.target_system}, comp={self.mav.target_component})")

    # ── Step 1: richiedi conteggio ─────────────────────────────────────────

    def _request_list(self) -> int:
        """
        Invia MISSION_REQUEST_LIST e attende MISSION_COUNT.
        Equivalente a PlanManager::_requestList().
        """
        for attempt in range(self.retries):
            print(f"[Mission] MISSION_REQUEST_LIST (tentativo {attempt + 1}/{self.retries})...")
            self.mav.mav.mission_request_list_send(
                self.mav.target_system,
                self.mav.target_component,
                mavutil.mavlink.MAV_MISSION_TYPE_MISSION,  # 0 = missione di volo
            )

            msg = self.mav.recv_match(
                type="MISSION_COUNT",
                blocking=True,
                timeout=self.timeout,
            )
            if msg:
                print(f"[Mission] Missione contiene {msg.count} item(s).")
                return msg.count

            print(f"[Mission] ⚠️  Timeout — nessun MISSION_COUNT ricevuto.")

        raise TimeoutError("Impossibile ottenere MISSION_COUNT dopo i tentativi.")

    # ── Step 2: richiedi ogni item ─────────────────────────────────────────

    def _request_item(self, seq: int) -> MissionItem:
        """
        Invia MISSION_REQUEST_INT per l'item seq e attende MISSION_ITEM_INT.
        Equivalente a PlanManager::_requestNextMissionItem().
        """
        for attempt in range(self.retries):
            self.mav.mav.mission_request_int_send(
                self.mav.target_system,
                self.mav.target_component,
                seq,
                mavutil.mavlink.MAV_MISSION_TYPE_MISSION,
            )

            msg = self.mav.recv_match(
                type="MISSION_ITEM_INT",
                blocking=True,
                timeout=self.timeout,
            )

            if msg and msg.seq == seq:
                return MissionItem(
                    seq=msg.seq,
                    frame=msg.frame,
                    command=msg.command,
                    current=msg.current,
                    autocontinue=msg.autocontinue,
                    param1=msg.param1,
                    param2=msg.param2,
                    param3=msg.param3,
                    param4=msg.param4,
                    x=msg.x / 1e7,   # MISSION_ITEM_INT usa int32 × 1e7 per lat/lon
                    y=msg.y / 1e7,
                    z=msg.z,
                    mission_type=msg.mission_type,
                )
            elif msg:
                print(f"[Mission] ⚠️  Ricevuto seq={msg.seq}, atteso seq={seq} — riprovo.")

        raise TimeoutError(f"Impossibile ricevere item seq={seq}.")

    # ── Step 3: invia ACK finale ───────────────────────────────────────────

    def _send_ack(self) -> None:
        """
        Invia MISSION_ACK al termine del download.
        Equivalente a PlanManager::_finishTransaction(true).
        """
        self.mav.mav.mission_ack_send(
            self.mav.target_system,
            self.mav.target_component,
            mavutil.mavlink.MAV_MISSION_ACCEPTED,
            mavutil.mavlink.MAV_MISSION_TYPE_MISSION,
        )
        print("[Mission] MISSION_ACK inviato — transazione completata.")

    # ── API pubblica ───────────────────────────────────────────────────────

    def download(self) -> list[MissionItem]:
        """
        Esegue il download completo della missione.
        Restituisce la lista ordinata di MissionItem.
        """
        if self.mav is None:
            self.connect()

        count = self._request_list()

        if count == 0:
            print("[Mission] La missione è vuota.")
            self._send_ack()
            return []

        items: list[MissionItem] = []
        for seq in range(count):
            print(f"[Mission] Richiedo item {seq + 1}/{count} ...", end="\r")
            item = self._request_item(seq)
            items.append(item)

        print()  # newline dopo il progress inline
        self._send_ack()
        return items

    def to_json(self, items: list[MissionItem], path: str = "mission.json") -> None:
        """
        Salva la missione in formato JSON compatibile con QGC (.plan).
        """
        plan = {
            "fileType": "Plan",
            "version": 1,
            "groundStation": "pymavlink-downloader",
            "mission": {
                "count": len(items),
                "items": [
                    {
                        "seq": i.seq,
                        "command": i.command,
                        "frame": i.frame,
                        "params": [i.param1, i.param2, i.param3, i.param4,
                                   i.x, i.y, i.z],
                        "autocontinue": bool(i.autocontinue),
                        "current": bool(i.current),
                    }
                    for i in items
                ],
            },
        }
        with open(path, "w") as f:
            json.dump(plan, f, indent=2)
        print(f"[Mission] Missione salvata in '{path}'")
