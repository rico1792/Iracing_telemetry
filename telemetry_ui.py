import sys
import time
import irsdk
import pandas as pd

from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QHBoxLayout, QListWidget, QListWidgetItem, QPushButton, QFileDialog
from PyQt6.QtCore import QThread, pyqtSignal, Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas


# ==============================================================================
# 1. LE WORKER (Le Thread en arrière-plan)
# ==============================================================================
class TelemetryWorker(QThread):
    telemetry_signal = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.ir = irsdk.IRSDK()
        self.running = True
        self.last_lap = -1
        self.lap_start_time = 0.0
        self.in_outlap = True

    def run(self):
        while self.running:
            if self.ir.startup():
                self.ir.freeze_var_buffer_latest()

                current_lap = self.ir["Lap"]
                is_on_track = self.ir["IsOnTrack"]
                session_time = self.ir["SessionTime"]

                try:
                    is_in_pit = bool(self.ir["OnPitRoad"])
                except Exception:
                    is_in_pit = False

                is_in_garage = not is_on_track

                # Extraction directe avec inspection temporaire
                track_name = "Circuit"
                car_name = "Voiture"

                try:
                    # WeekendInfo et DriverInfo sont des clés racines du YAML iRacing,
                    # pas des enfants de SessionInfo
                    weekend_info = self.ir['WeekendInfo']
                    if isinstance(weekend_info, dict):
                        track_name = weekend_info.get('TrackDisplayName',
                                                      weekend_info.get('TrackName', 'Circuit'))

                    driver_info = self.ir['DriverInfo']
                    if isinstance(driver_info, dict):
                        my_idx = driver_info.get('DriverCarIdx', 0)
                        drivers = driver_info.get('Drivers', [])
                        for d in drivers:
                            if d.get('CarIdx') == my_idx:
                                car_name = d.get(
                                    'CarScreenNameShort', 'Voiture')
                                break
                except Exception as e:
                    print(f"⚠️ Erreur parsing dictionnaire : {e}")
                    track_name = "Circuit"
                    car_name = "Voiture"

                if is_on_track and current_lap > 0:
                    if is_in_pit:
                        self.in_outlap = True

                    if current_lap != self.last_lap:
                        self.lap_start_time = session_time
                        self.last_lap = current_lap

                        if not is_in_pit:
                            self.in_outlap = False

                    lap_time = session_time - self.lap_start_time
                    speed_kmh = self.ir["Speed"] * 3.6

                    throttle = self.ir["Throttle"] * 100.0
                    brake = self.ir["Brake"] * 100.0
                    gear = self.ir["Gear"]

                    self.telemetry_signal.emit({
                        "lap_num": current_lap,
                        "lap_time": lap_time,
                        "speed": speed_kmh,
                        "throttle": throttle,
                        "brake": brake,
                        "gear": gear,
                        "in_outlap": self.in_outlap,
                        "is_in_pit": is_in_pit,
                        "is_in_garage": is_in_garage,
                        "track_name": track_name,
                        "car_name": car_name
                    })
                else:
                    self.last_lap = -1
                    self.in_outlap = True
                    self.telemetry_signal.emit({
                        "lap_num": 0, "lap_time": 0.0, "speed": 0.0, "throttle": 0.0, "brake": 0.0, "gear": 0,
                        "in_outlap": True, "is_in_pit": is_in_pit, "is_in_garage": True,
                        "track_name": track_name, "car_name": car_name
                    })
            else:
                self.telemetry_signal.emit({
                    "lap_num": 0, "lap_time": 0.0, "speed": 0.0, "throttle": 0.0, "brake": 0.0, "gear": 0,
                    "in_outlap": True, "is_in_pit": False, "is_in_garage": True,
                    "track_name": "Circuit", "car_name": "Voiture"
                })

            time.sleep(1 / 60)

    def stop(self):
        self.running = False
        self.ir.shutdown()
        self.wait()


# ==============================================================================
# 2. L'INTERFACE GRAPHIQUE (UI)
# ==============================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("iRacing Télémétrie Pro - Multi-Inputs")
        self.resize(1000, 750)

        self.time_data = []
        self.speed_data = []
        self.throttle_data = []
        self.brake_data = []
        self.gear_data = []
        self.current_lap_buffer = []

        self.live_saved_laps = {}
        self.imported_laps = {}
        self.saved_lap_lines = {}

        self.last_processed_lap = -1
        self.saved_view_active = False
        self.current_track = "Circuit"
        self.current_car = "Voiture"

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.label_info = QLabel("Piste : --- | Voiture : ---")
        self.label_info.setStyleSheet("font-size: 13px; color: gray;")
        layout.addWidget(self.label_info)

        self.label_speed = QLabel(
            "Vitesse : 0 km/h | Accel : 0% | Frein : 0% | Rapport : N")
        self.label_speed.setStyleSheet(
            "font-size: 14px; font-weight: bold; color: #111;")
        layout.addWidget(self.label_speed)

        ctrl_layout = QHBoxLayout()

        self.lap_list = QListWidget()
        self.lap_list.itemChanged.connect(self.on_lap_selection_changed)
        ctrl_layout.addWidget(self.lap_list)

        btn_layout = QVBoxLayout()
        self.btn_export = QPushButton("📤 Exporter les tours Live (Excel)")
        self.btn_export.clicked.connect(self.export_laps_to_excel)
        self.btn_import = QPushButton("📥 Importer des tours (Excel)")
        self.btn_import.clicked.connect(self.import_laps_from_excel)
        btn_layout.addWidget(self.btn_export)
        btn_layout.addWidget(self.btn_import)
        btn_layout.addStretch()

        ctrl_layout.addLayout(btn_layout)
        layout.addLayout(ctrl_layout)

        # --- CONFIGURATION MULTI-GRAPHIQUES MATPLOTLIB ---
        self.figure, (self.ax_speed, self.ax_inputs, self.ax_gear) = plt.subplots(
            3, 1, sharex=True, gridspec_kw={'height_ratios': [3, 2, 1]})
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Graphique 1 : Vitesse
        self.ax_speed.set_ylabel("Vitesse (km/h)")
        self.ax_speed.set_ylim(0, 310)
        self.line_speed, = self.ax_speed.plot(
            [], [], color='crimson', lw=2, label='Vitesse (Live)')
        self.ax_speed.legend(loc='upper right')
        self.ax_speed.grid(True, alpha=0.3)

        # Graphique 2 : Pédales (Throttle & Brake)
        self.ax_inputs.set_ylabel("Pédales (%)")
        self.ax_inputs.set_ylim(-5, 105)
        self.line_throttle, = self.ax_inputs.plot(
            [], [], color='green', lw=1.5, label='Accélérateur')
        self.line_brake, = self.ax_inputs.plot(
            [], [], color='red', lw=1.5, label='Frein')
        self.ax_inputs.legend(loc='upper right')
        self.ax_inputs.grid(True, alpha=0.3)

        # Graphique 3 : Rapport (Gear)
        self.ax_gear.set_ylabel("Rapport")
        self.ax_gear.set_xlabel("Temps (s)")
        self.ax_gear.set_ylim(-1.5, 8.5)
        self.line_gear, = self.ax_gear.plot(
            [], [], color='purple', lw=1.5, drawstyle='steps-mid', label='Rapport')
        self.ax_gear.grid(True, alpha=0.3)

        self.figure.tight_layout()

        self.worker = TelemetryWorker()
        self.worker.telemetry_signal.connect(self.update_gui)
        self.worker.start()

    def update_gui(self, data):
        lap_time = data["lap_time"]
        speed = data["speed"]
        throttle = data["throttle"]
        brake = data["brake"]
        gear = data["gear"]
        lap_num = data["lap_num"]
        in_outlap = data["in_outlap"]
        is_in_pit = data["is_in_pit"]
        is_in_garage = data["is_in_garage"]

        self.current_track = data["track_name"]
        self.current_car = data["car_name"]
        self.label_info.setText(
            f"Piste : {self.current_track} | Voiture : {self.current_car}")

        gear_str = "R" if gear == -1 else ("N" if gear == 0 else str(gear))

        # --- DÉTECTION DU CHANGEMENT OU DE L'ANNULATION DU TOUR ---
        if lap_time == 0.0 or (len(self.time_data) > 0 and lap_time < self.time_data[-1]) or is_in_garage or is_in_pit:

            if self.current_lap_buffer and self.last_processed_lap > 0 and not in_outlap and not is_in_garage and not is_in_pit:
                if self.current_lap_buffer[-1][0] > 10.0:
                    self.live_saved_laps[self.last_processed_lap] = list(
                        self.current_lap_buffer)
                    print(
                        f"🏁 Tour {self.last_processed_lap} COMPLET sauvegardé ! ({self.current_lap_buffer[-1][0]:.2f}s)")
                    self.add_lap_to_list_widget(
                        self.last_processed_lap, self.current_lap_buffer[-1][0], origin="Session")
            else:
                if (is_in_garage or is_in_pit) and self.current_lap_buffer:
                    print(
                        f"⚠️ Tour {self.last_processed_lap} INCOMPLET (Retour Stands/Garage). Sauvegarde annulée.")

            self.time_data.clear()
            self.speed_data.clear()
            self.throttle_data.clear()
            self.brake_data.clear()
            self.gear_data.clear()
            self.current_lap_buffer.clear()
            self.last_processed_lap = lap_num

        status_text = "OUTLAP" if in_outlap else "TRACK"
        self.label_speed.setText(
            f"Lap {lap_num} [{status_text}] | Temps : {lap_time:.2f}s | Vitesse : {speed:.1f} km/h | Gaz : {throttle:.0f}% | Frein : {brake:.0f}% | Vitesse : {gear_str}")

        # --- ACCUMULATION & TRACÉ DU TOUR EN DIRECT ---
        if lap_time > 0 and not in_outlap and not is_in_pit and not is_in_garage:
            self.time_data.append(lap_time)
            self.speed_data.append(speed)
            self.throttle_data.append(throttle)
            self.brake_data.append(brake)
            self.gear_data.append(gear)

            self.current_lap_buffer.append(
                (lap_time, speed, throttle, brake, gear))

            self.line_speed.set_visible(True)
            self.line_speed.set_data(self.time_data, self.speed_data)

            self.line_throttle.set_visible(True)
            self.line_throttle.set_data(self.time_data, self.throttle_data)

            self.line_brake.set_visible(True)
            self.line_brake.set_data(self.time_data, self.brake_data)

            self.line_gear.set_visible(True)
            self.line_gear.set_data(self.time_data, self.gear_data)
        else:
            if in_outlap or is_in_pit or is_in_garage:
                self.line_speed.set_visible(False)
                self.line_throttle.set_visible(False)
                self.line_brake.set_visible(False)
                self.line_gear.set_visible(False)

        if not self.saved_view_active and self.time_data:
            self.ax_gear.set_xlim(0, max(10, self.time_data[-1] + 1))

        self.canvas.draw_idle()

    def add_lap_to_list_widget(self, lap_num, total_time, origin="Session"):
        for i in range(self.lap_list.count()):
            it = self.lap_list.item(i)
            if it.data(256) == lap_num and it.data(257) == origin:
                return

        prefix = "🔴 [Session]" if origin == "Session" else "🔵 [Import]"
        item = QListWidgetItem(f"{prefix} Tour {lap_num} — {total_time:.2f}s")
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(Qt.CheckState.Unchecked)

        item.setData(256, lap_num)
        item.setData(257, origin)
        self.lap_list.addItem(item)

    def on_lap_selection_changed(self, item):
        self.redraw_saved_laps()

    def redraw_saved_laps(self):
        """Redessine proprement l'ensemble des overlays sur tous les graphiques concernés."""
        # Nettoyage des anciennes courbes d'overlay
        for sub_dict in list(self.saved_lap_lines.values()):
            for ln in sub_dict:
                try:
                    ln.remove()
                except Exception:
                    pass
        self.saved_lap_lines.clear()

        laps_to_show = []
        for i in range(self.lap_list.count()):
            it = self.lap_list.item(i)
            if it.checkState() == Qt.CheckState.Checked:
                laps_to_show.append(
                    {"id": it.data(256), "origin": it.data(257)})

        self.saved_view_active = len(laps_to_show) > 0

        max_x = 0
        for lap in laps_to_show:
            k = lap["id"]
            origin = lap["origin"]

            pts = self.live_saved_laps.get(
                k, []) if origin == "Session" else self.imported_laps.get(k, [])

            if pts:
                # CORRECTION ICI : On vérifie la taille du premier élément de 'pts'
                has_extended_telemetry = len(pts[0]) > 2

                xs = [p[0] for p in pts]
                ys_speed = [p[1] for p in pts]

                # Utilisation de la condition corrigée
                ys_throt = [p[2]
                            for p in pts] if has_extended_telemetry else [0]*len(pts)
                ys_brake = [p[3]
                            for p in pts] if has_extended_telemetry else [0]*len(pts)
                ys_gear = [p[4]
                           for p in pts] if has_extended_telemetry else [0]*len(pts)

                max_x = max(max_x, max(xs))

                label_name = f"S-T{k}" if origin == "Session" else f"I-T{k}"

                # Tracé sur l'axe Vitesse
                ln_sp, = self.ax_speed.plot(
                    xs, ys_speed, lw=1.2, alpha=0.6, linestyle='--', label=f"{label_name} (Vit)")
                # Tracé sur l'axe Inputs (Accélérateur en pointillé vert léger, Frein en rouge léger)
                ln_th, = self.ax_inputs.plot(
                    xs, ys_throt, lw=1.0, alpha=0.5, linestyle=':', color='green')
                ln_bk, = self.ax_inputs.plot(
                    xs, ys_brake, lw=1.0, alpha=0.5, linestyle=':', color='red')
                # Tracé sur l'axe Boite
                ln_gr, = self.ax_gear.plot(
                    xs, ys_gear, lw=1.0, alpha=0.5, linestyle='--', drawstyle='steps-mid')

                self.saved_lap_lines[f"{origin}_{k}"] = [
                    ln_sp, ln_th, ln_bk, ln_gr]

        current_max_x = self.time_data[-1] if self.time_data else 0
        overall_max_x = max(max_x, current_max_x)

        if overall_max_x > 0:
            self.ax_gear.set_xlim(0, overall_max_x + 1)

        try:
            self.ax_speed.legend(loc='upper right')
        except Exception:
            pass

        self.canvas.draw_idle()

    def export_laps_to_excel(self):
        if not self.live_saved_laps:
            print("❌ Aucun tour de session valide enregistré à exporter.")
            return

        # 1. Récupération de la date du jour au format AAAA-MM-JJ
        from datetime import datetime
        date_str = datetime.now().strftime("%Y-%m-%d")

        # 2. Intégration de la date dans le nom par défaut du fichier
        track_clean = self.current_track.replace(' ', '_')
        car_clean = self.current_car.replace(' ', '_')
        default_name = f"{date_str}_{track_clean}_{car_clean}.xlsx"

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Exporter les données", default_name, "Excel Files (*.xlsx)")
        if not filepath:
            return

        try:
            with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
                for lap_num, points in self.live_saved_laps.items():
                    df = pd.DataFrame(points, columns=[
                                      "Temps (s)", "Vitesse (km/h)", "Throttle (%)", "Brake (%)", "Gear"])
                    df.to_excel(
                        writer, sheet_name=f"Tour_{lap_num}", index=False)
            print(f"📤 Données exportées avec succès : {filepath}")
        except Exception as e:
            print(f"❌ Erreur lors de l'exportation : {e}")

    def import_laps_from_excel(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Importer des tours", "", "Excel Files (*.xlsx)")
        if not filepath:
            return

        try:
            excel_file = pd.ExcelFile(filepath)
            for sheet_name in excel_file.sheet_names:
                if sheet_name.startswith("Tour_"):
                    lap_num = int(sheet_name.split("_")[1])

                    df = excel_file.parse(sheet_name)

                    if "Throttle (%)" in df.columns:
                        points = list(zip(
                            df["Temps (s)"], df["Vitesse (km/h)"], df["Throttle (%)"], df["Brake (%)"], df["Gear"]))
                    else:
                        points = list(
                            zip(df["Temps (s)"], df["Vitesse (km/h)"], [0]*len(df), [0]*len(df), [0]*len(df)))

                    self.imported_laps[lap_num] = points
                    total_time = points[-1][0] if points else 0.0
                    self.add_lap_to_list_widget(
                        lap_num, total_time, origin="Import")

            print(f"📥 Données importées avec succès depuis : {filepath}")
            self.redraw_saved_laps()
        except Exception as e:
            print(f"❌ Erreur lors de l'importation : {e}")

    def closeEvent(self, event):
        self.worker.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
