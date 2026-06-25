import sys
import time
import math
import irsdk
import pandas as pd

from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QHBoxLayout, QListWidget, QListWidgetItem, QPushButton, QFileDialog, QRadioButton, QScrollArea, QFrame
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QTimer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavToolbar


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
        self._dumped_ir_keys = False

    def _safe_ir_read(self, name, default=0.0):
        try:
            val = self.ir[name]
            if val is None:
                return default
            return float(val)
        except Exception:
            return default

    def _derive_wheel_temp(self, wheel_prefix):
        """Compute average temperature for a wheel from possible fields.
        Known fields in CSV: RFtempCL, RFtempCM, RFtempCR, etc.
        """
        try:
            # try common full-name first
            name_map = {
                'LF': ['TireTempLeftFront'],
                'RF': ['TireTempRightFront'],
                'LR': ['TireTempLeftRear'],
                'RR': ['TireTempRightRear']
            }
            for n in name_map.get(wheel_prefix, []):
                v = self._safe_ir_read(n, None)
                if v is not None:
                    return float(v)

            # then try CL/CM/CR triplet
            parts = ['tempCL', 'tempCM', 'tempCR']
            vals = []
            for p in parts:
                key = f"{wheel_prefix}{p}"
                try:
                    v = self.ir[key]
                    if v is not None:
                        vals.append(float(v))
                except Exception:
                    pass
            if vals:
                return sum(vals) / len(vals)
        except Exception:
            pass
        return 0.0

    def _derive_wheel_wear(self, wheel_prefix):
        """Compute average wear for a wheel from wearL/wearM/wearR fields."""
        try:
            parts = ['wearL', 'wearM', 'wearR']
            vals = []
            for p in parts:
                key = f"{wheel_prefix}{p}"
                try:
                    v = self.ir[key]
                    if v is not None:
                        vals.append(float(v))
                except Exception:
                    pass
            if vals:
                # wear values in CSV look like percentage integers
                return sum(vals) / len(vals)
        except Exception:
            pass
        return 0.0

    def run(self):
        while self.running:
            if self.ir.startup():
                # Dump available keys/attributes once when connected (diagnostic)
                if not self._dumped_ir_keys:
                    try:
                        with open("ir_keys_dump.txt", "w", encoding="utf-8") as f:
                            f.write("--- dir(self.ir) ---\n")
                            for name in sorted(dir(self.ir)):
                                f.write(f"{name}\n")
                            f.write("\n--- readable variables (attempt) ---\n")
                            for name in sorted(dir(self.ir)):
                                try:
                                    val = self.ir[name]
                                    f.write(f"{name} : {repr(val)}\n")
                                except Exception:
                                    # ignore non-indexable attributes
                                    pass
                    except Exception:
                        pass
                    self._dumped_ir_keys = True
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
                    _ld = self.ir["LapDist"]
                    lap_dist = float(_ld) if _ld is not None else 0.0

                    rpm = self._safe_ir_read("RPM", 0.0)
                    try:
                        steer = math.degrees(
                            float(self.ir["SteeringWheelAngle"] or 0.0))
                    except Exception:
                        steer = 0.0

                    self.telemetry_signal.emit({
                        "lap_num": current_lap,
                        "lap_time": lap_time,
                        "speed": speed_kmh,
                        "throttle": throttle,
                        "brake": brake,
                        "gear": gear,
                        # Tentative de lecture des températures pneus (noms courants iRacing)
                        # Lecture sûre des températures pneus (SDK via index)
                        # lecture classique puis fallback sur champs par capot/clm/cr
                        "tire_temp_lf": self._derive_wheel_temp("LF"),
                        "tire_temp_rf": self._derive_wheel_temp("RF"),
                        "tire_temp_lr": self._derive_wheel_temp("LR"),
                        "tire_temp_rr": self._derive_wheel_temp("RR"),
                        # Lecture de l'usure des pneus (si disponible)
                        "tire_wear_lf": self._derive_wheel_wear("LF"),
                        "tire_wear_rf": self._derive_wheel_wear("RF"),
                        "tire_wear_lr": self._derive_wheel_wear("LR"),
                        "tire_wear_rr": self._derive_wheel_wear("RR"),
                        "lap_dist": lap_dist,
                        "rpm": rpm,
                        "steer": steer,
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
                        "lap_dist": 0.0, "in_outlap": True, "is_in_pit": is_in_pit, "is_in_garage": True,
                        "track_name": track_name, "car_name": car_name
                    })
            else:
                self.telemetry_signal.emit({
                    "lap_num": 0, "lap_time": 0.0, "speed": 0.0, "throttle": 0.0, "brake": 0.0, "gear": 0,
                    "lap_dist": 0.0, "in_outlap": True, "is_in_pit": False, "is_in_garage": True,
                    "track_name": "Circuit", "car_name": "Voiture"
                })

            time.sleep(1 / 30)

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
        self.dist_data = []
        self.rpm_data = []
        self.steer_data = []
        self.x_axis_mode = "time"
        self.current_lap_buffer = []

        self.live_saved_laps = {}
        self.imported_laps = {}
        self.saved_lap_lines = {}

        self.last_processed_lap = -1
        self.saved_view_active = False
        self.current_track = "Circuit"
        self.current_car = "Voiture"
        self._plot_dirty = False
        self._fill_throttle = None
        self._fill_brake = None
        # Palette de couleurs cyclique pour les tours compar\u00e9s (style Garage 61)
        self._lap_palette = ["#e8112d", "#1565c0", "#2e7d32",
                             "#f9a825", "#6a1b9a", "#00838f", "#d84315"]

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.label_info = QLabel("Piste : --- | Voiture : ---")
        self.label_info.setStyleSheet(
            "font-size: 12px; color: #777; padding: 2px 4px;")
        layout.addWidget(self.label_info)

        # --- RUBAN DE DONNÉES (bandeau stylisé clair) ---
        ribbon = QFrame()
        ribbon.setObjectName("ribbon")
        ribbon.setStyleSheet(
            "#ribbon { background-color: #f5f5f7; border: 1px solid #dcdce0;"
            " border-radius: 8px; }")
        info_row = QHBoxLayout(ribbon)
        info_row.setContentsMargins(14, 10, 14, 10)
        info_row.setSpacing(0)

        self.label_speed = QLabel(
            "Vitesse : 0 km/h | Accel : 0% | Frein : 0% | Rapport : N")
        self.label_speed.setStyleSheet(
            "font-size: 17px; font-weight: bold; color: #1a1a1a; background: transparent;")
        info_row.addWidget(self.label_speed)

        info_row.addStretch()

        # Étiquettes températures pneus (LF, RF, LR, RR)
        self.tire_label_lf = QLabel("LF: --°C")
        self.tire_label_rf = QLabel("RF: --°C")
        self.tire_label_lr = QLabel("LR: --°C")
        self.tire_label_rr = QLabel("RR: --°C")
        for lbl in (self.tire_label_lf, self.tire_label_rf, self.tire_label_lr, self.tire_label_rr):
            lbl.setStyleSheet(
                "font-size:14px; font-weight:bold; color: #555;"
                " background:#ececef; border-radius:6px; padding:4px 10px; margin-left:8px;")
            info_row.addWidget(lbl)

        layout.addWidget(ribbon)

        ctrl_layout = QHBoxLayout()

        # Liste des tours (réduite en largeur et hauteur)
        self.lap_list = QListWidget()
        self.lap_list.setMaximumWidth(280)
        self.lap_list.setMinimumWidth(180)
        self.lap_list.setMaximumHeight(110)
        self.lap_list.itemChanged.connect(self.on_lap_selection_changed)
        ctrl_layout.addWidget(self.lap_list)

        # --- PANNEAU DE CONTRÔLE COMPACT ---
        _btn_css = (
            "QPushButton { background:#ffffff; color:#222; border:1px solid #cfcfcf;"
            " border-radius:6px; padding:6px 12px; font-size:12px; }"
            " QPushButton:hover { background:#eef2ff; border-color:#9db4ff; }")
        self.btn_export = QPushButton("\U0001F4E4 Exporter")
        self.btn_export.setToolTip("Exporter les tours Live (Excel)")
        self.btn_export.setStyleSheet(_btn_css)
        self.btn_export.clicked.connect(self.export_laps_to_excel)
        self.btn_import = QPushButton("\U0001F4E5 Importer")
        self.btn_import.setToolTip("Importer des tours (Excel)")
        self.btn_import.setStyleSheet(_btn_css)
        self.btn_import.clicked.connect(self.import_laps_from_excel)

        xaxis_label = QLabel("Axe X :")
        xaxis_label.setStyleSheet("font-size: 12px; font-weight: bold;")
        self.radio_time = QRadioButton("Temps (s)")
        self.radio_dist = QRadioButton("Distance (m)")
        self.radio_time.setChecked(True)
        self.radio_time.toggled.connect(self.on_xaxis_changed)
        self.radio_dist.toggled.connect(self.on_xaxis_changed)

        # Une seule ligne horizontale compacte : boutons + séparateur + axe X
        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(8)
        ctrl_row.addWidget(self.btn_export)
        ctrl_row.addWidget(self.btn_import)
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet("color:#cfcfcf;")
        ctrl_row.addWidget(sep)
        ctrl_row.addWidget(xaxis_label)
        ctrl_row.addWidget(self.radio_time)
        ctrl_row.addWidget(self.radio_dist)
        ctrl_row.addStretch()

        # Panneau des temps par secteur (S1/S2/S3 + écarts)
        self.sector_label = QLabel(
            "Secteurs : sélectionne un ou plusieurs tours")
        self.sector_label.setTextFormat(Qt.TextFormat.RichText)
        self.sector_label.setStyleSheet(
            "font-size:12px; color:#333; background:#f5f5f7;"
            " border:1px solid #dcdce0; border-radius:6px; padding:6px 10px;")

        ctrl_col = QVBoxLayout()
        ctrl_col.addLayout(ctrl_row)
        ctrl_col.addWidget(self.sector_label)
        ctrl_col.addStretch()
        ctrl_layout.addLayout(ctrl_col)
        layout.addLayout(ctrl_layout)

        # --- THÈME CLAIR (style Garage 61) ---
        plt.rcParams.update({
            'figure.facecolor':  '#ffffff',
            'axes.facecolor':    '#ffffff',
            'axes.edgecolor':    '#cfcfcf',
            'axes.linewidth':    1.0,
            'text.color':        '#333333',
            'axes.labelcolor':   '#333333',
            'xtick.color':       '#666666',
            'ytick.color':       '#666666',
            'xtick.labelsize':   9,
            'ytick.labelsize':   9,
            'grid.color':        '#e3e3e3',
            'grid.linestyle':    '--',
            'grid.linewidth':    0.7,
            'grid.alpha':        0.9,
            'font.size':         10,
            'axes.spines.top':   False,
            'axes.spines.right': False,
        })

        self.figure, axes = plt.subplots(
            6, 1, sharex=True,
            gridspec_kw={'height_ratios': [2.4, 1.3, 1.3, 1.0, 1.4, 1.4]})
        (self.ax_speed, self.ax_throttle, self.ax_brake,
         self.ax_gear, self.ax_rpm, self.ax_steer) = axes
        self.all_axes = list(axes)
        self.ax_bottom = self.ax_steer
        self.figure.set_facecolor('#ffffff')
        self.figure.subplots_adjust(
            left=0.07, right=0.93, top=0.99, bottom=0.04, hspace=0.18)

        self.canvas = FigureCanvas(self.figure)

        # Barre d'outils matplotlib : zoom, pan, reset, sauvegarde
        self.nav_toolbar = NavToolbar(self.canvas, self)
        self.nav_toolbar.setStyleSheet(
            "background:#f0f0f0; color:#333; border: none;")
        layout.addWidget(self.nav_toolbar)

        # Conteneur scrollable verticalement
        canvas_container = QWidget()
        canvas_container.setStyleSheet("background-color: #ffffff;")
        canvas_layout = QVBoxLayout(canvas_container)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        canvas_layout.addWidget(self.canvas)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(canvas_container)
        self.scroll_area.setStyleSheet(
            "background-color: #ffffff; border: none;")
        canvas_container.setMinimumHeight(1150)

        # stretch=1 : le scroll area prend tout l'espace vertical restant
        layout.addWidget(self.scroll_area, 1)

        # --- BARRE DE SECTEURS (bas, style Garage 61) ---
        self.sector_bar_labels = []
        sector_bar = QHBoxLayout()
        sector_bar.setSpacing(3)
        for name in ("S1", "S2", "S3"):
            seg = QLabel(name)
            seg.setAlignment(Qt.AlignmentFlag.AlignCenter)
            seg.setStyleSheet(
                "font-size:13px; font-weight:bold; color:#555;"
                " background:#e6e6ea; padding:6px; border-radius:4px;")
            self.sector_bar_labels.append(seg)
            sector_bar.addWidget(seg)
        layout.addLayout(sector_bar)

        # Couleur principale du tour Live
        self.live_col = '#e8112d'

        def _channel_label(ax, text):
            ax.text(1.012, 0.5, text, transform=ax.transAxes, rotation=270,
                    va='center', ha='left', fontsize=10, fontweight='bold',
                    color='#555555')

        # Graphique 1 : Vitesse
        self.ax_speed.set_ylim(0, 310)
        self.line_speed, = self.ax_speed.plot(
            [], [], color=self.live_col, lw=1.4)
        _channel_label(self.ax_speed, "Vitesse (km/h)")

        # Graphique 2 : Accélérateur
        self.ax_throttle.set_ylim(-3, 103)
        self.line_throttle, = self.ax_throttle.plot(
            [], [], color=self.live_col, lw=1.4)
        _channel_label(self.ax_throttle, "Accélérateur (%)")

        # Graphique 3 : Frein
        self.ax_brake.set_ylim(-3, 103)
        self.line_brake, = self.ax_brake.plot(
            [], [], color=self.live_col, lw=1.4)
        _channel_label(self.ax_brake, "Frein (%)")

        # Graphique 4 : Rapport
        self.ax_gear.set_ylim(-1.5, 8.5)
        self.line_gear, = self.ax_gear.plot(
            [], [], color=self.live_col, lw=1.4, drawstyle='steps-mid')
        _channel_label(self.ax_gear, "Rapport")

        # Graphique 5 : RPM
        self.ax_rpm.set_ylim(0, 9000)
        self.line_rpm, = self.ax_rpm.plot([], [], color=self.live_col, lw=1.4)
        _channel_label(self.ax_rpm, "RPM")

        # Graphique 6 : Angle volant
        self.ax_steer.set_ylim(-200, 200)
        self.line_steer, = self.ax_steer.plot(
            [], [], color=self.live_col, lw=1.4)
        _channel_label(self.ax_steer, "Volant (°)")

        self.ax_bottom.set_xlabel("Temps (s)")
        for ax in self.all_axes:
            ax.grid(True)
            ax.margins(x=0)

        # Timer de rendu découplé : matplotlib ne redessine qu'à ~15 fps
        self._plot_timer = QTimer(self)
        self._plot_timer.setInterval(66)
        self._plot_timer.timeout.connect(self._flush_plot)
        self._plot_timer.start()

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
        # Récupération températures et usures pneus (si présentes dans le signal)
        t_lf = data.get("tire_temp_lf", None)
        t_rf = data.get("tire_temp_rf", None)
        t_lr = data.get("tire_temp_lr", None)
        t_rr = data.get("tire_temp_rr", None)
        w_lf = data.get("tire_wear_lf", None)
        w_rf = data.get("tire_wear_rf", None)
        w_lr = data.get("tire_wear_lr", None)
        w_rr = data.get("tire_wear_rr", None)

        lap_dist = data.get("lap_dist", 0.0)
        rpm = data.get("rpm", 0.0)
        steer = data.get("steer", 0.0)
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
            self.dist_data.clear()
            self.rpm_data.clear()
            self.steer_data.clear()
            self.current_lap_buffer.clear()
            self.last_processed_lap = lap_num

        status_text = "OUTLAP" if in_outlap else "TRACK"
        self.label_speed.setText(
            f"Lap {lap_num} [{status_text}] | Temps : {lap_time:.2f}s | Vitesse : {speed:.1f} km/h | Gaz : {throttle:.0f}% | Frein : {brake:.0f}% | Vitesse : {gear_str}")

        # Mise à jour des températures pneus (si disponibles)
        def color_for_temp(t):
            if t is None:
                return "#666"
            try:
                t = float(t)
            except Exception:
                return "#666"
            if t < 70:
                return "#2e7d32"  # vert
            if t < 90:
                return "#f9a825"  # jaune
            return "#c62828"  # rouge

        def color_for_wear(w):
            if w is None:
                return "#666"
            try:
                w = float(w)
            except Exception:
                return "#666"
            # w is fraction: 1.0 == 100% remaining. Higher is better.
            pct = w * 100.0
            if pct >= 90:
                return "#2e7d32"
            if pct >= 70:
                return "#f9a825"
            return "#c62828"

        def badge_css(color):
            return (f"font-size:14px; font-weight:bold; color: {color};"
                    " background:#ececef; border-radius:6px; padding:4px 10px; margin-left:8px;")

        if t_lf is not None or w_lf is not None:
            txt = f"LF: {t_lf:.0f}°C / W:{((w_lf or 0.0) * 100):.0f}%"
            self.tire_label_lf.setText(txt)
            self.tire_label_lf.setStyleSheet(badge_css(color_for_wear(w_lf)))

        if t_rf is not None or w_rf is not None:
            txt = f"RF: {t_rf:.0f}°C / W:{((w_rf or 0.0) * 100):.0f}%"
            self.tire_label_rf.setText(txt)
            self.tire_label_rf.setStyleSheet(badge_css(color_for_wear(w_rf)))

        if t_lr is not None or w_lr is not None:
            txt = f"LR: {t_lr:.0f}°C / W:{((w_lr or 0.0) * 100):.0f}%"
            self.tire_label_lr.setText(txt)
            self.tire_label_lr.setStyleSheet(badge_css(color_for_wear(w_lr)))

        if t_rr is not None or w_rr is not None:
            txt = f"RR: {t_rr:.0f}°C / W:{((w_rr or 0.0) * 100):.0f}%"
            self.tire_label_rr.setText(txt)
            self.tire_label_rr.setStyleSheet(badge_css(color_for_wear(w_rr)))
        # --- ACCUMULATION & TRACÉ DU TOUR EN DIRECT ---
        if lap_time > 0 and not in_outlap and not is_in_pit and not is_in_garage:
            self.time_data.append(lap_time)
            self.speed_data.append(speed)
            self.throttle_data.append(throttle)
            self.brake_data.append(brake)
            self.gear_data.append(gear)
            self.dist_data.append(lap_dist)
            self.rpm_data.append(rpm)
            self.steer_data.append(steer)

            self.current_lap_buffer.append(
                (lap_time, speed, throttle, brake, gear, lap_dist, rpm, steer))

            x_data = self.get_x_data()
            self.line_speed.set_visible(True)
            self.line_speed.set_data(x_data, self.speed_data)

            self.line_throttle.set_visible(True)
            self.line_throttle.set_data(x_data, self.throttle_data)

            self.line_brake.set_visible(True)
            self.line_brake.set_data(x_data, self.brake_data)

            self.line_gear.set_visible(True)
            self.line_gear.set_data(x_data, self.gear_data)

            self.line_rpm.set_visible(True)
            self.line_rpm.set_data(x_data, self.rpm_data)

            self.line_steer.set_visible(True)
            self.line_steer.set_data(x_data, self.steer_data)
        else:
            if in_outlap or is_in_pit or is_in_garage:
                self.line_speed.set_visible(False)
                self.line_throttle.set_visible(False)
                self.line_brake.set_visible(False)
                self.line_gear.set_visible(False)
                self.line_rpm.set_visible(False)
                self.line_steer.set_visible(False)

        if not self.saved_view_active and self.time_data:
            live_x = self.get_x_data()
            self.ax_bottom.set_xlim(0, max(10, live_x[-1] + 1))

        self._plot_dirty = True

    def _flush_plot(self):
        """Appelé par le QTimer : redessine le canvas seulement si nécessaire."""
        if self._plot_dirty:
            # Suppression des anciennes zones remplies
            for attr in ('_fill_throttle', '_fill_brake'):
                coll = getattr(self, attr, None)
                if coll is not None:
                    try:
                        coll.remove()
                    except Exception:
                        pass
                    setattr(self, attr, None)

            # Zones remplies pour le tour en direct
            if self.time_data and self.line_speed.get_visible():
                x_data = self.get_x_data()
                self._fill_throttle = self.ax_throttle.fill_between(
                    x_data, 0, self.throttle_data,
                    alpha=0.15, color=self.live_col, linewidth=0)
                self._fill_brake = self.ax_brake.fill_between(
                    x_data, 0, self.brake_data,
                    alpha=0.15, color=self.live_col, linewidth=0)

            self.canvas.draw_idle()
            self._plot_dirty = False

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
        sector_rows = []  # (label, color, (s1, s2, s3))
        for idx, lap in enumerate(laps_to_show):
            k = lap["id"]
            origin = lap["origin"]
            color = self._lap_palette[idx % len(self._lap_palette)]

            pts = self.live_saved_laps.get(
                k, []) if origin == "Session" else self.imported_laps.get(k, [])

            if pts:
                has_extended_telemetry = len(pts[0]) > 2
                has_dist = len(pts[0]) >= 6
                has_rpm = len(pts[0]) >= 7
                has_steer = len(pts[0]) >= 8

                if self.x_axis_mode == "distance" and has_dist:
                    xs = [p[5] for p in pts]
                else:
                    xs = [p[0] for p in pts]
                ys_speed = [p[1] for p in pts]
                ys_throt = [p[2] for p in pts] if has_extended_telemetry else [
                    0] * len(pts)
                ys_brake = [p[3] for p in pts] if has_extended_telemetry else [
                    0] * len(pts)
                ys_gear = [p[4] for p in pts] if has_extended_telemetry else [
                    0] * len(pts)
                ys_rpm = [p[6] for p in pts] if has_rpm else [0] * len(pts)
                ys_steer = [p[7] for p in pts] if has_steer else [0] * len(pts)

                max_x = max(max_x, max(xs))

                label_name = f"T{k}" if origin == "Session" else f"I-T{k}"

                ln_sp, = self.ax_speed.plot(
                    xs, ys_speed, lw=1.4, alpha=0.9, color=color, label=label_name)
                ln_th, = self.ax_throttle.plot(
                    xs, ys_throt, lw=1.3, alpha=0.9, color=color)
                ln_bk, = self.ax_brake.plot(
                    xs, ys_brake, lw=1.3, alpha=0.9, color=color)
                ln_gr, = self.ax_gear.plot(
                    xs, ys_gear, lw=1.3, alpha=0.9, color=color, drawstyle='steps-mid')
                ln_rp, = self.ax_rpm.plot(
                    xs, ys_rpm, lw=1.3, alpha=0.9, color=color)
                ln_st, = self.ax_steer.plot(
                    xs, ys_steer, lw=1.3, alpha=0.9, color=color)

                self.saved_lap_lines[f"{origin}_{k}"] = [
                    ln_sp, ln_th, ln_bk, ln_gr, ln_rp, ln_st]

                sectors = self._compute_sectors(pts)
                if sectors is not None:
                    sector_rows.append((label_name, color, sectors))

        live_x = self.get_x_data()
        current_max_x = live_x[-1] if live_x else 0
        overall_max_x = max(max_x, current_max_x)

        if overall_max_x > 0:
            self.ax_bottom.set_xlim(0, overall_max_x + 1)

        self.ax_bottom.set_xlabel(
            "Distance (m)" if self.x_axis_mode == "distance" else "Temps (s)")

        try:
            self.ax_speed.legend(loc='upper right', fontsize=9, ncol=4)
        except Exception:
            pass

        self._update_sector_display(sector_rows)
        self._plot_dirty = True

    def _compute_sectors(self, pts):
        """Découpe un tour en 3 secteurs égaux en distance et retourne (s1, s2, s3)."""
        if not pts or len(pts[0]) < 6:
            return None
        total_dist = pts[-1][5]
        total_time = pts[-1][0]
        if total_dist <= 0:
            return None
        b1, b2 = total_dist / 3.0, 2.0 * total_dist / 3.0
        t1 = t2 = None
        for p in pts:
            if t1 is None and p[5] >= b1:
                t1 = p[0]
            if t2 is None and p[5] >= b2:
                t2 = p[0]
        if t1 is None:
            t1 = total_time
        if t2 is None:
            t2 = total_time
        return (t1, t2 - t1, total_time - t2)

    def _update_sector_display(self, sector_rows):
        """Met à jour le panneau des temps secteur + la barre du bas."""
        if not sector_rows:
            self.sector_label.setText(
                "Secteurs : sélectionne un ou plusieurs tours")
            for i, seg in enumerate(self.sector_bar_labels):
                seg.setText(f"S{i+1}")
                seg.setStyleSheet(
                    "font-size:13px; font-weight:bold; color:#555;"
                    " background:#e6e6ea; padding:6px; border-radius:4px;")
            return

        # Meilleur temps par secteur (référence)
        best = [min(r[2][s] for r in sector_rows) for s in range(3)]

        def fmt(t):
            return f"{t:.2f}"

        # Tableau HTML des tours et écarts
        html = ("<table cellspacing='6'><tr>"
                "<td><b>Tour</b></td><td><b>S1</b></td>"
                "<td><b>S2</b></td><td><b>S3</b></td><td><b>Total</b></td></tr>")
        for name, color, sec in sector_rows:
            total = sum(sec)
            cells = ""
            for s in range(3):
                gap = sec[s] - best[s]
                gtxt = "" if gap <= 0.001 else f" <span style='color:#888'>(+{gap:.2f})</span>"
                cells += f"<td>{fmt(sec[s])}{gtxt}</td>"
            html += (f"<tr><td><b style='color:{color}'>■</b> {name}</td>"
                     f"{cells}<td><b>{fmt(total)}</b></td></tr>")
        html += "</table>"
        self.sector_label.setText(html)

        # Barre du bas : meilleur secteur global, coloré par le tour qui le détient
        for s, seg in enumerate(self.sector_bar_labels):
            # Couleur du tour qui possède le meilleur secteur s
            holder = min(sector_rows, key=lambda r: r[2][s])
            seg.setText(f"S{s+1}  {fmt(best[s])}s")
            seg.setStyleSheet(
                f"font-size:13px; font-weight:bold; color:#fff;"
                f" background:{holder[1]}; padding:6px; border-radius:4px;")

    def get_x_data(self):
        """Retourne les données X actives (temps ou distance) pour le tour en cours."""
        if self.x_axis_mode == "distance" and self.dist_data:
            return self.dist_data
        return self.time_data

    def on_xaxis_changed(self):
        """Bascule l'axe X entre temps et distance et redessine."""
        new_mode = "distance" if self.radio_dist.isChecked() else "time"
        if new_mode == self.x_axis_mode:
            return
        self.x_axis_mode = new_mode
        x_label = "Distance (m)" if self.x_axis_mode == "distance" else "Temps (s)"
        self.ax_bottom.set_xlabel(x_label)
        x_data = self.get_x_data()
        if x_data:
            self.line_speed.set_data(x_data, self.speed_data)
            self.line_throttle.set_data(x_data, self.throttle_data)
            self.line_brake.set_data(x_data, self.brake_data)
            self.line_gear.set_data(x_data, self.gear_data)
            self.line_rpm.set_data(x_data, self.rpm_data)
            self.line_steer.set_data(x_data, self.steer_data)
            self.ax_bottom.set_xlim(0, max(10, x_data[-1] + 1))
        self.redraw_saved_laps()

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
                    n = len(points[0]) if points else 5
                    cols = ["Temps (s)", "Vitesse (km/h)",
                            "Throttle (%)", "Brake (%)", "Gear"]
                    if n >= 6:
                        cols.append("Distance (m)")
                    if n >= 7:
                        cols.append("RPM")
                    if n >= 8:
                        cols.append("Volant (deg)")
                    df = pd.DataFrame(points, columns=cols[:n])
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

                    if "Volant (deg)" in df.columns:
                        points = list(zip(
                            df["Temps (s)"], df["Vitesse (km/h)"], df["Throttle (%)"],
                            df["Brake (%)"], df["Gear"], df["Distance (m)"],
                            df["RPM"], df["Volant (deg)"]))
                    elif "RPM" in df.columns:
                        points = list(zip(
                            df["Temps (s)"], df["Vitesse (km/h)"], df["Throttle (%)"],
                            df["Brake (%)"], df["Gear"], df["Distance (m)"], df["RPM"]))
                    elif "Distance (m)" in df.columns:
                        points = list(zip(
                            df["Temps (s)"], df["Vitesse (km/h)"], df["Throttle (%)"], df["Brake (%)"], df["Gear"], df["Distance (m)"]))
                    elif "Throttle (%)" in df.columns:
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
