import sys
import time
import math
import irsdk
import numpy as np
import pandas as pd

from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QHBoxLayout, QListWidget, QListWidgetItem, QPushButton, QFileDialog, QRadioButton, QScrollArea, QFrame
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QTimer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavToolbar


class ClickableLabel(QLabel):
    """QLabel émettant un signal clicked(index) au clic souris."""
    clicked = pyqtSignal(int)

    def __init__(self, index, text="", parent=None):
        super().__init__(text, parent)
        self._index = index
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mousePressEvent(self, ev):
        self.clicked.emit(self._index)
        super().mousePressEvent(ev)


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
                    my_idx = 0
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
                    my_idx = 0

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

                    # Lecture des coordonnées GPS classiques (Lat/Lon)
                    lat = self._safe_ir_read("Lat", 0.0)
                    lon = self._safe_ir_read("Lon", 0.0)

                    # (CarIdxPos* removed) — on ne lit plus de positions absolues
                    # Vitesses monde (m/s) — utiles pour reconstruire la trajectoire
                    velx = self._safe_ir_read("VelocityX", None)
                    if velx is None:
                        velx = self._safe_ir_read("VelocityX_ST", 0.0)
                    vely = self._safe_ir_read("VelocityY", None)
                    if vely is None:
                        vely = self._safe_ir_read("VelocityY_ST", 0.0)

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
                        "lat": lat,
                        "lon": lon,
                        "velx": velx,
                        "vely": vely,
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
        self.lat_data = []
        self.lon_data = []
        self.posx_data = []
        self.posy_data = []
        self.velx_data = []
        self.vely_data = []

        # Intégration des vitesses -> positions (état pour intégration incrémentale)
        self._last_velx = None
        self._last_vely = None
        self._last_time_for_pos = None
        self._last_posx = 0.0
        self._last_posy = 0.0
        # Flag pour activer/désactiver l'utilisation des coordonnées monde
        # Désactivé par défaut
        self.use_world_coords = False
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
        # État du curseur vertical et du zoom secteur
        self._cursor_x = None
        self._cursor_active = False
        self._sector_bounds_x = None
        self._zoomed_sector = None
        self._map_overlay_lines = []
        self._ref_track = None
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

        # --- MINI-CARTE DU CIRCUIT (trajectoire) ---
        self.live_col = '#e8112d'
        self.map_figure = plt.figure(facecolor='#ffffff')
        self.ax_map = self.map_figure.add_axes([0, 0, 1, 1])
        self.ax_map.set_aspect('equal', adjustable='datalim')
        self.ax_map.axis('off')
        self.map_canvas = FigureCanvas(self.map_figure)
        self.map_canvas.setFixedSize(260, 200)
        self.map_line, = self.ax_map.plot(
            [], [], color=self.live_col, lw=1.6)
        self.map_dot, = self.ax_map.plot(
            [], [], 'o', color='#111', markersize=7, zorder=5)
        ctrl_layout.addWidget(self.map_canvas)

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

        # --- BARRE DE SECTEURS (bas, style Garage 61, cliquable) ---
        self.sector_bar_labels = []
        sector_bar = QHBoxLayout()
        sector_bar.setSpacing(3)
        for i, name in enumerate(("S1", "S2", "S3")):
            seg = ClickableLabel(i, name)
            seg.setAlignment(Qt.AlignmentFlag.AlignCenter)
            seg.setToolTip("Cliquer pour zoomer sur ce secteur")
            seg.setStyleSheet(
                "font-size:13px; font-weight:bold; color:#555;"
                " background:#e6e6ea; padding:6px; border-radius:4px;")
            seg.clicked.connect(self._on_sector_click)
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

        # --- CURSEUR VERTICAL DÉPLAÇABLE (style Garage 61) ---
        self.cursor_lines = []
        self.cursor_texts = []
        for ax in self.all_axes:
            cl = ax.axvline(0, color='#222', lw=0.9, alpha=0.0, zorder=4)
            self.cursor_lines.append(cl)
            txt = ax.text(
                0, 0.92, "", transform=ax.get_xaxis_transform(),
                ha='center', va='top', fontsize=9, fontweight='bold',
                color='#ffffff', zorder=6,
                bbox=dict(boxstyle='round,pad=0.25', fc=self.live_col,
                          ec='none', alpha=0.92))
            txt.set_visible(False)
            self.cursor_texts.append(txt)
        self.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
        self.canvas.mpl_connect('axes_leave_event', self._on_mouse_leave)

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
        lat = data.get("lat", 0.0)
        lon = data.get("lon", 0.0)
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
            self.lat_data.clear()
            self.lon_data.clear()
            self.posx_data.clear()
            self.posy_data.clear()
            self.current_lap_buffer.clear()
            self.last_processed_lap = lap_num

            self.posx_data.clear()
            self.posy_data.clear()
            self.velx_data.clear()
            self.vely_data.clear()
            self._last_velx = None
            self._last_vely = None
            self._last_time_for_pos = None
            self._last_posx = 0.0
            self._last_posy = 0.0
            self.current_lap_buffer.clear()
        # Déterminer un libellé d'état pour l'affichage
        if in_outlap:
            status_text = "Outlap"
        elif is_in_garage:
            status_text = "Garage"
        elif is_in_pit:
            status_text = "Pit"
        else:
            status_text = "Sur piste"

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
            self.lat_data.append(lat)
            self.lon_data.append(lon)
            # Vitesses
            velx = data.get('velx', 0.0)
            vely = data.get('vely', 0.0)
            self.velx_data.append(velx)
            self.vely_data.append(vely)

            # Calcul incrémental de la position par intégration trapézoïdale
            if self._last_time_for_pos is None:
                cur_posx = 0.0
                cur_posy = 0.0
            else:
                dt = max(0.0, lap_time - self._last_time_for_pos)
                prev_vx = float(
                    self._last_velx) if self._last_velx is not None else float(velx or 0.0)
                prev_vy = float(
                    self._last_vely) if self._last_vely is not None else float(vely or 0.0)
                cur_posx = self._last_posx + 0.5 * \
                    (prev_vx + float(velx or 0.0)) * dt
                cur_posy = self._last_posy + 0.5 * \
                    (prev_vy + float(vely or 0.0)) * dt
            # update integration state
            self._last_posx = float(cur_posx)
            self._last_posy = float(cur_posy)
            self._last_velx = float(velx or 0.0)
            self._last_vely = float(vely or 0.0)
            self._last_time_for_pos = lap_time

            self.posx_data.append(cur_posx)
            self.posy_data.append(cur_posy)

            self.current_lap_buffer.append(
                (lap_time, speed, throttle, brake, gear, lap_dist, rpm, steer, lat, lon, cur_posx, cur_posy, 0.0))

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

            # Trajectoire live sur la mini-carte (GPS ou reconstruite)
            if not self.saved_view_active and len(self.time_data) > 3:
                xy = self._series_track_xy(self._get_reference_series())
                if xy is not None:
                    self.map_line.set_data(xy[0], xy[1])
                    self.ax_map.relim()
                    self.ax_map.autoscale_view()
                    self.ax_map.margins(0.05)
                    self._update_map_reference()
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
            if self._zoomed_sector is None:
                self.ax_bottom.set_xlim(0, max(10, live_x[-1] + 1))
            self._sector_bounds_x = self._compute_sector_bounds_live()

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
            self.map_canvas.draw_idle()
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

        # Nettoyage des trajectoires de la carte + reset du zoom secteur
        for ln in self._map_overlay_lines:
            try:
                ln.remove()
            except Exception:
                pass
        self._map_overlay_lines = []
        self._zoomed_sector = None

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

                # Trajectoire sur la mini-carte (GPS ou reconstruite)
                xy = self._series_track_xy(self._series_from_pts(pts))
                if xy is not None:
                    ml, = self.ax_map.plot(
                        xy[0], xy[1], lw=1.4, alpha=0.9, color=color)
                    self._map_overlay_lines.append(ml)

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

        # Carte : masquer la trajectoire live quand on compare des tours sauvés
        self.map_line.set_visible(not self.saved_view_active)
        if self._map_overlay_lines:
            self.ax_map.relim()
            self.ax_map.autoscale_view()
            self.ax_map.margins(0.05)

        # Tracé de référence pour le point de position sur la carte
        self._update_map_reference()

        # Série de référence (1er tour coché) pour le curseur et les secteurs
        self._sector_bounds_x = self._compute_sector_bounds(
            self._get_reference_series())

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

    # ------------------------------------------------------------------
    #  Série de référence, curseur interactif et zoom secteur
    # ------------------------------------------------------------------
    def _series_from_pts(self, pts):
        """Construit un dict de séries (numpy) à partir d'une liste de points."""
        arr = np.asarray(pts, dtype=float)
        m = arr.shape[1]
        xcol = 5 if (self.x_axis_mode == "distance" and m >= 6) else 0
        n = arr.shape[0]
        zeros = np.zeros(n)
        return {
            'x': arr[:, xcol],
            'time': arr[:, 0],
            'speed': arr[:, 1] if m > 1 else zeros,
            'throttle': arr[:, 2] if m > 2 else zeros,
            'brake': arr[:, 3] if m > 3 else zeros,
            'gear': arr[:, 4] if m > 4 else zeros,
            'dist': arr[:, 5] if m >= 6 else None,
            'rpm': arr[:, 6] if m >= 7 else zeros,
            'steer': arr[:, 7] if m >= 8 else zeros,
            'lat': arr[:, 8] if m >= 9 else None,
            'lon': arr[:, 9] if m >= 10 else None,
            'posx': arr[:, 10] if m >= 11 else None,
            'posy': arr[:, 11] if m >= 12 else None,
            'posz': arr[:, 12] if m >= 13 else None,
        }

    def _get_reference_series(self):
        """Retourne la série de référence : 1er tour coché, sinon le tour live."""
        if self.saved_view_active:
            for i in range(self.lap_list.count()):
                it = self.lap_list.item(i)
                if it.checkState() == Qt.CheckState.Checked:
                    k = it.data(256)
                    origin = it.data(257)
                    pts = (self.live_saved_laps.get(k, []) if origin == "Session"
                           else self.imported_laps.get(k, []))
                    if pts:
                        return self._series_from_pts(pts)
            return None
        if not self.time_data:
            return None
        xs = self.get_x_data()
        return {
            'x': np.asarray(xs, dtype=float),
            'time': np.asarray(self.time_data, dtype=float),
            'speed': np.asarray(self.speed_data, dtype=float),
            'throttle': np.asarray(self.throttle_data, dtype=float),
            'brake': np.asarray(self.brake_data, dtype=float),
            'gear': np.asarray(self.gear_data, dtype=float),
            'dist': np.asarray(self.dist_data, dtype=float) if self.dist_data else None,
            'rpm': np.asarray(self.rpm_data, dtype=float),
            'steer': np.asarray(self.steer_data, dtype=float),
            'lat': np.asarray(self.lat_data, dtype=float) if self.lat_data else None,
            'lon': np.asarray(self.lon_data, dtype=float) if self.lon_data else None,
            'posx': np.asarray(self.posx_data, dtype=float) if self.posx_data else None,
            'posy': np.asarray(self.posy_data, dtype=float) if self.posy_data else None,
        }

    # ------------------------------------------------------------------
    #  Reconstruction de la trajectoire (carte du circuit)
    # ------------------------------------------------------------------
    def _reconstruct_xy(self, v_kmh, steer_deg, t):
        """Reconstruit un tracé (x, y) par intégration vitesse + angle volant.

        Le cap est normalisé pour que le tour se referme (~360°), ce qui
        donne une forme de circuit reconnaissable même sans coordonnées GPS.
        """
        v = np.asarray(v_kmh, dtype=float) / 3.6  # m/s
        steer = np.radians(np.asarray(steer_deg, dtype=float))
        t = np.asarray(t, dtype=float)
        n = len(t)
        if n < 3:
            return None
        dt = np.diff(t, prepend=t[0])
        dt[dt < 0] = 0.0
        weighted = steer * dt
        total = float(np.sum(weighted))
        if abs(total) < 1e-6:
            return None
        gain = (2.0 * np.pi) / total
        heading = np.cumsum(weighted * gain)
        x = np.cumsum(v * np.cos(heading) * dt)
        y = np.cumsum(v * np.sin(heading) * dt)
        return x, y

    def _series_track_xy(self, s):
        """Retourne (xs, ys) pour la mini-carte à partir d'une série.

        Utilise Lat/Lon si elles varient suffisamment, sinon reconstruit le
        tracé à partir de la vitesse et de l'angle volant.
        """
        if s is None:
            return None
        # Coordonnées monde (posx/posy) : si présentes, seront détectées
        # plus bas et utilisées en priorité avant Lat/Lon.

        # Si des positions reconstruits (par intégration des vitesses) sont
        # présentes dans la série, les utiliser (priorité avant Lat/Lon).
        px = s.get('posx')
        py = s.get('posy')
        if px is not None and py is not None and len(px) > 2:
            px = np.asarray(px, dtype=float)
            py = np.asarray(py, dtype=float)
            if (np.nanmax(px) - np.nanmin(px) > 1e-6 or
                    np.nanmax(py) - np.nanmin(py) > 1e-6):
                return px, py

        lat = s.get('lat')
        lon = s.get('lon')
        if lat is not None and lon is not None and len(lat) > 2:
            lat = np.asarray(lat, dtype=float)
            lon = np.asarray(lon, dtype=float)
            if (np.nanmax(lon) - np.nanmin(lon) > 1e-5 or
                    np.nanmax(lat) - np.nanmin(lat) > 1e-5):
                return lon, lat
        t = s.get('time')
        if t is None:
            t = s.get('x')
        return self._reconstruct_xy(s.get('speed'), s.get('steer'), t)

    def _update_map_reference(self):
        """Met à jour le tracé de référence pour le point de curseur sur la carte."""
        ref = self._get_reference_series()
        xy = self._series_track_xy(ref)
        if xy is None or ref is None:
            self._ref_track = None
        else:
            self._ref_track = (np.asarray(ref['x'], dtype=float), xy[0], xy[1])

    def _compute_sector_bounds(self, ref):
        """Calcule les bornes X (début, fin) de chaque secteur depuis la référence."""
        if ref is None or ref.get('dist') is None:
            return None
        dist = ref['dist']
        x = ref['x']
        if len(dist) < 2 or dist[-1] <= 0:
            return None
        total = dist[-1]
        x1 = float(np.interp(total / 3.0, dist, x))
        x2 = float(np.interp(2.0 * total / 3.0, dist, x))
        return [(float(x[0]), x1), (x1, x2), (x2, float(x[-1]))]

    def _compute_sector_bounds_live(self):
        return self._compute_sector_bounds(self._get_reference_series())

    def _on_mouse_move(self, event):
        """Déplace le curseur vertical et met à jour les bulles de valeurs."""
        if event.inaxes not in self.all_axes or event.xdata is None:
            return
        self._cursor_x = event.xdata
        self._cursor_active = True
        self._update_cursor()
        self._plot_dirty = True

    def _on_mouse_leave(self, event):
        # On conserve le curseur à sa dernière position (lecture figée)
        pass

    def _update_cursor(self):
        """Positionne les lignes de curseur, les bulles et le point sur la carte."""
        x = self._cursor_x
        if x is None:
            return
        for cl in self.cursor_lines:
            cl.set_xdata([x, x])
            cl.set_alpha(0.55)

        ref = self._get_reference_series()
        channels = ['speed', 'throttle', 'brake', 'gear', 'rpm', 'steer']
        if ref is not None and len(ref['x']) > 1:
            xp = ref['x']
            for i, ch in enumerate(channels):
                v = float(np.interp(x, xp, ref[ch]))
                if ch == 'speed':
                    txt = f"{v:.0f} km/h"
                elif ch in ('throttle', 'brake'):
                    txt = f"{v:.0f} %"
                elif ch == 'gear':
                    g = int(round(v))
                    txt = "N" if g == 0 else ("R" if g < 0 else str(g))
                elif ch == 'rpm':
                    txt = f"{v:.0f} rpm"
                else:
                    txt = f"{v:.0f}°"
                bubble = self.cursor_texts[i]
                bubble.set_text(txt)
                bubble.set_position((x, 0.92))
                bubble.set_visible(True)

            # Point de position sur la mini-carte
            if self._ref_track is not None:
                tx, txs, tys = self._ref_track
                if len(tx) > 1:
                    dotx = float(np.interp(x, tx, txs))
                    doty = float(np.interp(x, tx, tys))
                    self.map_dot.set_data([dotx], [doty])
                    self.map_dot.set_visible(True)
        else:
            for bubble in self.cursor_texts:
                bubble.set_visible(False)

    def _on_sector_click(self, s):
        """Zoom sur le secteur cliqué (ou retour à la vue complète)."""
        bounds = self._sector_bounds_x or self._compute_sector_bounds_live()
        if not bounds or s >= len(bounds):
            return
        if self._zoomed_sector == s:
            self._zoomed_sector = None
            ref = self._get_reference_series()
            xmax = float(
                ref['x'][-1]) if (ref is not None and len(ref['x'])) else 10
            self.ax_bottom.set_xlim(0, xmax + 1)
        else:
            self._zoomed_sector = s
            x0, x1 = bounds[s]
            pad = max((x1 - x0) * 0.02, 0.01)
            self.ax_bottom.set_xlim(x0 - pad, x1 + pad)
        self._plot_dirty = True

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
                    if n >= 9:
                        cols.append("Lat")
                    if n >= 10:
                        cols.append("Lon")
                    # Ajouter les colonnes PosX/PosY/PosZ uniquement si
                    # l'option est activée (garder le support inactif par défaut)
                    if self.use_world_coords:
                        if n >= 11:
                            cols.append("PosX")
                        if n >= 12:
                            cols.append("PosY")
                        if n >= 13:
                            cols.append("PosZ")
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

                    if "PosX" in df.columns and "PosY" in df.columns:
                        points = list(zip(
                            df["Temps (s)"], df["Vitesse (km/h)"], df["Throttle (%)"],
                            df["Brake (%)"], df["Gear"], df.get(
                                "Distance (m)", [0]*len(df)),
                            df.get("RPM", [0]*len(df)
                                   ), df.get("Volant (deg)", [0]*len(df)),
                            df.get("Lat", [0]*len(df)
                                   ), df.get("Lon", [0]*len(df)),
                            df["PosX"], df["PosY"], df.get("PosZ", [0]*len(df))))
                    elif "Lon" in df.columns:
                        points = list(zip(
                            df["Temps (s)"], df["Vitesse (km/h)"], df["Throttle (%)"],
                            df["Brake (%)"], df["Gear"], df["Distance (m)"],
                            df["RPM"], df["Volant (deg)"], df["Lat"], df["Lon"]))
                    elif "Volant (deg)" in df.columns:
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
