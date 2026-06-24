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

                # La voiture est considérée au garage si elle n'est plus activement sur la piste
                is_in_garage = not is_on_track

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

                    self.telemetry_signal.emit({
                        "lap_num": current_lap,
                        "lap_time": lap_time,
                        "speed": speed_kmh,
                        "in_outlap": self.in_outlap,
                        "is_in_pit": is_in_pit,
                        "is_in_garage": is_in_garage
                    })
                else:
                    self.last_lap = -1
                    self.in_outlap = True
                    self.telemetry_signal.emit({
                        "lap_num": 0, "lap_time": 0.0, "speed": 0.0,
                        "in_outlap": True, "is_in_pit": is_in_pit, "is_in_garage": True
                    })
            else:
                self.telemetry_signal.emit({
                    "lap_num": 0, "lap_time": 0.0, "speed": 0.0,
                    "in_outlap": True, "is_in_pit": False, "is_in_garage": True
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
        self.setWindowTitle("iRacing Télémétrie - Analyse & Overlay")
        self.resize(900, 600)

        self.time_data = []
        self.speed_data = []
        self.current_lap_buffer = []

        # Séparation des dictionnaires pour éviter les collisions de numéros de tours
        self.live_saved_laps = {}
        self.imported_laps = {}

        self.saved_lap_lines = {}
        self.last_processed_lap = -1
        self.saved_view_active = False

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.label_speed = QLabel("Vitesse : 0 km/h")
        self.label_speed.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(self.label_speed)

        ctrl_layout = QHBoxLayout()

        # Liste visuelle des tours (Widget)
        self.lap_list = QListWidget()
        self.lap_list.itemChanged.connect(self.on_lap_selection_changed)
        ctrl_layout.addWidget(self.lap_list)

        # Zone des boutons d'action
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

        # Graphique principal
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self.ax.set_ylim(0, 320)
        self.ax.set_xlim(0, 100)
        self.line, = self.ax.plot(
            self.time_data, self.speed_data, color='crimson', lw=2.5, label='Tour en cours')
        self.ax.legend(loc='upper right')

        # Démarrage du capteur de télémétrie
        self.worker = TelemetryWorker()
        self.worker.telemetry_signal.connect(self.update_gui)
        self.worker.start()

    def update_gui(self, data):
        lap_time = data["lap_time"]
        speed = data["speed"]
        lap_num = data["lap_num"]
        in_outlap = data["in_outlap"]
        is_in_pit = data["is_in_pit"]
        is_in_garage = data["is_in_garage"]

        # --- DÉTECTION DU CHANGEMENT OU DE L'ANNULATION DU TOUR ---
        if lap_time == 0.0 or (len(self.time_data) > 0 and lap_time < self.time_data[-1]) or is_in_garage or is_in_pit:

            # Sauvegarde uniquement si le tour s'est terminé normalement sur la ligne
            if self.current_lap_buffer and self.last_processed_lap > 0 and not in_outlap and not is_in_garage and not is_in_pit:
                # Seuil de sécurité minimal pour valider un tour
                if self.current_lap_buffer[-1][0] > 30.0:
                    self.live_saved_laps[self.last_processed_lap] = list(
                        self.current_lap_buffer)
                    print(
                        f"🏁 Tour {self.last_processed_lap} COMPLET sauvegardé ! ({self.current_lap_buffer[-1][0]:.2f}s)")
                    self.add_lap_to_list_widget(
                        self.last_processed_lap, self.current_lap_buffer[-1][0], origin="Session")
            else:
                # Signalement en cas d'abandon de la boucle
                if (is_in_garage or is_in_pit) and self.current_lap_buffer:
                    print(
                        f"⚠️ Tour {self.last_processed_lap} INCOMPLET (Retour Stands/Garage). Sauvegarde annulée.")

            # Reset complet des données en mémoire vive pour la nouvelle boucle
            self.time_data.clear()
            self.speed_data.clear()
            self.current_lap_buffer.clear()
            self.last_processed_lap = lap_num

        status_text = "OUTLAP" if in_outlap else "TRACK"
        self.label_speed.setText(
            f"Lap {lap_num} [{status_text}] | Temps : {lap_time:.2f}s | Vitesse : {speed:.1f} km/h")

        # --- ACCUMULATION & TRACÉ DU TOUR EN DIRECT ---
        if lap_time > 0 and speed > 0.5 and not in_outlap and not is_in_pit and not is_in_garage:
            self.time_data.append(lap_time)
            self.speed_data.append(speed)
            self.current_lap_buffer.append((lap_time, speed))

            self.line.set_visible(True)
            self.line.set_xdata(self.time_data)
            self.line.set_ydata(self.speed_data)
        else:
            if in_outlap or speed <= 0.5 or is_in_pit or is_in_garage:
                self.line.set_visible(False)

        if not self.saved_view_active and self.time_data:
            self.ax.set_xlim(0, max(10, self.time_data[-1] + 2))

        self.canvas.draw_idle()

    def add_lap_to_list_widget(self, lap_num, total_time, origin="Session"):
        """Ajoute et formate visuellement un tour dans l'interface."""
        # Évite les doublons stricts sur le couple ID + Provenance
        for i in range(self.lap_list.count()):
            it = self.lap_list.item(i)
            if it.data(256) == lap_num and it.data(257) == origin:
                return

        # Application du marquage distinctif demandé
        prefix = "🔴 [Session]" if origin == "Session" else "🔵 [Import]"
        item = QListWidgetItem(f"{prefix} Tour {lap_num} — {total_time:.2f}s")
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(Qt.CheckState.Unchecked)

        # Enregistrement des données invisibles pour la gestion interne
        item.setData(256, lap_num)
        item.setData(257, origin)

        self.lap_list.addItem(item)

    def on_lap_selection_changed(self, item):
        self.redraw_saved_laps()

    def redraw_saved_laps(self):
        """Met à jour le graphique en redessinant uniquement les overlays cochés."""
        for ln in list(self.saved_lap_lines.values()):
            try:
                ln.remove()
            except Exception:
                pass
        self.saved_lap_lines.clear()

        laps_to_show = []
        for i in range(self.lap_list.count()):
            it = self.lap_list.item(i)
            if it.checkState() == Qt.CheckState.Checked:
                laps_to_show.append({
                    "id": it.data(256),
                    "origin": it.data(257)
                })

        self.saved_view_active = len(laps_to_show) > 0

        max_x = 0
        for lap in laps_to_show:
            k = lap["id"]
            origin = lap["origin"]

            # Extraction des données depuis le bon dictionnaire source
            pts = self.live_saved_laps.get(
                k, []) if origin == "Session" else self.imported_laps.get(k, [])

            if pts:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                max_x = max(max_x, max(xs))

                label_name = f"Session - T{k}" if origin == "Session" else f"Import - T{k}"
                ln, = self.ax.plot(xs, ys, lw=1.5, alpha=0.7,
                                   linestyle='--', label=label_name)
                self.saved_lap_lines[f"{origin}_{k}"] = ln

        current_max_x = self.time_data[-1] if self.time_data else 0
        overall_max_x = max(max_x, current_max_x)

        if overall_max_x > 0:
            self.ax.set_xlim(0, overall_max_x + 2)

        try:
            self.ax.legend(loc='upper right')
        except Exception:
            pass

        self.canvas.draw_idle()

    # ==============================================================================
    # 3. LOGIQUE EXCEL (IMPORTATION / EXPORTATION)
    # ==============================================================================
    def export_laps_to_excel(self):
        """Génère un fichier Excel multi-onglets contenant uniquement les tours valides de la session active."""
        if not self.live_saved_laps:
            print("❌ Aucun tour de session valide enregistré à exporter.")
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Exporter les tours de la session", "session_iracing.xlsx", "Excel Files (*.xlsx)")
        if not filepath:
            return

        try:
            with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
                for lap_num, points in self.live_saved_laps.items():
                    df = pd.DataFrame(
                        points, columns=["Temps (s)", "Vitesse (km/h)"])
                    df.to_excel(
                        writer, sheet_name=f"Tour_{lap_num}", index=False)
            print(f"📤 Exportation réussie dans : {filepath}")
        except Exception as e:
            print(f"❌ Erreur lors de l'exportation : {e}")

    def import_laps_from_excel(self):
        """Importe des données de télémétrie externe et les injecte proprement avec l'origine 'Import'."""
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
                    points = list(zip(df["Temps (s)"], df["Vitesse (km/h)"]))

                    self.imported_laps[lap_num] = points

                    total_time = points[-1][0] if points else 0.0
                    # Correction appliquée : origin="Import" est désormais bien transmis
                    self.add_lap_to_list_widget(
                        lap_num, total_time, origin="Import")

            print(f"📥 Importation réussie depuis : {filepath}")
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
