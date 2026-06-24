import sys
import time
import irsdk  # La bibliothèque pyirsdk qui gère la liaison mémoire avec iRacing

# Importation des composants nécessaires pour créer l'interface graphique (IHM)
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel
# QThread permet de lancer des calculs en arrière-plan sans bloquer la fenêtre.
# pyqtSignal permet au thread d'envoyer de manière sécurisée des données à la fenêtre.
from PyQt6.QtCore import QThread, pyqtSignal

# Importation de Matplotlib pour dessiner le graphique
import matplotlib.pyplot as plt
# Le "Canvas" est la passerelle (le pont) qui permet d'intégrer un graphique Matplotlib dans une fenêtre Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas


# ==============================================================================
# 1. LE WORKER (Le Thread en arrière-plan)
# ==============================================================================
# Pourquoi un Thread ? Si on mettait la boucle "while True" d'iRacing directement
# dans l'interface, la fenêtre Qt figerait instantanément ("Ne répond pas").
# Ce composant tourne sur son propre cœur CPU, en parallèle de l'affichage.
class TelemetryWorker(QThread):

    # ÉTAPE A : Définition d'un signal personnalisé.
    # Ce signal transportera un nombre à virgule (float) qui représentera la vitesse.
    speed_signal = pyqtSignal(float)

    def __init__(self):
        super().__init__()
        self.ir = irsdk.IRSDK()  # Initialisation de la connexion iRacing
        self.running = True      # Booléen pour contrôler l'arrêt propre de la boucle

    def run(self):
        """Le code dans cette méthode s'exécute automatiquement en arrière-plan."""
        while self.running:
            # ÉTAPE B : Tentative de connexion / vérification de session active sur iRacing
            if self.ir.startup():
                # On fige le buffer de variables actuel pour s'assurer que toutes les données
                # de cette frame (vitesse, rpm, etc.) soient synchronisées au même instant exact.
                self.ir.freeze_var_buffer_latest()

                # Récupération de la vitesse. iRacing l'envoie nativement en mètres par seconde (m/s).
                speed_mps = self.ir["Speed"]

                # Conversion des m/s en km/h (Formule : m/s * 3.6)
                speed_kmh = speed_mps * 3.6

                # ÉTAPE C : Émission du signal.
                # On "propulse" la vitesse calculée vers l'interface graphique.
                self.speed_signal.emit(speed_kmh)
            else:
                # Si iRacing n'est pas démarré ou qu'on est au menu, on envoie 0.0 pour ne pas laisser le graph vide
                self.speed_signal.emit(0.0)

            # ÉTAPE D : Temporisation.
            # La télémétrie en direct d'iRacing est rafraîchie à 60Hz.
            # On dort donc pendant 1/60ème de seconde pour calquer notre rythme sur le jeu.
            time.sleep(1 / 60)

    def stop(self):
        """Méthode appelée depuis l'extérieur pour arrêter le thread proprement."""
        self.running = False
        self.ir.shutdown()  # Fermeture propre de la connexion au SDK
        self.wait()         # Attend que la boucle while se termine définitivement


# ==============================================================================
# 2. L'INTERFACE GRAPHIQUE (La Fenêtre principale)
# ==============================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # Configuration basique de la fenêtre Qt
        self.setWindowTitle("iRacing Télémétrie - Mode Pédagogique")
        self.resize(800, 400)

        # ÉTAPE E : Gestion de l'historique des données pour l'animation.
        # Pour faire un effet de défilement de gauche à droite, on crée une liste fixe de 100 points.
        # Axe X : Représente les 100 dernières frames (ex: 0, 1, 2... 99)
        self.time_data = list(range(100))
        # Axe Y : Initialisé avec 100 zéros.
        self.speed_data = [0.0] * 100

        # ÉTAPE F : Création des composants visuels Qt (Layout)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        # Un layout vertical empile les éléments du haut vers le bas
        layout = QVBoxLayout(central_widget)

        # Ajout d'un Label texte pour afficher la vitesse sous forme numérique
        self.label_speed = QLabel("Vitesse : 0 km/h")
        self.label_speed.setStyleSheet(
            "font-size: 22px; font-weight: bold; color: #2c3e50;")
        layout.addWidget(self.label_speed)

        # ÉTAPE G : Initialisation du graphique Matplotlib
        # 'figure' est le conteneur global, 'ax' représente la zone de dessin (axes X/Y)
        self.figure, self.ax = plt.subplots()
        # Transformation de la figure en widget Qt
        self.canvas = FigureCanvas(self.figure)
        # Intégration du widget Matplotlib dans le layout Qt
        layout.addWidget(self.canvas)

        # Personnalisation des axes du graphique
        # On bloque l'axe Y de 0 à 320 km/h (parfait pour la monoplace/GT3)
        self.ax.set_ylim(0, 320)
        self.ax.set_title("Graphique de vitesse en temps réel")
        self.ax.set_ylabel("Vitesse (km/h)")
        self.ax.set_xlabel("Temps (Frames)")

        # Le fait d'ajouter une virgule après 'self.line' (destructuring) permet de récupérer
        # directement l'objet Line2D renvoyé dans une liste par Matplotlib.
        self.line, = self.ax.plot(
            self.time_data, self.speed_data, color='crimson', lw=2)

        # ÉTAPE H : Instanciation et liaison du Thread
        self.worker = TelemetryWorker()

        # C'est la ligne magique : Chaque fois que le Thread exécute "speed_signal.emit(vitesse)",
        # Qt intercepte le signal et exécute automatiquement notre fonction "self.update_gui" en lui passant la valeur.
        self.worker.speed_signal.connect(self.update_gui)

        # Lancement effectif du thread en arrière-plan
        self.worker.start()

    def update_gui(self, speed):
        """Cette fonction reçoit la vitesse extraite du thread 60 fois par seconde."""

        # 1. Mise à jour du texte à l'écran
        self.label_speed.setText(f"Vitesse : {speed:.1f} km/h")

        # 2. Gestion de l'effet "Défilement" (Scrolling)
        # Supprime la valeur la plus ancienne (à l'index 0, tout à gauche)
        self.speed_data.pop(0)
        # Ajoute la toute nouvelle vitesse reçue (tout à droite)
        self.speed_data.append(speed)

        # 3. Rafraîchissement graphique
        # Au lieu de tout recalculer et recréer le graphique (ce qui consommerait trop de CPU),
        # on met simplement à jour les données de la ligne existante.
        self.line.set_ydata(self.speed_data)

        # 'draw_idle' demande à Matplotlib de redessiner le graphique dès que l'interface Qt a un moment de libre.
        # C'est beaucoup plus fluide et moins lourd que 'draw()'.
        self.canvas.draw_idle()

    def closeEvent(self, event):
        """Cette méthode est un hook de Qt déclenché automatiquement si l'utilisateur clique sur le 'X' de la fenêtre."""
        print("Fermeture de l'application... Arrêt du thread de télémétrie.")
        # On stoppe impérativement la boucle du thread pour éviter un crash ou un processus fantôme
        self.worker.stop()
        event.accept()      # On accepte la fermeture de la fenêtre


# ==============================================================================
# Lancement de l'application
# ==============================================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())  # Démarre la boucle événementielle principale de Qt
