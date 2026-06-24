import time
import irsdk


class IRacingTelemetryParser:
    def __init__(self):
        # Initialise le SDK d'iRacing
        self.ir = irsdk.IRSDK()
        self.is_connected = False

    def check_connection(self):
        """Vérifie si iRacing est lancé et si la session est active."""
        if self.ir.startup():
            if not self.is_connected:
                print("✅ Connecté à la télémétrie iRacing !")
                self.is_connected = True
            return True
        else:
            if self.is_connected:
                print("❌ Déconnecté d'iRacing.")
                self.is_connected = False
            return False

    def parse_current_frame(self):
        """Parse les données de la frame de télémétrie actuelle."""
        # On freeze les données de la frame actuelle pour éviter les désynchronisations
        self.ir.freeze_var_buffer_latest()

        # Liste des variables de télémétrie que l'on souhaite extraire
        # Tu peux en ajouter d'autres selon tes besoins (Speed, RPM, Gear, etc.)
        telemetry_data = {
            "timestamp": time.time(),
            "session_time": self.ir["SessionTime"],
            # Pédale d'accélérateur (0.0 à 1.0)
            "throttle": self.ir["Throttle"],
            # Pédale de frein (0.0 à 1.0)
            "brake": self.ir["Brake"],
            "clutch": self.ir["Clutch"],             # Embrayage (0.0 à 1.0)
            # Angle du volant en radians
            "steering_angle": self.ir["SteeringWheelAngle"],
            "rpm": self.ir["RPM"],                   # Régime moteur
            # Vitesse enclenchée (-1 = R, 0 = N, 1 = 1ère...)
            "gear": self.ir["Gear"]
        }

        return telemetry_data

    def start_logging(self, frequency_hz=60):
        """Boucle principale qui récupère les données à une fréquence donnée (ex: 60Hz)."""
        print("En attente d'iRacing... Lance une session en piste.")
        delay = 1.0 / frequency_hz

        try:
            while True:
                if self.check_connection():
                    data = self.parse_current_frame()

                    # \r remet le curseur au début de la ligne, end="" empêche de sauter à la ligne suivante
                    print(
                        f"\rRPM: {data['rpm']:4.0f} | Throttle: {data['throttle']*100:5.1f}% | Brake: {data['brake']*100:5.1f}%",
                        end="",
                        flush=True
                    )

                time.sleep(delay)

        except KeyboardInterrupt:
            print("\nArrêt du parser de télémétrie.")
            self.ir.shutdown()


if __name__ == "__main__":
    # Instanciation et lancement du parser
    parser = IRacingTelemetryParser()
    # iRacing envoie la télémétrie à 60Hz maximum en direct
    parser.start_logging(frequency_hz=60)
