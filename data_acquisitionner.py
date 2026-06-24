import csv
import time
import irsdk


class IRacingRawDumper:
    def __init__(self):
        self.ir = irsdk.IRSDK()
        self.filename = "telemetrie_brute_iracing.csv"
        self.file = None
        self.writer = None
        self.headers_written = False

    def start_dumping(self):
        print(f"🚀 Initialisation du dumper brut léger (Fréquence : 1Hz)...")
        print(f"💾 Les données seront écrites en direct dans : {self.filename}")
        print("⌨️ Appuie sur Ctrl+C pour arrêter l'enregistrement.\n")

        try:
            while True:
                if self.ir.startup():
                    self.ir.freeze_var_buffer_latest()

                    # Récupération de la liste des noms de variables exposées par iRacing
                    var_names = self.ir.var_headers_names

                    # Extraction dynamique de la valeur de chaque variable
                    current_data = {}
                    for name in var_names:
                        current_data[name] = self.ir[name]

                    # Initialisation et création du fichier CSV dès la première frame reçue
                    if not self.headers_written:
                        self.file = open(self.filename, mode="w",
                                         newline="", encoding="utf-8")
                        self.writer = csv.DictWriter(
                            self.file, fieldnames=var_names)
                        self.writer.writeheader()  # Écrit la ligne des en-têtes (titres des colonnes)
                        self.headers_written = True
                        print(
                            "🔴 Enregistrement en cours... iRacing transmet ses données (1 ligne / sec).")

                    # Écriture de la ligne de données actuelle
                    self.writer.writerow(current_data)

                    # Optionnel : On force l'écriture physique sur le disque immédiatement
                    # pour éviter que Windows garde les données en cache trop longtemps.
                    self.file.flush()

                    # Affichage de suivi basique dans la console
                    print(
                        f"\rVolume de données : {self.file.tell()} octets enregistrés...", end="", flush=True)

                else:
                    print("\r❌ En attente du simulateur iRacing...",
                          end="", flush=True)
                    if self.file:
                        self.file.close()
                        self.file = None
                        self.headers_written = False

                # --- MODIFICATION DE LA FRÉQUENCE ---
                # Au lieu de 1/60ème de seconde, on dort 1 seconde entière.
                # La boucle s'exécutera donc précisément une fois par seconde (1Hz).
                time.sleep(1.0)

        except KeyboardInterrupt:
            print("\n\n🛑 Arrêt de l'enregistrement demandé par l'utilisateur.")
        finally:
            # Sécurité : Fermeture du fichier si le script est coupé
            if self.file:
                self.file.close()
            self.ir.shutdown()
            print(
                f"💾 Terminé ! Ton fichier '{self.filename}' est prêt et optimisé.")


if __name__ == "__main__":
    dumper = IRacingRawDumper()
    dumper.start_dumping()
