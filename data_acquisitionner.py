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
                # 💡 On vérifie si iRacing est démarré ET si les données de variables sont prêtes et valides
                if self.ir.startup() and self.ir.is_initialized:
                    self.ir.freeze_var_buffer_latest()

                    # Récupération de la liste de toutes les variables disponibles
                    var_names = self.ir.var_headers_names

                    # Si pour une raison obscure la liste est vide, on attend
                    if not var_names:
                        time.sleep(0.5)
                        continue

                    # Extraction dynamique de la valeur de chaque variable
                    current_data = {}
                    for name in var_names:
                        current_data[name] = self.ir[name]

                    # Initialisation et création du fichier CSV dès la première frame valide
                    if not self.headers_written:
                        self.file = open(
                            self.filename,
                            mode="w",
                            newline="",
                            encoding="utf-8",
                        )
                        self.writer = csv.DictWriter(
                            self.file, fieldnames=var_names
                        )
                        self.writer.writeheader()
                        self.headers_written = True
                        print(
                            "\n🔴 Enregistrement en cours... iRacing transmet ses données (1 ligne / sec)."
                        )

                    # Écriture de la ligne de données actuelle
                    self.writer.writerow(current_data)

                    # On force l'écriture sur le disque
                    self.file.flush()

                    # Affichage de suivi basique dans la console
                    print(
                        f"\rVolume de données : {self.file.tell()} octets enregistrés...",
                        end="",
                        flush=True,
                    )

                else:
                    print(
                        "\r❌ En attente du simulateur iRacing...",
                        end="",
                        flush=True,
                    )
                    if self.file:
                        self.file.close()
                        self.file = None
                        self.headers_written = False

                # Boucle cadencée à 1Hz (1 fois par seconde)
                time.sleep(1.0)

        except KeyboardInterrupt:
            print(
                "\n\n🛑 Arrêt de l'enregistrement demandé par l'utilisateur."
            )
        finally:
            if self.file:
                self.file.close()
            self.ir.shutdown()
            print(f"💾 Terminé ! Ton fichier '{self.filename}' est prêt.")


if __name__ == "__main__":
    # Petit script de diagnostic rapide
    dumper = IRacingRawDumper()
    print("🔍 Recherche des variables de pneus disponibles...")

    while not dumper.ir.startup() or not dumper.ir.is_initialized:
        print("\r⏳ En attente de la connexion à iRacing...", end="", flush=True)
        time.sleep(1)

    dumper.ir.freeze_var_buffer_latest()
    toutes_les_variables = sorted(dumper.ir.var_headers_names)

    # On filtre pour voir s'il y a des variables qui parlent de "temp" ou "wear"
    variables_pneus = [v for v in toutes_les_variables if "temp" in v.lower(
    ) or "wear" in v.lower() or "pressione" in v.lower()]

    print("\n\n✅ Connexion réussie !")
    print(
        f"📊 Nombre total de variables envoyées par cette voiture : {len(toutes_les_variables)}")
    print("\n📦 Variables détectées liées aux pneus/températures :")
    if variables_pneus:
        for var in variables_pneus:
            print(f"  - {var}")
    else:
        print("  ❌ Aucune variable contenant 'temp' ou 'wear' n'a été trouvée dans le flux initial.")
        print("\n💡 Variables dispo (les 15 premières pour voir la structure) :")
        for var in toutes_les_variables[:15]:
            print(f"  - {var}")
