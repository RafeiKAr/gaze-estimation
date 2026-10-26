
import torch
from model import build_model, set_trainable_layers, TRAINABLE_LAYER_MODES


# test trainable layers
"""
Unit-Test: Prüft die Anzahl trainierbarer Parameter für die drei Trainingsphasen.
Kein Training, keine Dateiänderungen, nur Modell-Instanziierung & Parameter-Count.
"""


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    print(f"Verfügbare Modi in TRAINABLE_LAYER_MODES: {TRAINABLE_LAYER_MODES}")
    print("-" * 60)

    # Einmal Modell bauen (ResNet18 + Head)
    model, model_name = build_model(activation="Sigmoid")
    print(f"Modell: {model_name}")
    print(f"Gesamtparameter: {sum(p.numel() for p in model.parameters()):,}")
    print("-" * 60)

    for mode in ("fc_only", "fc_layer4", "fc_resnet"):
        # Prüfen ob Modus gültig ist
        if mode not in TRAINABLE_LAYER_MODES:
            print(f"\t {mode}: NICHT in TRAINABLE_LAYER_MODES definiert! Übersprungen.")
            continue

        # Frisches Modell für jeden Test (sauberer State)
        model, _ = build_model(activation="Sigmoid")

        # Trainingsphase anwenden
        set_trainable_layers(model, mode)

        # Zählen
        n_trainable = count_trainable_params(model)
        print(f"{mode:12s}: {n_trainable:>12,} trainable params")


if __name__ == "__main__":
    main()
