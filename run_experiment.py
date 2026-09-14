import argparse
from pathlib import Path

from src.config import ExperimentConfig
from src.experiment import run_experiment

def main():
    parser = argparse.ArgumentParser(
        description="Experimento Tomek Links + D-min multiclase."
    )
    parser.add_argument(
        "--output",
        default="results/experimento_principal",
        help="Directorio de salida.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=10,
        help="Número de repeticiones de Stratified K-Fold.",
    )
    args = parser.parse_args()

    cfg = ExperimentConfig(n_repeats=args.repeats)
    result = run_experiment(cfg, Path(args.output))

    print("\nExperimento finalizado.")
    print("Resultados:", result["output_dir"])
    print("ZIP:", result["zip_path"])
    print("\nHipótesis:")
    print(result["final_hyp_df"].to_string(index=False))

if __name__ == "__main__":
    main()
