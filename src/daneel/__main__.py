import argparse
from datetime import datetime, timezone

from daneel.detection.transit import run_for_planet


def main():
    parser = argparse.ArgumentParser(description="Model and analyse exoplanet transits.")

    parser.add_argument(
        "-i",
        "--input",
        dest="input_file",
        type=str,
        required=True,
        help="Input par file to pass",
    )

    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("-t", "--transit", action="store_true", help="Plot a transit light curve")

    actions.add_argument("-d", "--detect", action="store_true", help="Run exoplanet detection")
    actions.add_argument("-a", "--atmosphere", action="store_true", help="Run atmospheric characterisation")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output image for --transit (default: <slug>_light_curve.png)",
    )

    args = parser.parse_args()

    """Launch Daneel"""
    start = datetime.now(timezone.utc)

    if args.transit:
        run_for_planet(config_file=args.input_file, output_file=args.output)
    elif args.detect:
        parser.error("detection is not implemented yet; use --transit")
    elif args.atmosphere:
        parser.error("atmospheric characterisation is not implemented yet; use --transit")

    finish = datetime.now(timezone.utc)
    print(f"Daneel completed in {(finish - start).total_seconds():.2f}s")


if __name__ == "__main__":
    main()
