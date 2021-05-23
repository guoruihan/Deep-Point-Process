import argparse
import json


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--config_path", default = "./hw1/config/univariate.json", help = "config saved path"
    )

    argparser.add_argument(
        "--save_path", default= "./hw1/results/", help="where to save simulation results"
    )

    args = argparser.parse_args()

    with open(args.config_path) as f:
        config = json.load(f)

    print(config)