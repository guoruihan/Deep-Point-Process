import argparse
import json

if __name__ == "main":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--config_path", default = "./hw1/config/bivariate.json", help = "config saved path"
    )