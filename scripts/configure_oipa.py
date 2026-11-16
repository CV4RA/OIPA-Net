"""Bind the archived OIPA-Net recipes to a local data and artifact layout."""
import argparse
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
AUTODL_RUNS = "/root/autodl-tmp/OIPA-Net/"
DATASETS = {"gsv-cities": "", "msls-derived": "_msls", "sf-xl-derived": "_sfxl"}


def configure(args):
    suffix = DATASETS[args.dataset]
    stages = ["oipa"]
    if not suffix:
        stages.append("oipa_loss")
    stages.extend(["oipa_backbones", "pe_lora", "backbone_lora", "oipa_itm"])
    model_root = args.model_root.resolve()
    run_root = args.run_root.resolve()
    config_root = run_root / "configs"
    config_root.mkdir(parents=True, exist_ok=True)
    output_locations = ("output", "reference", "control", "pe_loss_study", "backbone_study")

    for stage in stages:
        template = REPO / "configs" / f"{stage}{suffix}_autodl.yaml"
        config = yaml.safe_load(template.read_text(encoding="utf-8"))
        for key in output_locations:
            if key in config:
                path = config[key]
                if not path.startswith(AUTODL_RUNS):
                    raise ValueError(f"Unexpected archived path for {key}: {path}")
                config[key] = (run_root / path.removeprefix(AUTODL_RUNS)).as_posix()
        if stage == "oipa":
            config["checkpoint"] = (model_root / "PE-Core-L14-336/PE-Core-L14-336.pt").as_posix()
            config["data"]["image_root"] = args.image_root.resolve().as_posix()
            for protocol in config["protocols"]:
                directory = args.data_root.resolve() / args.dataset / "formal" / protocol
                if not (directory / "manifest.json").is_file():
                    raise FileNotFoundError(directory / "manifest.json")
                config["protocols"][protocol] = directory.as_posix()
            for key in ("archive_plan", "human_review"):
                if key in config["diagnostics"]:
                    config["diagnostics"][key] = (REPO / config["diagnostics"][key]).as_posix()
        if stage == "oipa_backbones":
            for backbone in config["backbones"].values():
                backbone["directory"] = (model_root / Path(backbone["directory"]).name).as_posix()
        if stage == "oipa_itm":
            config["blip"] = (model_root / Path(config["blip"]).name).as_posix()
            config["published_visual_cache"] = {
                protocol: (run_root / "visual_cache" / protocol).as_posix()
                for protocol in config["published_visual_cache"]
            }
        output = config_root / f"{stage}.yaml"
        output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--model-root", type=Path, default=REPO / "artifacts")
    parser.add_argument("--run-root", required=True, type=Path)
    configure(parser.parse_args())
