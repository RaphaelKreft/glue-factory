from pathlib import Path

cluster_team_folder = Path("/cluster/courses/3dv/data/team-2")  # cluster team folder for 3dv
root = Path(__file__).parent.parent  # top-level directory
DATA_PATH = cluster_team_folder  # datasets and pretrained weights
TRAINING_PATH = root / "outputs/training/"  # training checkpoints
EVAL_PATH = cluster_team_folder / "outputs/results/"  # evaluation results
