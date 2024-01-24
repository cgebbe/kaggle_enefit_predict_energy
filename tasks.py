from invoke import task

COMPETITION_NAME = "predict-energy-behavior-of-prosumers"
SCRIPT_DIRNAME = "enefitscripts"
MODEL_DIRNAME = "enefitmodel"


@task
def dsup(ctx, message: str = "'new version'"):
    """Upload datasets.

    NOTE:
    - first `kaggle datasets init` to create metadata
    - then `kaggle datasets create` for first time
    - only then version to update

    TODO:
    - delete __pycache__ folder
    - for reproducibility: insert datetime, commit and git-diff ?! into it
    - when importing enefitscripts, print reproducibility information!
    """
    # TODO: construct message from datetime and commit-hash
    ctx.run(
        f"kaggle datasets version -m {message} --path {SCRIPT_DIRNAME} --dir-mode=zip"
    )
    ctx.run(
        f"kaggle datasets version -m {message} --path {MODEL_DIRNAME} --dir-mode=zip"
    )
