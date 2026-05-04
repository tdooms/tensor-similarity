import os


def push_run_to_hf(run_dir, repo_id: str, private: bool = True):
    token = os.environ.get("HF_TOKEN")
    if not token or not repo_id:
        return

    from huggingface_hub import create_repo, upload_folder

    create_repo(repo_id, repo_type="model", private=private, exist_ok=True, token=token)
    upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(run_dir),
        token=token,
        ignore_patterns=["wandb/*", "errors/*", "**/__pycache__/*"],
        commit_message=f"Upload run {run_dir.name}",
    )
