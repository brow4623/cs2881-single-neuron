# Running the remaining models on a rented GPU (RunPod)

Qwen3-1.7B generations are already in `results/generations/`. The GPU run covers Qwen3-4B,
Qwen3-8B, Llama-3.1-8B and the Llama-Guard-3-8B judge. This is the recipe that worked on
2026-10-15: one A100 SXM 80 GB (secure cloud, $1.79/hr), about an hour.

## 1. One-time setup on your laptop

```bash
brew install runpod/runpodctl/runpodctl
runpodctl doctor        # paste a RunPod API key (Settings > API Keys); it also creates and registers an SSH key
```

Have about $10 of credit on the RunPod account.

## 2. Create the pod

```bash
runpodctl gpu list      # what's in stock and the hourly price; L40S / A40 / A6000 / A100 all work
runpodctl pod create --name cs2881-fig2 --gpu-id "NVIDIA A100-SXM4-80GB" --cloud-type SECURE \
  --image runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404 \
  --container-disk-in-gb 40 --volume-in-gb 100 --ports 22/tcp
runpodctl ssh info <pod-id>   # prints the ssh command (IP, port, key)
```

`--ports 22/tcp` is required for SSH with a custom image. If you forget it, add it afterwards with
`runpodctl pod update <pod-id> --ports 22/tcp`. Stock changes quickly: if `pod create` says no
instances are available, pick another GPU from `gpu list`.

## 3. Copy the code and install (from the laptop)

The repo is private, so copy it instead of cloning (the `chown ... not permitted` warnings are harmless):

```bash
rsync -az --exclude .venv --exclude .env -e "ssh -i ~/.runpod/ssh/runpodctl-ssh-key -p <port>" ./ root@<ip>:/workspace/repo/
```

Then on the pod (`ssh -i ~/.runpod/ssh/runpodctl-ssh-key root@<ip> -p <port>`). The image's
system Python refuses `pip install` (PEP 668), so use a venv that reuses the image's CUDA PyTorch:

```bash
python -m venv --system-site-packages /root/venv
/root/venv/bin/pip install -e "/workspace/repo[dev]"
export HF_HOME=/workspace/hf-cache PATH=/root/venv/bin:$PATH   # model downloads go on the 100 GB volume
hf auth login            # "Log in with your browser" works from the remote shell
cd /workspace/repo && python scripts/check_setup.py
```

## 4. Run (inside tmux so a dropped connection doesn't kill it)

```bash
tmux new -s fig2
bash scripts/run_all.sh qwen3-4b qwen3-8b llama-3.1-8b 2>&1 | tee results/scratch/gpu_run.log
```

This prints a sanity-check PASS per model, generates baseline and pinned responses, runs
Llama-Guard-3-8B on **all** generation files (including the Qwen3-1.7B ones), and draws the figure.
The Claude judge is skipped unless `.env` and `prompts/llm_judge_k1.txt` are on the pod; it can run
later from a laptop, since it needs no GPU. Rerunning skips finished work.

## 5. Bring results back and delete the pod

```bash
rsync -az -e "ssh -i ~/.runpod/ssh/runpodctl-ssh-key -p <port>" root@<ip>:/workspace/repo/results/ ./results/
runpodctl pod delete <pod-id>      # billing stops; the volume and downloaded models are deleted too
```
