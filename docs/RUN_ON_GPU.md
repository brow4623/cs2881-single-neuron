# Running the models on a rented GPU (RunPod)

The full Figure 2 run (Qwen3-1.7B, Qwen3-4B, Qwen3-8B, Llama-3.1-8B, plus the Llama-Guard-3-8B
judge) needs one GPU with at least 40 GB of memory. This is the recipe we used on 2026-10-15:
one A100 SXM 80 GB (secure cloud, $1.79/hr). It took about an hour.

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

## 3. Get the code and install (on the pod)

SSH in (`ssh -i ~/.runpod/ssh/runpodctl-ssh-key root@<ip> -p <port>`), then clone the repo. The
image's system Python refuses `pip install` (PEP 668), so use a venv that reuses the image's CUDA
PyTorch:

```bash
git clone https://github.com/brow4623/cs2881-single-neuron.git /workspace/repo
python -m venv --system-site-packages /root/venv
/root/venv/bin/pip install -e "/workspace/repo[dev]"
export HF_HOME=/workspace/hf-cache PATH=/root/venv/bin:$PATH   # model downloads go on the 100 GB volume
hf auth login            # needs access to the gated Llama repos; "Log in with your browser" works remotely
cd /workspace/repo && python scripts/check_setup.py
```

## 4. Run (inside tmux so a dropped connection doesn't kill it)

```bash
tmux new -s fig2
mkdir -p results/scratch
bash scripts/run_all.sh 2>&1 | tee results/scratch/gpu_run.log
```

With no arguments, `run_all.sh` does all four models. For each it prints a sanity-check PASS,
generates baseline and pinned responses, then runs Llama-Guard-3-8B on all of them and draws the
figure. The Claude judge is skipped unless `.env` and `prompts/llm_judge_k1.txt` are on the pod. It
can run later from a laptop, since it needs no GPU (see the README). Rerunning skips finished work.

## 5. Bring results back and delete the pod (from the laptop)

```bash
rsync -az --no-owner --no-group -e "ssh -i ~/.runpod/ssh/runpodctl-ssh-key -p <port>" root@<ip>:/workspace/repo/results/ ./results/
runpodctl pod delete <pod-id>      # billing stops; the volume and downloaded models are deleted too
```

Model responses (`results/generations/`) contain jailbroken text and are gitignored. Keep them
local rather than committing them.
