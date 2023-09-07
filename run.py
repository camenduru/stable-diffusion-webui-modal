import modal
import os

stub = modal.Stub("stable-diffusion-webui")
volume = modal.NetworkFileSystem.new().persisted("stable-diffusion-webui")

@stub.function(
    modal.Image.from_registry("nvidia/cuda:12.2.0-base-ubuntu22.04", add_python="3.11")
    .run_commands(
        "apt-get update -y && \
        apt-get install -y git git-lfs aria2 libgl1 libglib2.0-0 wget && \
        pip install -q torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 torchtext==0.15.2 torchdata==0.6.1 --extra-index-url https://download.pytorch.org/whl/cu118 && \
        pip install -q xformers==0.0.20 triton==2.0.0 packaging==23.1"
    ),
    network_file_systems={"/content/stable-diffusion-webui": volume},
    gpu="T4",
    timeout=60000,
)
async def run():
    os.system(f"git clone -b v2.6 https://github.com/camenduru/stable-diffusion-webui /content/stable-diffusion-webui")
    os.chdir(f"/content/stable-diffusion-webui")
    os.system(f"git lfs install")
    os.system(f"git reset --hard")
    os.system(f"sed -i 's/--refetch //g' /content/stable-diffusion-webui/modules/launch_utils.py")
    os.system(f"aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/counterfeit-xl/resolve/main/counterfeitxl_v10.safetensors -d /content/stable-diffusion-webui/models/Stable-diffusion -o counterfeitxl_v10.safetensors")
    os.system(f"python launch.py --cors-allow-origins=* --xformers --theme dark --gradio-queue")

@stub.local_entrypoint()
def main():
    run.call()