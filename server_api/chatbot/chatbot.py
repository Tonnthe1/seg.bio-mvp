import os
import subprocess
import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import Tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver

# --------------------------------------------------------------------------- #
# Paths and configuration
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(os.getenv("PYTC_PROJECT_ROOT", "/home/zhangdjr/projects/seg-bio/pytorch_connectomics"))
CONFIG_DIR = Path(os.getenv("PYTC_CONFIG_DIR", PROJECT_ROOT / "tutorials"))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://cscigpu08.bc.edu:11434")
CHATBOT_MODEL = os.getenv("CHATBOT_MODEL", "gpt-oss:20b")

# Explicit config list so the agent can describe options clearly
CONFIG_FILES = [
    "monai_lucchi++.yaml",
    "monai_nucmm-z.yaml",
    "monai_fiber.yaml",
    "monai_hydra-bv.yaml",
    "monai_bouton-bv.yaml",
    "monai2d_worm.yaml",
    "mednext2d_cem-mitolab.yaml",
    "mednext_mitoEM.yaml",
    "rsunet_snemi.yaml",
]

CONFIG_DESCRIPTIONS = {
    "monai_lucchi++.yaml": "MONAI UNet for mitochondria segmentation on Lucchi++ (165×1024×1024). Best for beginners.",
    "monai_nucmm-z.yaml": "MONAI UNet for nucleus segmentation with multi-task learning (binary + contour + distance).",
    "monai_fiber.yaml": "MONAI UNet for fiber/axon segmentation with anisotropic patches.",
    "monai_hydra-bv.yaml": "MONAI UNet for large vesicle segmentation with valid region masks.",
    "monai_bouton-bv.yaml": "MONAI UNet for synaptic bouton segmentation.",
    "monai2d_worm.yaml": "2D MONAI UNet for slice-by-slice segmentation.",
    "mednext2d_cem-mitolab.yaml": "2D MedNeXt (state-of-the-art) for mitochondria on 21K images.",
    "mednext_mitoEM.yaml": "3D MedNeXt (state-of-the-art) for mitochondria segmentation.",
    "rsunet_snemi.yaml": "RSUNet for neuron instance segmentation (award-winning architecture).",
}


# --------------------------------------------------------------------------- #
# Training agent tools
# --------------------------------------------------------------------------- #

def list_training_configs() -> List[Dict[str, str]]:
    """List all available training configuration files with descriptions."""
    configs: List[Dict[str, str]] = []
    for filename in CONFIG_FILES:
        yaml_file = CONFIG_DIR / filename
        try:
            with open(yaml_file, "r") as f:
                config_data = yaml.safe_load(f)
            configs.append(
                {
                    "name": filename,
                    "path": str(yaml_file),
                    "model": config_data.get("model", {}).get("architecture", "unknown"),
                    "experiment_name": config_data.get("experiment_name", "unknown"),
                    "description": CONFIG_DESCRIPTIONS.get(filename, ""),
                }
            )
        except Exception:
            # If a file cannot be parsed, skip it but keep the rest available
            continue
    return configs


def read_config(yaml_path: str) -> Dict[str, Any]:
    """Read and parse a YAML configuration file."""
    yaml_path_obj = Path(yaml_path)
    if not yaml_path_obj.is_absolute():
        yaml_path_obj = PROJECT_ROOT / yaml_path_obj

    if not yaml_path_obj.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path_obj}")

    with open(yaml_path_obj, "r") as f:
        config = yaml.safe_load(f)
    return config


def submit_training_job(
    config_path: str,
    overrides: Optional[List[str]] = None,
    job_name: str = "sample",
) -> Dict[str, str]:
    """Submit a PyTorch Connectomics training job to SLURM."""
    working_dir = str(PROJECT_ROOT)
    config = read_config(config_path)
    experiment_name = config.get("experiment_name", "unknown")

    override_str = " ".join(overrides) if overrides else ""
    training_cmd = f"python scripts/main.py --config {config_path}"
    if override_str:
        training_cmd += f" {override_str}"

    script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1 --cpus-per-task=4
#SBATCH --partition=medium
#SBATCH --mem-per-cpu=8g
#SBATCH --time=40:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=zhangdjr@bc.edu

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pytc
cd {working_dir}

{training_cmd}
"""

    timestamp = int(time.time())
    script_path = Path(f"/tmp/{job_name}_{timestamp}.sl")
    script_path.write_text(script_content)

    result = subprocess.run(
        ["sbatch", str(script_path)],
        capture_output=True,
        text=True,
        cwd=working_dir,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Failed to submit job: {result.stderr}")

    job_id = result.stdout.strip().split()[-1]

    return {
        "job_id": job_id,
        "job_name": job_name,
        "config_path": config_path,
        "experiment_name": experiment_name,
        "overrides": overrides if overrides else [],
        "slurm_script": str(script_path),
        "output_log": f"{job_name}_{job_id}.out",
        "error_log": f"{job_name}_{job_id}.err",
        "working_dir": working_dir,
        "status": "submitted",
        "submitted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def check_training_progress(job_id: str, job_name: str = "sample") -> Dict[str, Any]:
    """Check SLURM training job status and retrieve full log output."""
    working_dir = str(PROJECT_ROOT)

    squeue_result = subprocess.run(
        ["squeue", "-j", job_id, "--format=%T", "--noheader"],
        capture_output=True,
        text=True,
    )

    if squeue_result.stdout.strip():
        job_status = squeue_result.stdout.strip()
    else:
        sacct_result = subprocess.run(
            ["sacct", "-j", job_id, "--format=State", "--noheader"],
            capture_output=True,
            text=True,
        )
        job_status = sacct_result.stdout.strip().split()[0] if sacct_result.stdout.strip() else "UNKNOWN"

    log_file = Path(working_dir) / f"{job_name}_{job_id}.out"
    log_content = log_file.read_text() if log_file.exists() else "Log file not found"

    return {
        "job_id": job_id,
        "job_status": job_status,
        "log_content": log_content,
    }


# --------------------------------------------------------------------------- #
# Inference agent tools
# --------------------------------------------------------------------------- #

def list_checkpoints(experiment_name: Optional[str] = None) -> List[Dict[str, str]]:
    """List available trained model checkpoints."""
    outputs_dir = PROJECT_ROOT / "outputs"

    if not outputs_dir.exists():
        return []

    checkpoints = []
    search_dirs = [outputs_dir / experiment_name] if experiment_name else [d for d in outputs_dir.iterdir() if d.is_dir()]

    for exp_dir in search_dirs:
        if not exp_dir.is_dir():
            continue

        for run_dir in exp_dir.iterdir():
            if not run_dir.is_dir():
                continue

            checkpoint_dir = run_dir / "checkpoints"
            if not checkpoint_dir.exists():
                continue

            for ckpt_file in checkpoint_dir.glob("*.ckpt"):
                checkpoints.append(
                    {
                        "experiment_name": exp_dir.name,
                        "run_dir": str(run_dir),
                        "checkpoint_path": str(ckpt_file),
                        "checkpoint_name": ckpt_file.name,
                        "modified_time": datetime.fromtimestamp(ckpt_file.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )

    checkpoints.sort(key=lambda x: x["modified_time"], reverse=True)
    return checkpoints


def run_inference(
    checkpoint_path: str,
    test_image_path: str,
    test_label_path: Optional[str] = None,
    output_path: Optional[str] = None,
    job_name: str = "inference",
    enable_tta: bool = True,
    overrides: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Submit SLURM job for inference or evaluation on test data."""
    working_dir = PROJECT_ROOT
    mode = "test" if test_label_path else "predict"

    checkpoint_path_obj = Path(checkpoint_path)
    run_dir = checkpoint_path_obj.parent.parent
    config_path = run_dir / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found at {config_path}")

    overrides_list = [
        f"inference.data.test_image={test_image_path}",
        f"inference.test_time_augmentation.enabled={str(enable_tta).lower()}",
    ]

    if test_label_path:
        overrides_list.append(f"inference.data.test_label={test_label_path}")
    if output_path:
        overrides_list.append(f"inference.data.output_path={output_path}")
    if overrides:
        overrides_list.extend(overrides)

    overrides_str = " ".join(overrides_list)
    train_cmd = f"python scripts/main.py --config {config_path} --mode {mode} --checkpoint {checkpoint_path} {overrides_str}"

    slurm_script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1 --cpus-per-task=4
#SBATCH --partition=medium
#SBATCH --mem-per-cpu=4g
#SBATCH --time=10:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=zhangdjr@bc.edu

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pytc
cd {working_dir}

{train_cmd}
"""

    import tempfile

    script_fd, script_path = tempfile.mkstemp(suffix=".sl", prefix=f"{job_name}_", text=True)
    with open(script_fd, "w") as f:
        f.write(slurm_script_content)

    result = subprocess.run(
        ["sbatch", script_path],
        cwd=str(working_dir),
        capture_output=True,
        text=True,
        check=True,
    )

    job_id = result.stdout.strip().split()[-1]

    return {
        "job_id": job_id,
        "job_name": job_name,
        "mode": mode,
        "checkpoint_path": checkpoint_path,
        "test_image_path": test_image_path,
        "test_label_path": test_label_path,
        "output_path": output_path,
        "slurm_script": script_path,
        "output_log": f"{job_name}_{job_id}.out",
        "error_log": f"{job_name}_{job_id}.err",
        "working_dir": str(working_dir),
        "status": "submitted",
        "submitted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def check_inference_progress(
    job_id: str,
    working_dir: str = str(PROJECT_ROOT),
    job_name: str = "inference",
) -> Dict[str, Any]:
    """Check inference/evaluation job status and retrieve log output."""
    squeue_result = subprocess.run(
        ["squeue", "-j", job_id, "--format=%T", "--noheader"],
        capture_output=True,
        text=True,
    )

    if squeue_result.stdout.strip():
        job_status = squeue_result.stdout.strip()
    else:
        sacct_result = subprocess.run(
            ["sacct", "-j", job_id, "--format=State", "--noheader"],
            capture_output=True,
            text=True,
        )
        job_status = sacct_result.stdout.strip().split()[0] if sacct_result.stdout.strip() else "UNKNOWN"

    log_file = Path(working_dir) / f"{job_name}_{job_id}.out"
    log_content = log_file.read_text() if log_file.exists() else "Log file not found"

    return {
        "job_id": job_id,
        "job_status": job_status,
        "log_content": log_content,
    }


# --------------------------------------------------------------------------- #
# Agent wiring
# --------------------------------------------------------------------------- #

def _build_training_agent(llm: ChatOllama):
    trainer_prompt = """
You are a Biomedical Image Segmentation Training Agent. You help train segmentation models.

Tools:
- list_training_configs: List available training configs
- read_config: Read a config file
- submit_training_job: Submit a training job
- check_training_progress: Check job progress

CRITICAL - Valid Config Override Format:
When using submit_training_job, the 'overrides' parameter MUST be a list of strings in this exact format:
- "optimization.max_epochs=10"
- "system.training.batch_size=4"
- "optimization.optimizer.lr=0.001"
- "experiment_name=my_experiment"

You can override ANY key that exists in the config file. Just use the exact dotted path from the YAML.

NEVER use keys like 'trainer', 'epochs', or other keys that don't exist in the config schema.

Common valid override keys (but you can use ANY key from the config):
- optimization.max_epochs (number of training epochs)
- optimization.optimizer.lr (learning rate)
- optimization.optimizer.weight_decay (weight decay)
- system.training.batch_size (batch size)
- data.iter_num_per_epoch (iterations per epoch)
- experiment_name (name of experiment)
- model.dropout (dropout rate)
- Any other key that exists in the config YAML

Instructions:
- Choose the best config for the user's needs
- Read the config first to see what keys are available
- Use ONLY valid config keys in overrides
- Ask for clarification if unclear
- When asked about job status/progress, ALWAYS use check_training_progress tool - don't just describe what to do
- CRITICAL: When calling check_training_progress, you MUST pass BOTH job_id AND job_name from the submit_training_job response
- The job_name is needed to locate the correct log file (format: <job_name>_<job_id>.out)
- Use tools when they're relevant to the task - but you can answer simple questions directly
"""

    tools = [
        Tool.from_function(list_training_configs, name="list_training_configs", description="List available training configs"),
        Tool.from_function(read_config, name="read_config", description="Read a config file"),
        Tool.from_function(submit_training_job, name="submit_training_job", description="Submit a training job"),
        Tool.from_function(check_training_progress, name="check_training_progress", description="Check training job progress"),
    ]
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", trainer_prompt),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=False)


def _build_inference_agent(llm: ChatOllama):
    inference_prompt = """
You are a Biomedical Image Segmentation Inference/Evaluation Agent. You run inference and evaluation.

Tools:
- list_checkpoints: List available checkpoints
- run_inference: Submit inference/evaluation job
- check_inference_progress: Check job progress
- read_config: Read config to get test data paths

Instructions:
- List checkpoints first
- CRITICAL: To read the config for a checkpoint, construct the config path from the checkpoint path:
  * Checkpoint structure: <experiment_dir>/<timestamp>/checkpoints/<checkpoint_file>
  * Config location: <experiment_dir>/<timestamp>/config.yaml
  * Example: If checkpoint is "outputs/monai_lucchi++/20251130_173307/checkpoints/last.ckpt"
            Then config is "outputs/monai_lucchi++/20251130_173307/config.yaml"
  * Algorithm: Go up 2 directories from checkpoint file, then append "config.yaml"
- Read config to extract test data paths (inference.data.test_image and inference.data.test_label)
- Submit inference/evaluation jobs with the extracted test paths
- When asked about job status/progress, use check_inference_progress tool - don't just describe what to do
- Use tools when they're relevant to the task - but you can answer simple questions directly
"""

    tools = [
        Tool.from_function(list_checkpoints, name="list_checkpoints", description="List available checkpoints"),
        Tool.from_function(run_inference, name="run_inference", description="Submit inference/evaluation job"),
        Tool.from_function(check_inference_progress, name="check_inference_progress", description="Check inference job progress"),
        Tool.from_function(read_config, name="read_config", description="Read a config file"),
    ]
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", inference_prompt),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=False)


def _build_supervisor(llm: ChatOllama):
    training_agent = _build_training_agent(llm)
    inference_agent = _build_inference_agent(llm)

    def delegate_to_training_agent(task: str) -> str:
        result = training_agent.invoke({"input": task})
        return result.get("output", "")

    def delegate_to_inference_agent(task: str) -> str:
        result = inference_agent.invoke({"input": task})
        return result.get("output", "")

    supervisor_prompt = """
You are a Biomedical Image Segmentation Supervisor Agent. You coordinate training and inference.

Sub-agents:
1. Training Agent: Handles training
2. Inference Agent: Handles inference/evaluation

Tools:
- delegate_to_training_agent: Delegate training tasks
- delegate_to_inference_agent: Delegate inference tasks

Instructions:
- Understand user requests
- Delegate to appropriate agent when the task requires training or inference tools
- Pass context between agents
- Provide clear summaries
- For simple questions (like math or general info), answer directly without delegating
- For job status/progress questions, delegate to the appropriate agent to use their monitoring tools
"""

    memory = MemorySaver()
    tools = [
        Tool.from_function(delegate_to_training_agent, name="delegate_to_training_agent", description="Delegate to training agent"),
        Tool.from_function(delegate_to_inference_agent, name="delegate_to_inference_agent", description="Delegate to inference agent"),
    ]
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", supervisor_prompt),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    agent = create_tool_calling_agent(llm, tools, prompt)
    supervisor = AgentExecutor(agent=agent, tools=tools, verbose=False)

    return supervisor, memory


@lru_cache(maxsize=1)
def get_supervisor_bundle():
    """Initialize and cache the supervisor + memory."""
    llm = ChatOllama(model=CHATBOT_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
    return _build_supervisor(llm)


def handle_query(prompt: str, thread_id: str) -> Dict[str, str]:
    """
    Route a user prompt through the supervisor agent and return the response.

    The caller controls the thread_id so UI clients can keep independent
    conversations separated (e.g., per browser tab).
    """
    supervisor, _memory = get_supervisor_bundle()
    try:
        result = supervisor.invoke(
            {"input": prompt},
            config={"configurable": {"thread_id": thread_id}},
        )
        print(f"[CHATBOT] supervisor.invoke result type={type(result)} value={result}")
        response = result.get("output", "") if isinstance(result, dict) else str(result)
        return {"response": response, "thread_id": thread_id}
    except Exception as exc:
        import traceback
        err_msg = f"[CHATBOT] Error during handle_query: {exc}"
        print(err_msg)
        print(traceback.format_exc())
        try:
            with open("/tmp/chatbot_error.log", "a") as f:
                f.write(err_msg + "\n")
                f.write(traceback.format_exc())
                f.write("\n")
        except Exception:
            pass
        # Try a direct LLM response as a fallback so UI still gets an answer
        try:
            llm = ChatOllama(model=CHATBOT_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
            llm_response = llm.invoke(prompt)
            text = getattr(llm_response, "content", str(llm_response))
            return {"response": text, "thread_id": thread_id}
        except Exception as llm_exc:
            print("[CHATBOT] Direct LLM fallback failed:", llm_exc)
            return {
                "response": "Sorry, the agent is currently unavailable.",
                "thread_id": thread_id,
            }


def reset_chat(thread_id: Optional[str] = None) -> None:
    """
    Clear chat memory. If a thread_id is provided, a full reset is still used
    because the in-memory checkpointer does not expose per-thread deletion.
    """
    # Rebuild the supervisor + checkpointer to drop all sessions.
    get_supervisor_bundle.cache_clear()
    get_supervisor_bundle()
