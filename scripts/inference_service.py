# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python3
import os
import argparse
import json
from pathlib import Path
from gr00t.experiment.data_config import ConfigGenerator
from gr00t.model.policy import Gr00tPolicy
from gr00t.eval.robot import RobotInferenceServer
from gr00t.data.transform.base import ComposedModalityTransform
from typing import cast

def is_huggingface_repo(path: str) -> bool:
    """Check if the path looks like a HuggingFace repository ID."""
    # HuggingFace repos typically have format: username/repo-name
    # They don't start with ./ or / and contain at least one /
    return ('/' in path and 
            not path.startswith('./') and 
            not path.startswith('/') and
            not os.path.exists(path))  # Not a local directory

def detect_model_config(model_path: str, default_num_cams: int = 2) -> tuple[int, int, list[str]]:
    """
    Auto-detect number of arms, cameras, and video keys from model metadata.
    Returns (num_arms, num_cams, video_keys)
    """
    try:
        # Align with policy._load_metadata: metadata.json is under experiment_cfg
        if is_huggingface_repo(model_path):
            from huggingface_hub import snapshot_download
            snapshot = snapshot_download(repo_id=model_path, repo_type="model")
            metadata_path = Path(snapshot) / "experiment_cfg" / "metadata.json"
        else:
            metadata_path = Path(model_path) / "experiment_cfg" / "metadata.json"
        
        metadata = json.loads(metadata_path.read_text())
        # Extract video keys
        video_keys = list(metadata.get('modalities', {}).get('video', {}).keys())
        num_cams = len(video_keys)
        action_keys = list(metadata.get('modalities', {}).get('action', {}).keys())
        num_arms = len([k for k in action_keys if k.startswith('arm')]) or 1
        return num_arms, num_cams, video_keys
    except Exception:
        return 1, default_num_cams, []  # fallback

def main():
    parser = argparse.ArgumentParser("Gr00t Inference Server")
    parser.add_argument(
        "--model_path", type=str, default="nvidia/GR00T-N1-2B",
        help="Local path or HF repo ID of the pretrained model"
    )
    parser.add_argument(
        "--embodiment_tag", type=str, default=None,
        help="Embodiment tag defined in metadata.json (auto-detect if None)"
    )
    parser.add_argument(
        "--num_arms", type=int, default=1,
        help="Number of robot arms in your embodiment"
    )
    parser.add_argument(
        "--num_cams", type=int, default=2,
        help="Number of cameras in your setup"
    )
    parser.add_argument(
        "--denoising_steps", type=int, default=4,
        help="Number of diffusion denoising steps"
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0",
        help="Server bind address"
    )
    parser.add_argument(
        "--port", type=int, default=5555,
        help="Server port"
    )
    args = parser.parse_args()
    
    # Set HF_HOME if not set
    if not os.environ.get("HF_HOME"):
        os.environ["HF_HOME"] = "/persistent/huggingface_cache"
    
    print(f"Using model: {args.model_path}")
    print(f"HF_HOME: {os.environ.get('HF_HOME')}")
    
    # Validate model path - distinguish between HuggingFace repos and local paths
    def is_huggingface_repo(path: str) -> bool:
        """Check if the path looks like a HuggingFace repository ID."""
        # HuggingFace repos typically have format: username/repo-name
        # They don't start with ./ or / and contain at least one /
        return ('/' in path and 
                not path.startswith('./') and 
                not path.startswith('/') and
                not os.path.exists(path))  # Not a local directory
    
    def validate_huggingface_repo(repo_id: str) -> bool:
        """Validate that a HuggingFace repository exists and is accessible."""
        try:
            from huggingface_hub import repo_exists
            return repo_exists(repo_id, repo_type="model")
        except Exception as e:
            print(f"Warning: Could not validate HuggingFace repo '{repo_id}': {e}")
            return True  # Assume valid and let policy handle any errors
    
    if is_huggingface_repo(args.model_path):
        print(f"Detected HuggingFace repo: {args.model_path}")
        if not validate_huggingface_repo(args.model_path):
            raise FileNotFoundError(f"HuggingFace repository '{args.model_path}' not found or not accessible.")
    elif not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Local model path '{args.model_path}' not found.")
    
    # Auto-detect camera configuration from model metadata
    num_arms, num_cams, video_keys = detect_model_config(args.model_path, args.num_cams)

    # Use detected configuration to override defaults
    args.num_arms = num_arms
    args.num_cams = num_cams

    # Create data config using metadata if available
    if video_keys:
        # Use explicit names generator to match metadata
        prefixed_video_keys = [f"video.{k}" if not k.startswith('video.') else k for k in video_keys]
        state_keys = [f"state.arm_{i}" for i in range(args.num_arms)]
        action_keys = [f"action.arm_{i}" for i in range(args.num_arms)]
        print(f"Creating ConfigGeneratorFromNames with video_keys={prefixed_video_keys}, state_keys={state_keys}, action_keys={action_keys}")
        from gr00t.experiment.data_config import ConfigGeneratorFromNames
        data_gen = ConfigGeneratorFromNames(prefixed_video_keys, state_keys, action_keys)
    else:
        data_gen = ConfigGenerator(num_arms=args.num_arms, num_cams=args.num_cams)
    