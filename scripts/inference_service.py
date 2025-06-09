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
from gr00t.experiment.data_config import ConfigGenerator
from gr00t.model.policy import Gr00tPolicy
from gr00t.eval.robot import RobotInferenceServer
from gr00t.data.transform.base import ComposedModalityTransform
from typing import cast


def main():
    parser = argparse.ArgumentParser("Gr00t Inference Server")
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Local path or HF repo ID of the pretrained model"
    )
    parser.add_argument(
        "--embodiment_tag", type=str, default="new_embodiment",
        help="Embodiment tag defined in metadata.json"
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
    # Validate model path or HF repo ID
    if not os.path.exists(args.model_path) and '://' not in args.model_path:
        raise FileNotFoundError(
            f"Model path '{args.model_path}' not found locally or as HuggingFace repo."
        )

    # Create data config and transforms
    data_gen = ConfigGenerator(num_arms=args.num_arms, num_cams=args.num_cams)
    modality_cfg = data_gen.modality_config()
    # cast to expected ComposedModalityTransform
    modality_transform = cast(ComposedModalityTransform, data_gen.transform())

    # Instantiate policy
    policy = Gr00tPolicy(
        model_path=args.model_path,
        modality_config=modality_cfg,
        modality_transform=modality_transform,
        embodiment_tag=args.embodiment_tag,
        denoising_steps=args.denoising_steps,
    )
    print(f"Loaded Gr00tPolicy from {args.model_path} [embodiment={args.embodiment_tag}] with {args.denoising_steps} steps.")

    # Start inference server
    server = RobotInferenceServer(policy, host=args.host, port=args.port)
    print(f"Starting Gr00t server at {args.host}:{args.port}")
    server.run()


if __name__ == '__main__':
    main()
