# Third-party notices

The repository's CC BY-NC 4.0 license applies to original NEXUS contributions.
Third-party components and assets retain their own copyright and license terms.

## RSL-RL-derived modules

The following files retain their BSD-3-Clause license and copyright headers:

- src/perceptive_bfm_rl/algorithms/ppo.py
- src/perceptive_bfm_rl/algorithms/distillation.py

The complete upstream notice from rsl-rl-lib 5.4.0 is reproduced below.
The CC BY-NC license does not replace the BSD terms for these modules.
The `licenses/dependencies` path in the quoted notice refers to the upstream
[RSL-RL package](https://github.com/leggedrobotics/rsl_rl), not this repository.

```text
Copyright (c) 2026, ETH Zurich
Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

See licenses/dependencies for license information of dependencies of this package.
```

## InstinctMJ-derived camera and utility code

Parts of the grouped ray-casting camera, camera-noise pipeline and asynchronous
buffers are adapted from [InstinctMJ](https://github.com/project-instinct/InstinctMJ),
the mjlab-native port of [InstinctLab](https://github.com/project-instinct/InstinctLab).
We credit the contributors of both projects for these implementations.
The adapted portions are located in:

- `src/perceptive_bfm/sensors/grouped_ray_caster/`
- `src/perceptive_bfm/sensors/noisy_camera/`
- `src/perceptive_bfm/utils/noise/`
- `src/perceptive_bfm/utils/buffers/`

These implementations have been modified for NEXUS, including camera
configuration and visualization, ray output buffer handling and noise artifact
filtering.

The InstinctMJ and InstinctLab contributions retain their
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) terms.
See the [InstinctMJ license](https://github.com/project-instinct/InstinctMJ/blob/main/LICENSE)
and [InstinctLab license](https://github.com/project-instinct/InstinctLab/blob/main/LICENSE).
Third-party material incorporated by those projects retains its applicable
copyright and license terms.

## Robot assets

The bundled G1 meshes match the official
[Unitree G1 description](https://github.com/unitreerobotics/unitree_ros/tree/5994d4faef0a9cadd3287f8de0199a67eeb2a259/robots/g1_description).
The MJCF files are modified G1 models; their body and inertial definitions
match the [MuJoCo Menagerie G1 model](https://github.com/google-deepmind/mujoco_menagerie/tree/cda5236f98383f6fbaa4820c6b18dba892cd087f/unitree_g1),
which derives from Unitree's description. Local collision geometry, joint
settings and simulation configuration differ from the upstream model.

These resources retain their upstream BSD-3-Clause terms. The repository's
CC BY-NC license does not replace those terms. The applicable resources are:

- `src/perceptive_bfm/asset_zoo/robots/unitree_g1/xmls/`
- `terrain_motion_adaptation/src/terrain_motion_adaptation/assets/robots/unitree_g1/`

### Unitree description and meshes

[Upstream license](https://github.com/unitreerobotics/unitree_ros/blob/5994d4faef0a9cadd3287f8de0199a67eeb2a259/LICENSE):

```text
BSD 3-Clause License

Copyright (c) 2016-2022 HangZhou YuShu TECHNOLOGY CO.,LTD. ("Unitree Robotics")
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

### MuJoCo Menagerie G1 model

[Model license](https://github.com/google-deepmind/mujoco_menagerie/blob/cda5236f98383f6fbaa4820c6b18dba892cd087f/unitree_g1/LICENSE):

```text
Copyright (c) 2016-2023 HangZhou YuShu TECHNOLOGY CO.,LTD. ("Unitree Robotics")
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## External dependencies

Dependencies installed separately, including mjlab, MuJoCo and RSL-RL, retain
their own licenses. This document does not replace the license information
supplied with those packages.
