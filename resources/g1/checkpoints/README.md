# Pretrained Perceptive Behavior Foundation Model

We provide a pretrained perceptive behavior foundation model for Unitree G1 in
ONNX format. The model combines motion references, proprioceptive history, and
depth observations for terrain-adaptive whole-body control.

[ONNX model](policy.onnx)

## Interface

The model uses ONNX opset 18 and a fixed batch size of one. All tensors are
`float32`.

| Direction | Name | Shape | Contents |
| --- | --- | --- | --- |
| Input | `obs` | `(1, 1473)` | Motion command and proprioceptive histories |
| Input | `camera` | `(1, 1, 18, 32)` | Processed depth image in NCHW order |
| Output | `actions` | `(1, 29)` | Joint action values |

[`config.json`](config.json) contains the joint order, PD gains, default
pose, action scales, motion-reference offsets, history length and depth settings.
The policy runs at 50 Hz (`controlDt = 0.02` seconds). Clip actions to
`[-10, 10]` and compute joint targets as:

```text
joint_target = defaultPose + actionScale * clipped_action
```

Observation construction, depth preprocessing and control are performed outside
the ONNX graph. Inputs must match the policy's observation ordering and history
handling; tensor shapes alone do not define compatible inputs. Apply depth
clipping, cropping, resizing and normalization using the camera configuration.
