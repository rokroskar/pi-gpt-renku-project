# Agent handoff: Renku MNIST training dashboard

This file captures operational context for future coding/Renku agents working on this repository.

## Project and repository

- Renku project: <https://dev.renku.ch/p/rokroskar/mnist-non-interactive-training-job>
- GitHub repository: <https://github.com/rokroskar/pi-gpt-renku-project>
- Current known good repo commit: `e3420d696428820de63f739e73915a7c2ff6fd84`

## Renku helper

Use the Renku skill/helper from the local skill checkout:

```bash
export RENKU_BASE_URL=https://dev.renku.ch
HELPER=/Users/rok/.pi/agent/git/github.com/rokroskar/renku-agent-skill/skills/renku/scripts/renku_agent.py
```

Before any Renku operation, verify the account is not admin:

```bash
$HELPER user
```

If `is_admin: true`, do not perform Renku operations except logout/explanation.

Ask before stopping/deleting active sessions or deleting/unlinking launchers/connectors.

## Important Renku IDs

- Project: `01KSNRFKT2WYD06YXT25B3WSWM`
- Build environment: `01KSNRHVFK8ND7NGPPNEZN1138`
- Build launcher: `01KSNRHVFK2ZZ25HWNMSKB5F38`
- Training/artifact job launcher: `01KSNRP4T8TSSB1GYKV1R20QP7`
- Dashboard launcher: `01KSSGCZ665NA9W020BP5FME7G`
- Dashboard environment: `01KSSGCZ66FV774GMM0J2ARBTR`
- MNIST Zenodo DOI connector: `01KSMP27C2MBAFWNH6RQP2CB9C`
- Private writable model-artifacts connector: `01KSSFN8PDX4D2RYEQ32TF779K`
- Public read-only pretrained-artifacts connector: `01KSSR8WFH5HSRQR6AA5E49BHZ`

## Important paths inside Renku sessions/jobs

- MNIST data connector:
  `/home/renku/work/mnist-dataset-doi-10.5281-zenodo.10058130`
- Training job writable output connector:
  `/home/renku/work/model-artifacts`
- Training job model directory:
  `/home/renku/work/model-artifacts/mnist-models`
- Dashboard public read-only pretrained model directory:
  `/home/renku/work/pretrained-model-artifacts/mnist-models`
- Dashboard interactive retraining output directory:
  `/home/renku/work/dashboard-trained-models`
- Runtime-cloned source repository:
  `/home/renku/work/pi-gpt-renku-project`

## Current launchers

The project should have three useful launchers:

1. **Build Python environment from repository**
   - Builds dependencies from `requirements.txt`.
   - Rebuild only when dependencies change.

2. **Prepare MNIST model artifacts job**
   - Non-interactive job launcher.
   - Runs `prepare_model_artifacts.py`.
   - Trains both NumPy MLP and PyTorch CNN.
   - Writes to `/home/renku/work/model-artifacts/mnist-models`.
   - Calls `os.sync()` and waits for rclone/output-connector sync.

3. **MNIST Gradio model dashboard**
   - Interactive Gradio launcher.
   - Runs dashboard from runtime-cloned source:
     `/home/renku/work/pi-gpt-renku-project/dashboard.py`
   - Reads public pretrained models by default from:
     `/home/renku/work/pretrained-model-artifacts/mnist-models`
   - Can retrain interactively into:
     `/home/renku/work/dashboard-trained-models`

## Data and artifact policy

- Do **not** download MNIST in Renku jobs/sessions for the project workflow.
- Use the mounted Zenodo DOI connector for MNIST.
- Keep training job writes on the private writable Polybox connector.
- Keep dashboard default reads on the public read-only pretrained Polybox connector.
- Dashboard-triggered retraining should write only to a user/session-writable path, not the public read-only connector.

## Known successful training results

A successful artifact job trained both models using mounted MNIST data only:

- NumPy MLP best test accuracy: `0.9857`
- PyTorch CNN test accuracy: `0.9923`
- Train examples: `60000`
- Test examples: `10000`

Expected artifacts:

- `mnist-mlp-model.npz`
- `mnist-small-cnn.pt`
- `_ARTIFACTS_READY.json`

## Dashboard status

Last verified dashboard session:

- Session name: `rokroskar-0fc6812148c0`
- URL: <https://dev.renku.ch/p/rokroskar/mnist-non-interactive-training-job/sessions/show/rokroskar-0fc6812148c0>
- Verified running on commit `e3420d696428820de63f739e73915a7c2ff6fd84`.
- Gradio served under the Renku path prefix using `RENKU_BASE_URL_PATH` as `root_path` in `dashboard.py`.

## Useful commands

```bash
export RENKU_BASE_URL=https://dev.renku.ch
HELPER=/Users/rok/.pi/agent/git/github.com/rokroskar/renku-agent-skill/skills/renku/scripts/renku_agent.py

$HELPER user
$HELPER launcher project-list --project 01KSNRFKT2WYD06YXT25B3WSWM
$HELPER session list
$HELPER job list
$HELPER session get rokroskar-0fc6812148c0
$HELPER session logs rokroskar-0fc6812148c0
$HELPER job run --launcher 01KSNRP4T8TSSB1GYKV1R20QP7
$HELPER session launch --launcher 01KSSGCZ665NA9W020BP5FME7G
```

When rerunning the non-interactive job, remove failed/stopped old job sessions first if Renku would otherwise reuse/conflict with the same generated session name. Ask before stopping running interactive sessions.

## Files to inspect first

- `README.md`
- `docs/dashboard.md`
- `dashboard.py`
- `prepare_model_artifacts.py`
- `train_mnist.py`
- `train_torch_mnist.py`
