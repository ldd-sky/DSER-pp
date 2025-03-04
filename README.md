# Video Frame Interpolation via Asymmetric Refinement with the Event-based Reference

This is the codebase for our paper DSER++. We provide pretrained weights and code.

## Quick Start
### Environment
Please download dependency packages by
```bash
pip install -r requirements.txt
```

### Dataset
Download supported datasets. Currently we support:
1. Vimeo90k dataset;
2. GOPRO dataset;
3. SNU-FILM dataset;
4. HSERGB dataset;
6. [BSERGB dataset](https://github.com/uzh-rpg/timelens-pp/?tab=readme-ov-file);
7. [EventAid-F dataset](https://sites.google.com/view/EventAid-benchmark);

Please notice that 1, 2 and 3 are synthetic datasets. We synthesis events based on [v2e](https://github.com/SensorsINI/v2e).

### Pretrained weights
[Checkpoints](https://drive.google.com/drive/folders/1Ixhmixa3yMU-2AN3RF5oan5ddNL3Nnko)