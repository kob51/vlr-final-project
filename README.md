# Experimenting with Low-Resolution Point Clouds and One-Shot Learning for 3D Object Detection

## Evaluating VoxelRCNN network
- Evaluation is done by running `Voxel-R-CNN/tools/scrips/eval_voxel_rcnn.sh`
- In that .sh file, we change the weights we want to evaluate in the flag labeled `--ckpt`; either we use the `trained_on_full_epoch_80.pth` or `trained_on_50_epoch_80.pth` weight file. The weight files are stored [here](https://drive.google.com/drive/folders/1U5Auzpqa3LFKWaAiijYqnAQN-gkSEsAN?usp=sharing)
- The evaluation results appear in `Voxel-R-CNN/output/voxel_rcnn/voxel_rcnn_car/default/eval/epoch_80/val/default` in the `result.pkl` file
- To change which downsample percentage dataset we evaluate on, we simply replace the entire folder labeled `Voxel-R-CNN/data/kitti/training/velodyne/` with the folder of the downsampled dataset and run the evaluation script as described above
- To obtain 2D and 3D visualizations, we run the `vlr-project/visualization.py` script with the correct downsampled dataset in place and refer to the desired `result.pkl` file
- To obtain downsampled datasets, we run `vlr-project/downsample.py` script and change the method of downsampling and the voxel/every-n size/number

## One-shot Detection
- One-shot detection is done by running `oneshot_voxelrcnn/tools/scripts/eval_voxel_rcnn.sh` with a `--batch_size` of 1, the `--ckpt` being the one trained on the standard resolution dataset, and the `CFG_NAME=voxel_rcnn/voxel_rcnn_van` since we are doing one-shot detection on vans in this case
- The evaluation results appear in `oneshot_voxelrcnn/output/voxel_rcnn/voxel_rcnn_van/default/eval/epoch_80/val/default` in the `result.pkl` file
- Similar to the standard evalution in 1), we run the `vlr-project/visualization_oneshot.py` script that refers to the desired `result.pkl` file to obtain 2D and 3D visualizations
- To change the target and query images, change the frame IDs in `oneshot_voxelrcnn/tools/eval_utils/eval_utils.py` on lines 70 and 74 
