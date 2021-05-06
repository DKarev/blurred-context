import os
import pathlib
import argparse
import glob
import json
import torch, torchvision
import numpy as np
import detectron2

from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, print_csv_format

from utils.detection2accuracy import detection2accuracy


def test_recognition(model_yaml, checkpoint, dataset, annotations, imagedir, outdir=None, use_rpn=False, record_individual_scores=False):
    """
    Computes detections and uses them to compute recognition accuracies.

    Arguments:
        model_yaml: Path to model config in yaml format.
        dataset: Dataset name, used to name output files.
        annotations: Path to ground truth annotations (json file with COCO-style annotations).
        imagedir: Path to image directory.
        outdir: Directory where output files are stored. When None is passed, the output directory specified in the model's config file is used.
        use_rpn: If True, the region proposal network of the model is used instead of bounding boxes from the ground truth.
    """

    # Register testset
    register_coco_instances(dataset, {}, annotations, imagedir)

    # Load model
    cfg = get_cfg()
    with open(model_yaml) as f:
        cfg = cfg.load_cfg(f)

    print("Numer of classes: {}".format(cfg.MODEL.ROI_HEADS.NUM_CLASSES))

    cfg.DATASETS.TEST = (dataset,)
    model = build_model(cfg)

    # Create outdir
    if outdir == None:
        outdir = cfg.OUTPUT_DIR

    pathlib.Path(outdir).mkdir(exist_ok=True)

    print("Evaluation output directory: " + outdir)

    # Create data loader
    if not use_rpn:
        # returns a list of dicts. Every entry in the list corresponds to one sample, represented by a dict.
        dataset_dicts = detectron2.data.get_detection_dataset_dicts(dataset)

        # add proposal boxes
        for i, s in enumerate(dataset_dicts):
            s["proposal_boxes"] = np.array([ ann["bbox"] for ann in dataset_dicts[i]["annotations"] ]) # np.array([[xmin, ymin, xmax, ymax],[xmin, ymin, xmax, ...], ...]) # kx4 matrix for k proposed bounding boxes
            s["proposal_objectness_logits"] = np.full((s["proposal_boxes"].shape[0],), 10) # logit of 10 is 99.999...%
            s["proposal_bbox_mode"] = detectron2.structures.BoxMode.XYWH_ABS # 1 # (x0, y0, w, h) in absolute floating points coordinates 
        
        print("Proposal boxes added.")

        model.proposal_generator = None # deactivate such that precomputed proposals are used
        print("Region proposal deactivated, ground truth bounding boxes are used.")

        val_loader = build_detection_test_loader(dataset_dicts, mapper=DatasetMapper(is_train=False, augmentations=[], image_format= cfg.INPUT.FORMAT, precomputed_proposal_topk=500))
    else:
        val_loader = build_detection_test_loader(cfg, dataset)

    # load model state (weights) from checkpoint
    DetectionCheckpointer(model).load(checkpoint)

    # evaluate detections
    evaluator = COCOEvaluator(dataset, ("bbox",), False, output_dir=outdir)
    result = inference_on_dataset(model, val_loader, evaluator)
    print_csv_format(result)

    with open(os.path.join(outdir, "evaluation_" + dataset + ".json"), "w") as outfile:
            json.dump(result, outfile)

    # compute accuracies
    detection2accuracy(detections=os.path.join(outdir,"coco_instances_results.json"), groundtruth=annotations, outdir=outdir, record_individual_scores=record_individual_scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_yaml", type=str, help="Path to model config file in yaml format.")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint.")
    parser.add_argument("--dataset", type=str, default="COCOSTUFF_COMPATIBLE" ,help="Dataset name or name of a predefined datsets: COCOSTUFF_UNREL_COMPATIBLE, COCOSTUFF_VIRTUALHOME_COMPATIBLE, COCOSTUFF, UNREL.")
    parser.add_argument("--annotations", type=str, help="Path to COCO-style annotations file. Not required if predefined dataset is used.")
    parser.add_argument("--imagedir", type=str, help="Path to images folder. Not required if predefined dataset is used.")
    parser.add_argument("--outdir", type=str, default="output", help="Path to output folder (will be created if it does not exist). Set None to use output directory from model config.")
    parser.add_argument("--use_rpn", action='store_true', default=False, help="If set, the region proposal network is used instead of ground truth bounding boxes.")
    parser.add_argument("--record_individual_scores", action='store_true', default=True, help="If set, will log for each individual annotion how it was predicted and if the prediction was correct")
    args = parser.parse_args()

    # pre-set datasets for convenience
    if args.dataset == "CONGRUENT_INCONGRUENT_EXP_J":
        args.dataset = "congruent_incongruent_exp_j"
        args.annotations = "/home/dimitar/experiments_I_and_J/annotations/test_annotations_exp_J.json"
    elif args.dataset == "CONGRUENT_INCONGRUENT_EXP_I":
        args.dataset = "congruent_incongruent_exp_i"
        args.annotations = "/home/dimitar/experiments_I_and_J/annotations/test_annotations_exp_I.json"

    test_recognition(args.model_yaml, args.checkpoint, args.dataset, args.annotations, args.imagedir, args.outdir, args.use_rpn, record_individual_scores=args.record_individual_scores)
