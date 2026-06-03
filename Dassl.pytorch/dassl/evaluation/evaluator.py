import numpy as np
import os.path as osp
from collections import OrderedDict, defaultdict
import torch
from sklearn.metrics import f1_score, confusion_matrix

from .build import EVALUATOR_REGISTRY


class EvaluatorBase:
    """Base evaluator."""

    def __init__(self, cfg):
        self.cfg = cfg

    def reset(self):
        raise NotImplementedError

    def process(self, mo, gt):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError


@EVALUATOR_REGISTRY.register()
class Classification(EvaluatorBase):
    """Evaluator for classification."""

    def __init__(self, cfg, lab2cname=None, **kwargs):
        super().__init__(cfg)
        self._lab2cname = lab2cname
        self._correct = 0
        self._total = 0
        self._per_class_res = None
        self._y_true = []
        self._y_pred = []
        self.cfg = cfg
        if cfg.TEST.PER_CLASS_RESULT:
            assert lab2cname is not None
            self._per_class_res = defaultdict(list)

    def reset(self):
        self._correct = 0
        self._total = 0
        self._y_true = []
        self._y_pred = []
        if self._per_class_res is not None:
            self._per_class_res = defaultdict(list)

    def process(self, mo, gt):
        # mo (torch.Tensor): model output [batch, num_classes]
        # gt (torch.LongTensor): ground truth [batch]
        pred = mo.max(1)[1]
        matches = pred.eq(gt).float()
        self._correct += int(matches.sum().item())
        self._total += gt.shape[0]

        self._y_true.extend(gt.data.cpu().numpy().tolist())
        self._y_pred.extend(pred.data.cpu().numpy().tolist())

        if self._per_class_res is not None:
            for i, label in enumerate(gt):
                label = label.item()
                matches_i = int(matches[i].item())
                self._per_class_res[label].append(matches_i)

    def evaluate(self):
        results = OrderedDict()
        acc = 100.0 * self._correct / self._total
        err = 100.0 - acc
        macro_f1 = 100.0 * f1_score(
            self._y_true,
            self._y_pred,
            average="macro",
            labels=np.unique(self._y_true)
        )

        # The first value will be returned by trainer.test()
        results["accuracy"] = acc
        results["error_rate"] = err
        results["macro_f1"] = macro_f1

        MEMEA = "\n" if self.cfg.LOREAL.STAGE == 1 else ""
        MEME = f"=> COEF1 {self.cfg.LOREAL.COEF1} COEF2 {self.cfg.LOREAL.COEF2} NATT {self.cfg.TRAINER.ATPROMPT.N_ATT1} result * total: {self._total:,} * correct: {self._correct:,} * accuracy: {acc:.2f}% * error: {err:.2f}% * macro_f1: {macro_f1:.2f}%"
        print(MEME)
        
        AUX = ""
        if self.cfg.LOREAL.MYTH !=0:
            AUX = str(self.cfg.LOREAL.MYTH)
        with open(self.cfg.LOREAL.SAVE.replace("AAA", self.cfg.TRAINER.NAME.split("_")[0]+AUX),"a")  as f:
            f.write(f"{MEMEA}OSIZE{self.cfg.LOREAL.OSIZE} TOSIZE{self.cfg.LOREAL.TOSIZE} SEVE: {self.cfg.LOREAL.SEVERITY} SEED: {self.cfg.SEED} DATASET-split: {self.cfg.DATASET.NAME:<10}-{self.cfg.DATASET.SUBSAMPLE_CLASSES:<4} METHOD: {self.cfg.TRAINER.NAME:<15} SEVERITY: {self.cfg.LOREAL.SEVERITY} STAGE: {self.cfg.LOREAL.STAGE} "+MEME+'\n')
        f.close()

        if self._per_class_res is not None:
            labels = list(self._per_class_res.keys())
            labels.sort()

            print("=> per-class result")
            accs = []

            for label in labels:
                classname = self._lab2cname[label]
                res = self._per_class_res[label]
                correct = sum(res)
                total = len(res)
                acc = 100.0 * correct / total
                accs.append(acc)
                print(
                    f"* class: {label} ({classname})\t"
                    f"total: {total:,}\t"
                    f"correct: {correct:,}\t"
                    f"acc: {acc:.1f}%"
                )
            mean_acc = np.mean(accs)
            print(f"* average: {mean_acc:.1f}%")

            results["perclass_accuracy"] = mean_acc

        if self.cfg.TEST.COMPUTE_CMAT:
            cmat = confusion_matrix(
                self._y_true, self._y_pred, normalize="true"
            )
            save_path = osp.join(self.cfg.OUTPUT_DIR, "cmat.pt")
            torch.save(cmat, save_path)
            print(f"Confusion matrix is saved to {save_path}")

        return results
