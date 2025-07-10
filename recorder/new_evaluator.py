import numpy as np
import os.path as osp
from collections import OrderedDict, defaultdict
import torch
from sklearn.metrics import f1_score, confusion_matrix
from utils import log

class EvaluatorBase:
    """Base evaluator."""

    def __init__(self):
        a=1

    def reset(self):
        raise NotImplementedError

    def process(self, mo, gt):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

class Classification(EvaluatorBase):
    """Evaluator for classification."""

    def __init__(self, lab2cname=None, per_class_result=False):
        super().__init__()
        self._lab2cname = lab2cname
        self._correct = 0
        self._total = 0
        self._per_class_res = None
        self._y_true = []
        self._y_pred = []
        if per_class_result:
            assert lab2cname is not None
            self._per_class_res = defaultdict(list)

    def reset(self):
        self._correct = 0
        self._correct_5 = 0
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
        # 处理top5
        pred_5 = mo.topk(5, dim=1)[1].squeeze(dim=1)
        for i in range(pred_5.shape[0]):
            if gt[i].item() in pred_5[i].tolist():
                self._correct_5 += 1

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
        top5 = 100.0 * self._correct_5 / self._total
        macro_f1 = 100.0 * f1_score(
            self._y_true,
            self._y_pred,
            average="macro",
            labels=np.unique(self._y_true)
        )

        # The first value will be returned by trainer.test()
        results["accuracy"] = acc
        results["error_rate"] = err
        results["top5"] = top5
        results["macro_f1"] = macro_f1

        log(
            "=> result\n"
            f"* total: {self._total:,}\n"
            f"* correct: {self._correct:,}\n"
            f"* accuracy: {acc:.2f}%\n"
            f"* error: {err:.2f}%\n"
            f"* macro_f1: {macro_f1:.2f}%\n"
            f"* top5: {top5:.2f}%"
        )

        if self._per_class_res is not None:
            labels = list(self._per_class_res.keys())
            labels.sort()

            log("=> per-class result")
            accs = []

            for label in labels:
                classname = self._lab2cname[label]
                res = self._per_class_res[label]
                correct = sum(res)
                total = len(res)
                acc = 100.0 * correct / total
                accs.append(acc)
                log(
                    f"* class: {label} ({classname})\t"
                    f"total: {total:,}\t"
                    f"correct: {correct:,}\t"
                    f"acc: {acc:.1f}%"
                )
            mean_acc = np.mean(accs)
            log(f"* average: {mean_acc:.1f}%")

            results["perclass_accuracy"] = mean_acc

        # if self.cfg.TEST.COMPUTE_CMAT:
        #     cmat = confusion_matrix(
        #         self._y_true, self._y_pred, normalize="true"
        #     )
        #     save_path = osp.join(self.cfg.OUTPUT_DIR, "cmat.pt")
        #     torch.save(cmat, save_path)
        #     print(f"Confusion matrix is saved to {save_path}")

        return results
