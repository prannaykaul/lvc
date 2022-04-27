import datetime
import logging
import time
from collections import OrderedDict
from contextlib import contextmanager
import torch

from detectron2.utils.comm import is_main_process
from lvc.data.utils import get_crops, convert_crops
from lvc.data.builtin_meta import COCO_UNSEEN_IDS
from detectron2.structures import ImageList


class DatasetEvaluator:
    """
    Base class for a dataset evaluator.

    The function :func:`inference_on_dataset` runs the model over
    all samples in the dataset, and have a DatasetEvaluator to process the inputs/outputs.

    This class will accumulate information of the inputs/outputs (by :meth:`process`),
    and produce evaluation results in the end (by :meth:`evaluate`).
    """

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        pass

    def process(self, input, output):
        """
        Process an input/output pair.

        Args:
            input: the input that's used to call the model.
            output: the return value of `model(output)`
        """
        pass

    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.

        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                In our train_net.py, we expect the following format:

                * key: the name of the task (e.g., bbox)
                * value: a dict of {metric name: score}, e.g.: {"AP50": 80}
        """
        pass


class DatasetEvaluators(DatasetEvaluator):
    def __init__(self, evaluators):
        assert len(evaluators)
        super().__init__()
        self._evaluators = evaluators

    def reset(self):
        for evaluator in self._evaluators:
            evaluator.reset()

    def process(self, input, output):
        for evaluator in self._evaluators:
            evaluator.process(input, output)

    def evaluate(self):
        results = OrderedDict()
        for evaluator in self._evaluators:
            result = evaluator.evaluate()
            if is_main_process():
                for k, v in result.items():
                    assert (
                        k not in results
                    ), "Different evaluators produce results with the same key {}".format(k)
                    results[k] = v
        return results


def inference_on_dataset(model, data_loader, evaluator):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    The model will be used in eval mode.

    Args:
        model (nn.Module): a module which accepts an object from
            `data_loader` and returns some outputs. It will be temporarily set to `eval` mode.

            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator (DatasetEvaluator): the evaluator to run. Use
            :class:`DatasetEvaluators([])` if you only want to benchmark, but
            don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} images".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    evaluator.reset()

    logging_interval = 50
    num_warmup = min(5, logging_interval - 1, total - 1)
    start_time = time.time()
    total_compute_time = 0
    with inference_context(model), torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            if idx == num_warmup:
                start_time = time.time()
                total_compute_time = 0

            start_compute_time = time.time()
            outputs = model(inputs)
            torch.cuda.synchronize()
            total_compute_time += time.time() - start_compute_time
            evaluator.process(inputs, outputs)

            if (idx + 1) % logging_interval == 0:
                duration = time.time() - start_time
                seconds_per_img = duration / (idx + 1 - num_warmup)
                eta = datetime.timedelta(
                    seconds=int(seconds_per_img * (total - num_warmup) - duration)
                )
                logger.info(
                    "Inference done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    )
                )

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = int(time.time() - start_time)
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results


def rpn_inference_flat(cfg_lin, model_lin, data_loader):
    num_devices = (torch.distributed.get_world_size()
                   if torch.distributed.is_initialized() else 1)
    logger = logging.getLogger(__name__)
    logger.info("Start rpn inference on {} images".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length

    logging_interval = 50
    num_warmup = min(5, logging_interval - 1, total - 1)
    start_time = time.time()
    total_compute_time = 0
    device = cfg_lin.MODEL.DEVICE
    mean_ = torch.Tensor(cfg_lin.TEMPLATE.PIXEL_MEAN).view(-1, 1, 1)
    std_ = torch.Tensor(cfg_lin.TEMPLATE.PIXEL_STD).view(-1, 1, 1)
    model_lin = model_lin.to(device)
    mean_ = mean_.to(device)
    std_ = std_.to(device)
    unseen = torch.LongTensor(COCO_UNSEEN_IDS).to(device)
    ret_ids = []
    all_scores = []
    all_clses = []
    with inference_context(model_lin), torch.no_grad():
        for idx, data in enumerate(data_loader):
            if idx == num_warmup:
                start_time = time.time()
                total_compute_time = 0

            start_compute_time = time.time()
            temps = [x['image'][[2, 1, 0]].to(device) for x in data]
            temps = [(x - mean_) / std_
                     for x in temps]
            temps = ImageList.from_tensors(
                temps, 32).tensor
            pred = model_lin.model(temps)
            include_data, scores, clses = rpn_process_flat(pred, data, unseen)
            ids = [d['instances'].ids[0].item() for d in include_data]
            ret_ids.extend(ids)
            all_scores.extend(scores)
            all_clses.extend(clses)
            torch.cuda.synchronize()
            total_compute_time += time.time() - start_compute_time
            if (idx + 1) % logging_interval == 0:
                duration = time.time() - start_time
                dur_comp = time.time() - start_compute_time
                seconds_per_img = duration / (idx + 1 - num_warmup)
                seconds_per_img_comp = dur_comp / (idx + 1 - num_warmup)
                eta = datetime.timedelta(
                    seconds=int(seconds_per_img * (total - num_warmup)
                                - duration)
                )
                assert len(ret_ids) == len(all_scores)
                logger.info(
                    ("RPN Inference done {}/{}. {:.4f} s /"
                     + " img. {:.4f} s / img .ETA={}, RetIds={}").format(
                        idx + 1, total, seconds_per_img, seconds_per_img_comp,
                        str(eta), len(ret_ids)
                    )
                )
    # Measure the time only for this worker (before the synchronization barrier)
    total_time = int(time.time() - start_time)
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total RPN inference time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total RPN inference pure compute time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )
    return ret_ids, all_scores, all_clses


def rpn_inference_cascade(model, model_lin, data_loader):
    num_devices = (torch.distributed.get_world_size()
                   if torch.distributed.is_initialized() else 1)
    logger = logging.getLogger(__name__)
    logger.info("Start rpn inference on {} images".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length

    logging_interval = 1
    num_warmup = min(5, logging_interval - 1, total - 1)
    start_time = time.time()
    total_compute_time = 0
    bs = 12
    with inference_context(model), torch.no_grad():
        with inference_context(model_lin), torch.no_grad():
            for idx, data in enumerate(data_loader):
                if idx == num_warmup:
                    start_time = time.time()
                    total_compute_time = 0

                start_compute_time = time.time()
                proposals, img = model(data, no_post=True)
                crops = get_crops(data[0]['image'],
                                  proposals[0].proposal_boxes.tensor.short())
                crops = convert_crops(crops)
                crops = [{'pos_shot': c, 'pos_cat': 0} for c in crops]
                all_pred = torch.zeros(0).long()
                for j in range(len(crops)//bs + 1):
                    pred, clses = model_lin(crops[j*bs:(j+1)*bs])
                    all_pred = torch.cat((all_pred, pred.cpu()))
                pred_proposals = rpn_process(
                    all_pred,
                    proposals)
                # print(pred_proposals)
                torch.cuda.synchronize()
                total_compute_time += time.time() - start_compute_time
                if (idx + 1) % logging_interval == 0:
                    duration = time.time() - start_time
                    seconds_per_img = duration / (idx + 1 - num_warmup)
                    eta = datetime.timedelta(
                        seconds=int(seconds_per_img * (total - num_warmup)
                                    - duration)
                    )
                    logger.info(
                        ("RPN Inference done {}/{}. {:.4f} s /"
                         + " img. ETA={}").format(
                            idx + 1, total, seconds_per_img, str(eta)
                        )
                    )
    # Measure the time only for this worker (before the synchronization barrier)
    total_time = int(time.time() - start_time)
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        ("Total inference time: {} ({:.6f} s / img per device,"
         + " on {} devices)").format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(
        datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: \
         {} ({:.6f} s / img per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup),
            num_devices
        )
    )


def rpn_process(all_pred, proposals):
    val = (all_pred.unsqueeze(0)
           == torch.LongTensor(COCO_UNSEEN_IDS).unsqueeze(1)).any(dim=0)
    proposals[0] = proposals[0][val]
    proposals[0].set('pred_classes', all_pred[val])
    proposals[0].set('pred_boxes', proposals[0].proposal_boxes)
    proposals[0].remove('proposal_boxes')
    return proposals


def rpn_process_flat(pred, data, unseen):
    pred_scores, pred_cls = pred.softmax(dim=1).max(dim=1)
    val = (pred_cls.unsqueeze(0)
           == unseen.unsqueeze(1)).any(dim=0)
    scores = pred_scores[val]
    pred_cls = pred_cls[val]
    return ([d for d, tog in zip(data, val) if tog],
            scores.tolist(), pred_cls.tolist())


@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)
