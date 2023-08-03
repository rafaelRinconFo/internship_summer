import torch
from torch import nn

from metrics import worst_samples_image_logger


class AccuracyThreshold(nn.Module):
    def __init__(self, threshold=1.25):
        super().__init__()
        self.threshold = threshold

    def forward(self, pred, y):
        error = (
            torch.max(torch.div(pred, y), torch.div(y, pred)) < self.threshold
        ).float()
        # percentage of pixels with error < threshold

        return error.mean()


class RMSE(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, y):
        eps = 1e-6
        return torch.sqrt(self.mse(pred, y) + eps)


# A loss robust to outliers + large data values


class RMSLE(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, y):
        return torch.sqrt(
            self.mse(torch.log(torch.abs(pred) + 1), torch.log(torch.abs(y) + 1))
        )


def log_metrics(
    model,
    dataloader,
    metrics,
    epoch,
    wandb,
    device,
    worst_metric_criteria="accuracy_threshold",
    worst_sample_number=5,
):
    model.eval()

    available_metrics = {
        "RMSE": RMSE(),
        "RMSLE": RMSLE(),
        "accuracy_threshold": AccuracyThreshold(),
    }
    metrics_logger = {metric: [] for metric in metrics}
    worst_samples = {"batch": None, "value": 0, "pred": None}
    with torch.no_grad():
        for data in dataloader:
            image, depth_map, names = data
            image = image.to(device)
            image = image.squeeze(1)

            if depth_map.dtype != torch.float32:
                depth_map = depth_map.type(torch.float32)

            depth_map = depth_map.to(device)
            pred = model(image)
            if len(pred.shape) < 4:
                pred = pred.unsqueeze(1)
            pred = torch.nn.functional.interpolate(
                pred,
                size=depth_map.shape[-2:],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

            if len(pred.shape) == 2:
                pred = pred.unsqueeze(0)

            for metric in metrics:
                metrics_logger[metric].append(
                    available_metrics[metric](pred, depth_map).item()
                )

            # Log the worst sample
            if worst_samples["batch"] is None:
                worst_samples["batch"] = data
                worst_samples["pred"] = pred
                worst_samples["value"] = metrics_logger[worst_metric_criteria][-1]
            elif (
                worst_metric_criteria == "accuracy_threshold"
                and metrics_logger[worst_metric_criteria][-1] < worst_samples["value"]
            ):
                worst_samples["batch"] = data
                worst_samples["pred"] = pred
                worst_samples["value"] = metrics_logger[worst_metric_criteria][-1]
            elif (
                worst_metric_criteria != "accuracy_threshold" 
                and metrics_logger[worst_metric_criteria][-1] > worst_samples["value"]
            ):
                worst_samples["batch"] = data
                worst_samples["pred"] = pred
                worst_samples["value"] = metrics_logger[worst_metric_criteria][-1]

    for metric in metrics:
        wandb.log(
            {
                f"val_{metric}": sum(metrics_logger[metric])
                / len(metrics_logger[metric])
            },
            commit=False,
        )

    worst_samples_image_logger(
        wandb,
        worst_sample_number,
        worst_samples["batch"],
        worst_samples["pred"],
        worst_samples["value"],
        worst_metric_criteria,
    )


def main():
    # Tests for the different functions
    test_1 = torch.tensor([1, 2, 3, 4, 5])
    test_2 = torch.tensor([1, 2, 2, 4, 5])
    accuracy_threshold = AccuracyThreshold()
    rmse = RMSE()
    rmsle = RMSLE()
    print(accuracy_threshold)
    print(rmse)
    print(rmsle)
    print(accuracy_threshold(test_1, test_2))


if __name__ == "__main__":
    print("Running tests for metrics/math_metrics.py")
    main()
