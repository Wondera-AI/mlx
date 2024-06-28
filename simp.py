import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler


def train_fn(config):
    for epoch in range(10):
        # Your training loop here
        ray.train.report(mean_loss=0.1 * epoch)


if __name__ == "__main__":
    ray.init()

    placement_group = ray.util.placement_group(
        [{"CPU": 1}, {"CPU": 1}], strategy="PACK"
    )
    ray.get(placement_group.ready())

    analysis = tune.run(
        train_fn,
        resources_per_trial={"cpu": 1, "gpu": 0},
        config={"param": tune.grid_search([1, 2, 3])},
        num_samples=1,
        scheduler=ASHAScheduler(),
        local_dir="~/ray_results",
        log_to_file=True,
        name="my_experiment",
        trial_resources={"placement_group": placement_group},
    )
