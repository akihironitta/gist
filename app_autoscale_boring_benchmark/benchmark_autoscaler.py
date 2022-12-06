import time
import datetime
from typing import Any, List

from pydantic import BaseModel

import lightning as L


class RequestModel(BaseModel):
    image: str


class BatchRequestModel(BaseModel):
    inputs: List[RequestModel]


class BatchResponse(BaseModel):
    outputs: List[Any]


class MyPythonServer(L.app.components.PythonServer):
    def __init__(self, *args, **kwargs):
        super().__init__(
            # cloud_compute=L.CloudCompute("cpu-small"),
            port=L.app.utilities.network.find_free_network_port(),
            input_type=BatchRequestModel,
            output_type=BatchResponse,
            start_with_flow=False,
        )
        self._time_requested_at = kwargs["time_requested_at"]
        self.drive = L.app.storage.Drive(f"lit://drive")

    def run(self, *args: Any, **kwargs: Any) -> Any:
        print(f"New work started running at: {datetime.datetime.now()} {self.url}")
        t = time.time() - self._time_requested_at
        filename = f"{self.name}.txt"
        with open(filename, "w") as f:
            f.write(str(t))
        self.drive.put(filename)

        super().run(*args, **kwargs)

    def predict(self, requests: BatchRequestModel):
        batch_size = len(requests.inputs)
        print(f"predicting (batch_size={batch_size})")
        time.sleep(300)
        return BatchResponse(outputs=[{"prediction": 0} for _ in range(batch_size)])


class MyAutoScaler(L.app.components.AutoScaler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_work(self, time_requested_at=time.time()):
        return self._work_cls(*self._work_args, **self._work_kwargs, time_requested_at=time_requested_at)

    def scale(self, replicas: int, metrics: dict) -> int:
        pending_requests_per_running_or_pending_work = metrics["pending_requests"] / (
            replicas + metrics["pending_works"]
        )

        # # scale out if the number of pending requests exceeds max batch size.
        # max_requests_per_work = self.max_batch_size
        # if pending_requests_per_running_or_pending_work >= max_requests_per_work:
        #     return replicas + 1

        # # scale in if the number of pending requests is below 25% of max_requests_per_work
        # min_requests_per_work = max_requests_per_work * 0.25
        # if pending_requests_per_running_or_pending_work < min_requests_per_work:
        #     return replicas - 1

        return 30

    def autoscale(self) -> None:
        """Adjust the number of works based on the target number returned by ``self.scale``."""
        if time.time() - self._last_autoscale < self.autoscale_interval:
            return

        self.load_balancer.update_servers(self.workers)

        metrics = {
            "pending_requests": self.num_pending_requests,
            "pending_works": self.num_pending_works,
        }

        # ensure min_replicas <= num_replicas <= max_replicas
        num_target_workers = max(
            self.min_replicas,
            min(self.max_replicas, self.scale(self.num_replicas, metrics)),
        )
        time_requested_at = time.time()

        # upscale
        num_workers_to_add = num_target_workers - self.num_replicas
        for _ in range(num_workers_to_add):
            print(f"Upscaling from {self.num_replicas} to {self.num_replicas + 1}")
            work = self.create_work(time_requested_at)
            new_work_id = self.add_work(work)
            print(f"Work created: '{new_work_id}'")

        # downscale
        num_workers_to_remove = self.num_replicas - num_target_workers
        for _ in range(num_workers_to_remove):
            print(f"Downscaling from {self.num_replicas} to {self.num_replicas - 1}")
            removed_work_id = self.remove_work(self.num_replicas - 1)
            print(f"Work removed: '{removed_work_id}'")

        self.load_balancer.update_servers(self.workers)
        self._last_autoscale = time.time()


app = L.LightningApp(
    MyAutoScaler(
        MyPythonServer,
        min_replicas=1,
        max_replicas=30,
        autoscale_interval=1,
        endpoint="predict",
        input_type=RequestModel,
        output_type=Any,
        timeout_batching=1,
        max_batch_size=1,
    )
)
