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
            port=L.app.utilities.network.find_free_network_port(),
            input_type=BatchRequestModel,
            output_type=BatchResponse,
            start_with_flow=False,
        )
        self._time_requested_at = time.time()
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

    def create_work(self):
        # TODO: Remove `start_with_flow=False` for faster initialization on the cloud
        return self._work_cls(*self._work_args, **self._work_kwargs, start_with_flow=False)

    def scale(self, replicas: int, metrics: dict) -> int:
        # scale out if the number of pending requests exceeds max batch size.
        max_requests_per_work = self.max_batch_size
        pending_requests_per_running_or_pending_work = metrics["pending_requests"] / (
            replicas + metrics["pending_works"]
        )
        min_requests_per_work = max_requests_per_work * 0.25
        pending_requests_per_running_work = metrics["pending_requests"] / replicas
        print(
            f"replicas={replicas}, "
            f"max_requests_per_work={max_requests_per_work}, "
            f"pending_works={metrics['pending_works']}, "
            f"pending_requests={metrics['pending_requests']}, "
            f"pending_requests_per_running_or_pending_work={pending_requests_per_running_or_pending_work}, "
            f"min_requests_per_work={min_requests_per_work}, "
            f"pending_requests_per_running_work={pending_requests_per_running_work}, "
        )

        if pending_requests_per_running_or_pending_work >= max_requests_per_work:
            print(f"New work requested at: {datetime.datetime.now()} ({replicas} -> {replicas+1})")
            return replicas + 1

        # scale in if the number of pending requests is below 25% of max_requests_per_work
        if pending_requests_per_running_work < min_requests_per_work:
            return replicas - 1

        return replicas


app = L.LightningApp(
    MyAutoScaler(
        MyPythonServer,
        min_replicas=1,
        max_replicas=11,
        autoscale_interval=1,
        endpoint="predict",
        input_type=RequestModel,
        output_type=Any,
        timeout_batching=1,
        max_batch_size=1,
    )
)
