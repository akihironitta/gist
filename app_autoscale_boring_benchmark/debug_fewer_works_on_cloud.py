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

    def run(self, *args: Any, **kwargs: Any) -> Any:
        print(f"New work started at: {datetime.datetime.now()} {self.url}")
        super().run(*args, **kwargs)

    def predict(self, requests: BatchRequestModel):
        batch_size = len(requests.inputs)
        print(f"predicting (batch_size={batch_size}, url={self.url}, port={self.port})")
        time.sleep(10)
        return BatchResponse(outputs=[{"prediction": 0} for _ in range(batch_size)])


# class MyAutoScaler(L.app.components.AutoScaler):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self._time_last_scale_called = time.time()

#     def scale(self, replicas: int, metrics: dict) -> int:
#         max_requests_per_work = self.max_batch_size
#         pending_requests_per_running_or_pending_work = metrics["pending_requests"] / (
#             replicas + metrics["pending_works"]
#         )
#         min_requests_per_work = max_requests_per_work * 0.25
#         current_time = time.time()
#         print(
#             f"last_called={current_time - self._time_last_scale_called:0.2f}, "
#             f"replicas={replicas}, "
#             f"max_requests_per_work={max_requests_per_work}, "
#             f"pending_works={metrics['pending_works']}, "
#             f"pending_works={metrics['pending_requests']}, "
#             f"pending_requests_per_running_or_pending_work={pending_requests_per_running_or_pending_work}, "
#             f"min_requests_per_work={min_requests_per_work}, "
#             f"replicas_to_create={pending_requests_per_running_or_pending_work // max_requests_per_work}, "
#         )
#         self._time_last_scale_called = current_time

#         # scale out if the number of pending requests exceeds max batch size
#         if pending_requests_per_running_or_pending_work >= max_requests_per_work:
#             # replicas_to_create = int(pending_requests_per_running_or_pending_work // max_requests_per_work)
#             print(f"ASDF: increasing from {replicas} to {replicas + 1}")
#             return replicas + 1

#         # scale in if the number of pending requests is below 25% of max_requests_per_work
#         if pending_requests_per_running_or_pending_work < min_requests_per_work:
#             print(f"ASDF: decreasing from {replicas} to {replicas - 1}")
#             return replicas - 1

#         return replicas


# the correct behaviour is:
# >>> send n requests -> create n works (n+1 works in total)
app = L.LightningApp(
    L.app.components.AutoScaler(
        MyPythonServer,
        min_replicas=1,
        max_replicas=100,
        autoscale_interval=1,
        endpoint="predict",
        input_type=RequestModel,
        output_type=Any,
        timeout_batching=1,
        max_batch_size=1,
    )
)

# the correct behaviour is:
# >>> send n requests -> create int(n/8) works (int(n/8)+1 works in total)  # needs revisit
# app = L.LightningApp(
#     MyAutoScaler(
#         MyPythonServer,
#         min_replicas=1,
#         max_replicas=100,
#         autoscale_interval=10,
#         endpoint="predict",
#         input_type=RequestModel,
#         output_type=Any,
#         timeout_batching=1,
#         max_batch_size=8,
#     )
# )
