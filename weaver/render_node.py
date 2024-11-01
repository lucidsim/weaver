import time
from importlib import import_module
from time import sleep
from typing import Any, Callable

from params_proto import ParamsProto, Proto, Flag
from zaku import TaskQ

from weaver.utils import center_crop_pil


class RenderWorker(ParamsProto, prefix="render_worker"):
    queue_name: str = Proto(env="$ZAKU_USER:lucidsim:weaver-queue-1")
    output_queue_name: str = Proto(env="$ZAKU_USER:lucidsim:weaver-queue-out")
    monitor_queue_name: str = Proto(env="vuer:$ZAKU_USER:monitor-queue")

    workflow_cls: str = "weaver.workflows.three_mask_workflow:ImagenCone"

    visualize: bool = Flag("whether to visualize the output")

    data_server = Proto(env="WEAVER_DATA_SERVER")

    # for saving the smaller images
    crop_size = (1280, 720)
    downscale = 4

    verbose = Flag("Print verbose output")

    def __post_init__(self, worker_kwargs=None):
        from ml_logger import ML_Logger

        self.logger = ML_Logger(root=self.data_server)

        if self.verbose:
            print("================================")
            for k, v in vars(self).items():
                print(f" {k} = {v},")
            print("================================")

        self.queue = TaskQ(name=self.queue_name)

        self.output_queue = TaskQ(name=self.output_queue_name)
        self.monitor_queue = TaskQ(name=self.monitor_queue_name)

        self.load_workflow(**worker_kwargs or {})

    def load_workflow(self, **worker_kwargs):
        # this mode of operation starts the worker on construction
        print("loading the worker", worker_kwargs)

        module_name, entrypoint_name = self.workflow_cls.split(":")
        module = import_module(module_name)
        FlowClass = getattr(module, entrypoint_name)

        print("loading the worker")
        self.workflow: Callable[[Any], None] = FlowClass(**worker_kwargs)

        print("created workflow", self.workflow_cls)

    def unload_workflow(self, collect_garbage=True) -> bool:
        if self.workflow is None:
            return False

        self.workflow = None
        print("unloaded workflow")

        if collect_garbage:
            import gc
            import torch

            torch.cuda.empty_cache()
            gc.collect()

        return True

    def run(self):
        print("starting the worker...")
        import GPUtil

        self.__last_task_ts = time.perf_counter()
        self.__last_unstale_ts = time.perf_counter()

        gpus = GPUtil.getGPUs()
        while True:
            gpu_availability = GPUtil.getAvailability(
                gpus,
                maxLoad=0.3,
                maxMemory=0.2,
                includeNan=False,
                excludeID=[],
                excludeUUID=[],
            )
            if not gpu_availability:
                sleep(3.0)
                continue

            if (time.perf_counter() - self.__last_task_ts) > 30 and (time.perf_counter() - self.__last_unstale_ts) > 30:
                print("unstaling tasks")
                self.queue.unstale_tasks(ttl=20)
                self.__last_unstale_ts = time.perf_counter()

            if (time.perf_counter() - self.__last_task_ts) > 360:
                self.unload_workflow()

            with self.queue.pop() as job_kwargs:
                if job_kwargs is None:
                    print(".", end="")
                    sleep(0.1)
                    continue

                print("job_kwargs", job_kwargs)

                # untested
                if job_kwargs.get("$kill", None):
                    # exist completes the task.
                    exit()

                # todo: add the ability to load a different workflow

                _deps = job_kwargs.pop("_deps", None)
                to_logger = job_kwargs.pop("to_logger", None)
                to_pipe = job_kwargs.pop("to_pipe", None)
                to_visualize = job_kwargs.pop("to_visualize", None)
                logger_prefix = job_kwargs.pop("logger_prefix", None)
                render_kwargs = job_kwargs.pop("render_kwargs", {})

                # if we need the stream mode, this will be some uuid4 topic name
                to_topic = job_kwargs.pop("_request_id", None)

                print(">>> render kwargs", render_kwargs)

                old_workflow_cls = self.workflow_cls
                self.update(render_kwargs)

                print(">>> render kwargs", self.workflow_cls)

                if old_workflow_cls != self.workflow_cls or self.workflow is None:
                    self.load_workflow()

                self.__last_task_ts = time.perf_counter()

                # print(*job_kwargs.keys())
                generated_image = self.workflow.generate(_deps, **job_kwargs)

                if self.downscale > 1:
                    # crop and resize
                    generated_image = center_crop_pil(generated_image, *self.crop_size)
                    new_size = self.crop_size[0] // self.downscale, self.crop_size[1] // self.downscale
                    generated_image = generated_image.resize(new_size)

                if to_logger:
                    with self.logger.Prefix(logger_prefix):
                        if self.downscale > 1:
                            fname, _ = to_logger.split(".")
                            saved_path = self.logger.save_image(generated_image, f"{fname}_{self.downscale}x.jpeg")
                        else:
                            saved_path = self.logger.save_image(generated_image, to_logger)

                        print(f"saved to {saved_path}")

                if to_pipe:
                    self.output_queue.add({"generated_rgb": generated_image})

                if self.visualize or to_visualize:
                    generated_image.format = "JPEG"
                    self.monitor_queue.add({"render": generated_image})

                if to_topic is not None:
                    self.queue.publish(dict(generated=generated_image), topic=to_topic)


def entrypoint(**deps):
    worker = RenderWorker(**deps)
    worker.run()


if __name__ == "__main__":
    entrypoint()
