import os

from PIL import Image as PImage
from params_proto import ParamsProto, Flag
from params_proto import Proto



class DemoArgs(ParamsProto):
    example_name = Proto(
        default="hurdle_many",
        help="Name of the example to run. See the examples folder for the full list.",
    )

    seed = Proto(default=42, help="Random seed for picking the prompt")
    save = Flag(default=False, help="Save the generated image under the examples folder.")

    # control parameters for image generation
    num_steps = 7
    denoising_strength = 1.0

    control_strength = 0.8

    grow_mask_amount = 6
    cone_grow_mask_amount = 10

    terrain_strength = 1.0
    background_strength = 0.7
    cone_strength = 1.5


def main(**deps):
    from weaver import EXAMPLES_ROOT
    from weaver.utils import pick
    from weaver.workflows.three_mask_workflow import ImagenCone
    import random

    DemoArgs._update(deps)
    from params_proto.hyper import Sweep

    random.seed(DemoArgs.seed)
    imagen = ImagenCone()

    example_dataset = f"{EXAMPLES_ROOT}/three_mask_workflow/{DemoArgs.example_name}"

    try:
        depth = PImage.open(f"{example_dataset}/midas_depth.png")
        foreground_mask = PImage.open(f"{example_dataset}/foreground_mask.png")
        background_mask = PImage.open(f"{example_dataset}/background_mask.png")
        cone_mask = PImage.open(f"{example_dataset}/cone_mask.png")
        prompts = Sweep.read(f"{example_dataset}/prompts.jsonl")
    except FileNotFoundError:
        print(f"Couldn't find the dataset {example_dataset}. Check the examples folder for the complete list.")
        exit()

    control_parameters = pick(
        vars(DemoArgs),
        "num_steps",
        "denoising_strength",
        "control_strength",
        "grow_mask_amount",
        "cone_grow_mask_amount",
        "terrain_strength",
        "background_strength",
        "cone_strength",
    )

    prompt = prompts[random.randint(0, len(prompts) - 1)]
    prompt = pick(prompt, "foreground_prompt", "background_prompt", "cone_prompt", "negative_prompt")

    image = imagen.generate(
        midas_depth=depth,
        foreground_mask=foreground_mask,
        background_mask=background_mask,
        cone_mask=cone_mask,
        **control_parameters,
        **prompt,
    )

    image.format = "jpeg"

    s2id = lambda x: int(x.split("/")[-1].split(".")[0].split("_")[-1])

    largest = max(
        os.listdir(f"{EXAMPLES_ROOT}/three_mask_workflow/{DemoArgs.example_name}/samples"),
        key=s2id,
    )

    img_count = s2id(largest) + 1

    if DemoArgs.save:
        image.save(f"{EXAMPLES_ROOT}/three_mask_workflow/{DemoArgs.example_name}/samples/sample_{img_count}.jpg")

    return image


if __name__ == "__main__":
    main()
