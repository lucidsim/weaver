import random
from typing import List

import PIL.Image as pil_image
import numpy as np
import torch
from params_proto import ParamsProto

from weaver.utils import (
    get_value_at_index,
    import_custom_nodes,
    add_extra_model_paths,
)


def disable_comfy_args():
    """
    Hacky injection to disable comfy args parsing.
    """
    import comfy.options
    def enable_args_parsing(enable=False):
        global args_parsing
        args_parsing = enable
        
    comfy.options.enable_args_parsing = enable_args_parsing


def image_grid(img_list: List[List[pil_image.Image]]):
    rows = len(img_list)
    cols = len(img_list[0])

    w, h = img_list[0][0].size
    grid = pil_image.new("RGB", size=(cols * w, rows * h))

    for i, row in enumerate(img_list):
        for j, img in enumerate(row):
            grid.paste(img, box=(j * w, i * h))
    return grid


class ImagenCone(ParamsProto, prefix="imagen", cli=False):
    """
    Image Generation from three semantic masks.

    foreground_prompt: str
    background_text: str
    cone_prompt: str (although this can be replaced with any third object)

    negative_text: str

    control_parameters:
        strength: float
        grow_mask_amount: int
        cone_grow_mask_amount: int

        terrain_strength: float
        background_strength: float
        cone_strength: float

    """

    foreground_prompt = "close up view, photo realistic curbs made of (wood:1.5), weathered, cracked, 35mm IMAX, very large"
    background_prompt = "close up view, photorealistic view of a college campus from a small dog’s perspective during golden hour, highlighting textures of grass, foliage, and buildings"
    cone_prompt = "large orange cones"

    negative_prompt: str = "dog"

    width = 1280
    height = 768
    batch_size: int = 1

    num_steps = 7
    denoising_strength = 1.0

    control_strength = 0.8

    grow_mask_amount = 6
    cone_grow_mask_amount = 10

    terrain_strength = 1.0
    background_strength = 0.7
    cone_strength = 1.5

    sdxl_path = "sd_xl_turbo_1.0_fp16.safetensors"
    device = "cuda"

    def __post_init__(self):
        disable_comfy_args()
        add_extra_model_paths()
        import_custom_nodes()

        from nodes import (
            EmptyLatentImage,
            CheckpointLoaderSimple,
            NODE_CLASS_MAPPINGS,
            VAEDecode,
            CLIPTextEncode,
            ControlNetLoader,
        )

        checkpointloadersimple = CheckpointLoaderSimple()
        self.checkpoint = checkpointloadersimple.load_checkpoint(ckpt_name=self.sdxl_path)
        self.clip_text_encode = CLIPTextEncode()
        self.empty_latent = EmptyLatentImage()

        ksamplerselect = NODE_CLASS_MAPPINGS["KSamplerSelect"]()
        self.ksampler = ksamplerselect.get_sampler(sampler_name="lcm")

        controlnetloader = ControlNetLoader()
        self.controlnet = controlnetloader.load_controlnet(control_net_name="controlnet_depth_sdxl_1.0.safetensors")

        self.imagetomask = NODE_CLASS_MAPPINGS["ImageToMask"]()
        self.growmask = NODE_CLASS_MAPPINGS["GrowMask"]()
        self.vaedecode = VAEDecode()

        print("loading is done.")

    @torch.no_grad
    @staticmethod
    def to_tensor(img: pil_image.Image):
        np_img = np.asarray(img)
        return torch.Tensor(np_img) / 255.0

    def generate(
            self,
            _deps=None,
            *,
            midas_depth: pil_image.Image,
            foreground_mask: pil_image.Image,
            background_mask: pil_image.Image,
            cone_mask: pil_image.Image,
            **deps,
    ) -> pil_image.Image:
        """
        depth: pil_image.Image
        background_mask: pil_image.Image
        foreground_mask: pil_image.Image

        shape: (height, width)
        """

        from nodes import (
            ConditioningSetMask,
            ConditioningCombine,
            NODE_CLASS_MAPPINGS,
            ControlNetApply,
        )

        # we reference the class to take advantage of the namespacing
        ImagenCone._update(_deps, **deps)

        depths_t = ImagenCone.to_tensor(midas_depth)[None, ..., None].repeat([1, 1, 1, 3])
        foreground_mask_t = ImagenCone.to_tensor(foreground_mask)[None, ..., None].repeat([1, 1, 1, 3])
        background_mask_t = ImagenCone.to_tensor(background_mask)[None, ..., None].repeat([1, 1, 1, 3])
        cone_mask_t = ImagenCone.to_tensor(cone_mask)[None, ..., None].repeat([1, 1, 1, 3])

        assert ImagenCone.batch_size == 1, "only generate one for now."

        with torch.inference_mode():
            emptylatentimage_5 = self.empty_latent.generate(
                width=ImagenCone.width,
                height=ImagenCone.height,
                batch_size=ImagenCone.batch_size,
            )

            background_textencode = self.clip_text_encode.encode(
                text=ImagenCone.background_prompt,
                clip=get_value_at_index(self.checkpoint, 1),
            )

            terrain_textencode = self.clip_text_encode.encode(
                text=ImagenCone.foreground_prompt,
                clip=get_value_at_index(self.checkpoint, 1),
            )

            negative_textencode = self.clip_text_encode.encode(
                text=ImagenCone.negative_prompt,
                clip=get_value_at_index(self.checkpoint, 1),
            )

            cones_textencode = self.clip_text_encode.encode(
                text=ImagenCone.cone_prompt,
                clip=get_value_at_index(self.checkpoint, 1),
            )

            conditioningsetmask = ConditioningSetMask()
            conditioningcombine = ConditioningCombine()
            controlnetapply = ControlNetApply()
            sdturboscheduler = NODE_CLASS_MAPPINGS["SDTurboScheduler"]()
            samplercustom = NODE_CLASS_MAPPINGS["SamplerCustom"]()

            imagetomask_69 = self.imagetomask.image_to_mask(channel="red",
                                                            image=get_value_at_index([background_mask_t], 0))

            growmask_69 = self.growmask.expand_mask(
                expand=ImagenCone.grow_mask_amount,
                tapered_corners=True,
                mask=get_value_at_index(imagetomask_69, 0),
            )

            background_condition = conditioningsetmask.append(
                strength=ImagenCone.background_strength,
                set_cond_area="default",
                conditioning=get_value_at_index(background_textencode, 0),
                mask=get_value_at_index(growmask_69, 0),
            )

            imagetomask_70 = self.imagetomask.image_to_mask(channel="red",
                                                            image=get_value_at_index([foreground_mask_t], 0))

            growmask_70 = self.growmask.expand_mask(
                expand=ImagenCone.grow_mask_amount,
                tapered_corners=True,
                mask=get_value_at_index(imagetomask_70, 0),
            )

            terrain_condition = conditioningsetmask.append(
                strength=ImagenCone.terrain_strength,
                set_cond_area="default",
                conditioning=get_value_at_index(terrain_textencode, 0),
                mask=get_value_at_index(growmask_70, 0),
            )

            terrain_background_combine = conditioningcombine.combine(
                conditioning_1=get_value_at_index(terrain_condition, 0),
                conditioning_2=get_value_at_index(background_condition, 0),
            )

            imagetomask_71 = self.imagetomask.image_to_mask(channel="red", image=get_value_at_index([cone_mask_t], 0))

            growmask_71 = self.growmask.expand_mask(
                expand=ImagenCone.cone_grow_mask_amount,
                tapered_corners=True,
                mask=get_value_at_index(imagetomask_71, 0),
            )

            cones_condition = conditioningsetmask.append(
                strength=ImagenCone.cone_strength,
                set_cond_area="default",
                conditioning=get_value_at_index(cones_textencode, 0),
                mask=get_value_at_index(growmask_71, 0),
            )

            final_combine = conditioningcombine.combine(
                conditioning_1=get_value_at_index(terrain_background_combine, 0),
                conditioning_2=get_value_at_index(cones_condition, 0),
            )

            controlnetapply_59 = controlnetapply.apply_controlnet(
                strength=ImagenCone.control_strength,
                conditioning=get_value_at_index(final_combine, 0),
                control_net=get_value_at_index(self.controlnet, 0),
                image=get_value_at_index((depths_t,), 0),
            )

            sdturboscheduler_22 = sdturboscheduler.get_sigmas(
                steps=ImagenCone.num_steps,
                denoise=ImagenCone.denoising_strength,
                model=get_value_at_index(self.checkpoint, 0),
            )

            samplercustom_13 = samplercustom.sample(
                add_noise=True,
                noise_seed=random.randint(1, 2 ** 64),
                cfg=1,
                model=get_value_at_index(self.checkpoint, 0),
                positive=get_value_at_index(controlnetapply_59, 0),
                negative=get_value_at_index(negative_textencode, 0),
                sampler=get_value_at_index(self.ksampler, 0),
                sigmas=get_value_at_index(sdturboscheduler_22, 0),
                latent_image=get_value_at_index(emptylatentimage_5, 0),
            )

            (image_batch,) = self.vaedecode.decode(
                samples=get_value_at_index(samplercustom_13, 0),
                vae=get_value_at_index(self.checkpoint, 2),
            )[:1]

            (generated_image,) = image_batch

            image_np = (generated_image * 255).cpu().numpy().astype("uint8")

            generated = pil_image.fromarray(image_np)
            generated.format = "jpeg"

            return generated
