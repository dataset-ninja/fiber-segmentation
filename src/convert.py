import os
import shutil

import supervisely as sly
from cv2 import connectedComponents
from dataset_tools.convert import unpack_if_archive
from supervisely.io.fs import (
    file_exists,
    get_file_ext,
    get_file_name,
    get_file_name_with_ext,
    get_file_size,
)
from tqdm import tqdm

import src.settings as s


def convert_and_upload_supervisely_project(
    api: sly.Api, workspace_id: int, project_name: str
) -> sly.ProjectInfo:
    # Possible structure for bbox case. Feel free to modify as you needs.

    batch_size = 30

    images_folder = "images"
    masks_folder = "masks"
    train_folder_path = "/home/alex/DATASETS/IMAGES/archive/fibers_geometric_aug/fibers"
    val_folder_path = "/home/alex/DATASETS/IMAGES/archive/fibers_geometric_aug/fibers/validation"
    test_folder_path = "/home/alex/DATASETS/IMAGES/archive/fibers_geometric_aug/fibers/test"
    full_volume_test_path = "/home/alex/DATASETS/IMAGES/archive/fibers_full_volume_for_testing/fibers_full_volume_for_testing"

    group_tag_name = "volumes_id"

    ds_name_to_data = {
        "train": train_folder_path,
        "validation": val_folder_path,
        "test": test_folder_path,
    }

    def create_ann(image_path):
        labels = []

        group_id = sly.Tag(tag_id, value=id_data)
        mask_name = get_file_name_with_ext(image_path)
        mask_path = os.path.join(masks_path, subfolder, mask_name)
        if file_exists(mask_path):
            mask_np = sly.imaging.image.read(mask_path)[:, :, 0]
            img_height = mask_np.shape[0]
            img_wight = mask_np.shape[1]
            mask = mask_np == 255
            ret, curr_mask = connectedComponents(mask.astype("uint8"), connectivity=8)
            for i in range(1, ret):
                obj_mask = curr_mask == i
                curr_bitmap = sly.Bitmap(obj_mask)
                curr_label = sly.Label(curr_bitmap, obj_class)
                labels.append(curr_label)
        else:
            img_height = 1600
            img_wight = 1040

        return sly.Annotation(img_size=(img_height, img_wight), labels=labels, img_tags=[group_id])

    obj_class = sly.ObjClass("polyethylene fiber", sly.Bitmap)

    tag_id = sly.TagMeta(group_tag_name, sly.TagValueType.ANY_STRING)
    group_tag_meta = sly.TagMeta(group_tag_name, sly.TagValueType.ANY_STRING)

    project = api.project.create(workspace_id, project_name, change_name_if_conflict=True)
    meta = sly.ProjectMeta(obj_classes=[obj_class])
    meta = meta.add_tag_meta(group_tag_meta)
    api.project.update_meta(project.id, meta.to_json())
    api.project.images_grouping(id=project.id, enable=True, tag_name=group_tag_name)

    for ds_name, ds_path in list(ds_name_to_data.items()):
        dataset = api.dataset.create(project.id, ds_name.lower(), change_name_if_conflict=True)
        images_path = os.path.join(ds_path, images_folder)
        masks_path = os.path.join(ds_path, masks_folder)

        for subfolder in os.listdir(images_path):
            id_data = subfolder.split("_")[0] + "_" + subfolder.split("_")[-1]
            images_names_prefix = subfolder.split("_")[0] + "_" + subfolder.split("_")[-1]
            curr_data_path = os.path.join(images_path, subfolder)
            images_names = os.listdir(curr_data_path)

            progress = sly.Progress(
                "Create dataset {}, add {} subfolder".format(ds_name, subfolder), len(images_names)
            )

            for img_names_batch in sly.batched(images_names, batch_size=batch_size):
                images_pathes_batch = [
                    os.path.join(curr_data_path, image_name) for image_name in img_names_batch
                ]

                unique_image_names = [
                    images_names_prefix + "_" + im_name for im_name in img_names_batch
                ]

                img_infos = api.image.upload_paths(
                    dataset.id, unique_image_names, images_pathes_batch
                )
                img_ids = [im_info.id for im_info in img_infos]

                anns_batch = [create_ann(image_path) for image_path in images_pathes_batch]
                api.annotation.upload_anns(img_ids, anns_batch)

                progress.iters_done_report(len(img_names_batch))

    # is need to add full volume data?
    images_names = [
        im_name for im_name in os.listdir(full_volume_test_path) if get_file_ext(im_name) == ".bmp"
    ]
    id_data = "full_volume"
    for img_names_batch in sly.batched(images_names, batch_size=batch_size):
        images_pathes_batch = [
            os.path.join(full_volume_test_path, image_name) for image_name in img_names_batch
        ]

        img_infos = api.image.upload_paths(dataset.id, img_names_batch, images_pathes_batch)
        img_ids = [im_info.id for im_info in img_infos]

        anns_batch = [create_ann(image_path) for image_path in images_pathes_batch]
        api.annotation.upload_anns(img_ids, anns_batch)
