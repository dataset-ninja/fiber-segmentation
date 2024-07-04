**Fiber Segmentation Dataset** is a dataset for instance segmentation, semantic segmentation, and object detection tasks. It is used in the manufacturing and construction industries. 

The dataset consists of 10941 images with 699597 labeled objects belonging to 1 single class (*polyethylene fiber*).

Images in the Fiber Segmentation dataset have pixel-level instance segmentation annotations. Due to the nature of the instance segmentation task, it can be automatically transformed into a semantic segmentation (only one mask for every class) or object detection (bounding boxes for every object) tasks. There are 993 (9% of the total) unlabeled images (i.e. without annotations). There are 3 splits in the dataset: *train* (7902 images), *val* (1986 images), and *test* (1053 images). Additionally, every image marked with its ***volumes_id*** tag. The dataset was released in 2023 by the <span style="font-weight: 600; color: grey; border-bottom: 1px dashed #d3d3d3;">Institute of Photogrammetry and Remote Sensing, Germany</span>.

<img src="https://github.com/dataset-ninja/fiber-segmentation/raw/main/visualizations/poster.png">
