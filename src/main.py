from DataDownload.DataDownloader import DataDownloader
from DataDownload.DataValidator import DataValidator
from Preprocessing.DataPreprocessor import DataPreprocessor
from DataDownload.RawDataInspector import RawDataInspector
from Preprocessing.BEVInspector import BEVInspector
from nuscenes.nuscenes import NuScenes
from Globals import NUSCENES_ROOT, NUSCENES_VERSION


def main():

    downloader = DataDownloader()
    if not downloader.check_and_prompt():
        return 1

    validator = DataValidator()
    if not validator.validate():
        print("Dataset validation failed")
        return 1

    nusc = NuScenes(version=NUSCENES_VERSION, dataroot=NUSCENES_ROOT, verbose=False)

    inspector = RawDataInspector(nusc)
    inspector.list_scenes()
    inspector.visualize_sample()
    inspector.visualize_sample_data()
    inspector.visualize_annotation()

    sample = nusc.sample[10]
    lidar_token = sample['data']['LIDAR_TOP']

    pc_info = inspector.inspect_point_cloud(lidar_token)
    annotations = inspector.inspect_annotations(sample['token'])
    inspector.visualize_3d_scene(sample['token'])

    print("\n=== Inspection Point 1: Raw Data ===")
    print(f"\nPoint Cloud: {pc_info['num_points']} points")
    print(f"X: [{pc_info['x_range'][0]:.2f}, {pc_info['x_range'][1]:.2f}] m")
    print(f"Y: [{pc_info['y_range'][0]:.2f}, {pc_info['y_range'][1]:.2f}] m")
    print(f"Z: [{pc_info['z_range'][0]:.2f}, {pc_info['z_range'][1]:.2f}] m")
    
    print(f"\nAnnotations: {len(annotations)} objects")
    for idx, ann in enumerate(annotations):
        print(f"  [{idx}] {ann['category']}: pos={ann['translation']}, size={ann['size']}")

    print("\n=== Preprocessing Stage ===")
    preprocessor = DataPreprocessor(nusc)
    total = preprocessor.process_all_samples(debug_first=True)
    print(f"Processed {total} samples")

    print("\n=== Inspection Point 2: Preprocessed Data ===")
    bev_inspector = BEVInspector()
    bev_images, yolo_labels_list = bev_inspector.load_samples(4)
    bev_inspector.visualize_grid(bev_images, yolo_labels_list, num_cols=2)

    return 0


if __name__ == "__main__":
    exit(main()) 
