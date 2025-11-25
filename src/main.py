from pathlib import Path
from DataDownload.DataDownloader import DataDownloader
from DataDownload.DataValidator import DataValidator
from Preprocessing.DataPreprocessor import DataPreprocessor
from DataDownload.RawDataInspector import RawDataInspector
from Preprocessing.BEVInspector import BEVInspector
from DataPreparation.DataSplitter import DataSplitter
from DataPreparation.DatasetConfigGenerator import DatasetConfigGenerator
from Training.ModelInitializer import ModelInitializer
from Training.TrainingOrchestrator import TrainingOrchestrator
from Evaluation.ModelEvaluator import ModelEvaluator
from Evaluation.ResultsVisualizer import ResultsVisualizer
from Evaluation.PerformanceAnalyzer import PerformanceAnalyzer
from nuscenes.nuscenes import NuScenes
from Globals import NUSCENES_ROOT, NUSCENES_VERSION, DATA_ROOT, PREPROCESSED_ROOT
import torch

def test_gpu():
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        for i in range(gpu_count):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"✓ GPU setup complete - {gpu_count} GPU(s) available")
        return gpu_count
    else:
        print("⚠ No GPU available")
        return 0


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

    inspector = RawDataInspector(nusc)

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

    print("\n=== Preprocessing Stage ===")
    preprocessor = DataPreprocessor(nusc)
    total = preprocessor.process_all_samples()
    print(f"Processed {total} samples")

    print("\n=== Inspection Point 2: Preprocessed Data ===")
    bev_inspector = BEVInspector()
    bev_images, yolo_labels_list = bev_inspector.load_samples(4)
    bev_inspector.visualize_grid(bev_images, yolo_labels_list, num_cols=2)


    print("\n=== Data Preparation Stage ===")
    
    splitter = DataSplitter(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    splits = splitter.split()
    
    config_generator = DatasetConfigGenerator()
    dataset_yaml_path = Path(DATA_ROOT) / 'dataset.yaml'
    config_generator.generate(splits, dataset_yaml_path)
    
    print("✓ Data preparation complete")


    print("\n=== Training Stage ===")
    
    user_input = input("\nProceed with training? This will take several hours. (y/n): ")
    if user_input.lower() != 'y':
        print("Training skipped by user")
        return 0
    
    model_initializer = ModelInitializer(model_size='s', pretrained=True)
    model = model_initializer.initialize()
    
    trainer = TrainingOrchestrator(model, dataset_yaml_path)
    
    stage1_results, stage2_results = trainer.train_full_pipeline(
        stage1_epochs=50,
        stage2_epochs=150,
        batch_size=16
    )
    
    best_model_path = Path(stage2_results.save_dir) / 'weights' / 'best.pt'
    print(f"\n✓ Training complete. Best model: {best_model_path}")

    print("\n=== Evaluation Stage ===")
    
    evaluator = ModelEvaluator(best_model_path, dataset_yaml_path)
    results = evaluator.evaluate()
    evaluator.print_metrics(results)
    
    analyzer = PerformanceAnalyzer()
    summary = analyzer.compute_performance_summary(results)

    # Analyze class distribution from test set labels
    test_labels_dir = splits['test']['labels_dir']
    analyzer.analyze_class_distribution(test_labels_dir)

    visualizer = ResultsVisualizer()
    test_images_dir = splits['test']['images_dir']
    visualizer.visualize_predictions(evaluator.model, test_images_dir, num_samples=10)
    visualizer.generate_performance_report(results)

    print("\n✓ Pipeline complete")
    print(f"mAP@0.5: {summary['overall']['mAP_50']:.4f}")
    print(f"mAP@0.5:0.95: {summary['overall']['mAP_50_95']:.4f}")

    return 0


if __name__ == "__main__":
    exit(main()) 
