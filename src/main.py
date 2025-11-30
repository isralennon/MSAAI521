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
from Globals import NUSCENES_ROOT, NUSCENES_VERSION, DATA_ROOT, PREPROCESSED_ROOT, RUNS_ROOT


def main():

    nusc = NuScenes(version=NUSCENES_VERSION, dataroot=NUSCENES_ROOT, verbose=False)

    run_datadownload_validation(nusc)
    run_data_preparation(nusc)

    dataset_yaml_path = Path(DATA_ROOT) / 'dataset.yaml'
    
    run_training_stage_1(dataset_yaml_path)
    run_training_stage_2(dataset_yaml_path)
    
    best_model_path = Path(RUNS_ROOT) / 'detect' / 'stage1_warmup' / 'weights' / 'best.pt'
    
    results, summary = run_evaluation(best_model_path, dataset_yaml_path)

    visualize_predictions(best_model_path, results)
    
    print("\n✓ Pipeline complete")
    print(f"mAP@0.5: {summary['overall']['mAP_50']:.4f}")
    print(f"mAP@0.5:0.95: {summary['overall']['mAP_50_95']:.4f}")

    return 0

def run_datadownload_validation(nusc: NuScenes):
    downloader = DataDownloader()
    if not downloader.check_and_prompt():
        return 1

    validator = DataValidator()
    if not validator.validate():
        print("Dataset validation failed")
        return 1



    print("\n=== Inspection Point 1: Raw Data ===")
    inspector = RawDataInspector(nusc)
    sample = nusc.sample[10]
    inspector.inspect(sample)

    print("\n=== Preprocessing Stage ===")
    preprocessor = DataPreprocessor(nusc)
    total = preprocessor.process_all_samples()
    print(f"Processed {total} samples")

    print("\n=== Inspection Point 2: Preprocessed Data ===")
    bev_inspector = BEVInspector()
    bev_images, yolo_labels_list = bev_inspector.load_samples(4)
    bev_inspector.visualize_grid(bev_images, yolo_labels_list, num_cols=2)

def run_data_preparation(nusc: NuScenes):

    print("\n=== Data Preparation Stage ===")
    
    splitter = DataSplitter(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    splits = splitter.split()
    
    config_generator = DatasetConfigGenerator()
    dataset_yaml_path = Path(DATA_ROOT) / 'dataset.yaml'
    config_generator.generate(splits, dataset_yaml_path)
    

def run_training_stage_1(yaml: Path):

    print("\n=== Training Stage 1 ===")

    model_initializer = ModelInitializer(model_size='s', pretrained=True)
    model = model_initializer.initialize()
    
    trainer = TrainingOrchestrator(model, yaml)

    # # train stage 1
    stage1_results = trainer.train_stage1(epochs=50, batch_size=4)

def run_training_stage_2(yaml: Path):

    print("\n=== Training Stage 2 ===")

    model_initializer = ModelInitializer(model_size='s', pretrained=True)
    model = model_initializer.initialize()
    
    trainer = TrainingOrchestrator(model, yaml)
    
    # Find best weights from Stage 1 directory produced by YOLO
    stage1_best = Path(trainer.runs_dir) / 'detect' / 'stage1_warmup' / 'weights' / 'best.pt'

    # train stage 2
    stage2_results = trainer.train_stage2(stage1_best, epochs=150, batch_size=4)

def run_evaluation(model_path: Path, yaml: Path):

    print("\n=== Evaluation Stage ===")

    evaluator = ModelEvaluator(model_path, yaml)
    results = evaluator.evaluate()
    evaluator.print_metrics(results)
    
    analyzer = PerformanceAnalyzer()
    summary = analyzer.compute_performance_summary(results)
    
    labels_dir = Path(PREPROCESSED_ROOT) / 'labels'
    analyzer.analyze_class_distribution(labels_dir)
    return results, summary


def visualize_predictions(model_path: Path, results: dict):

    print("\n=== Visualization Stage ===")
    visualizer = ResultsVisualizer()
    test_images_dir = Path(PREPROCESSED_ROOT) / 'test' / 'images'
    visualizer.visualize_predictions(model_path, test_images_dir, num_samples=3)
    visualizer.generate_performance_report(results)


if __name__ == "__main__":
    exit(main()) 
