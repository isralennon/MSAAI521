from utilsTF import load_cuda_libraries
from utilsTF import configure_tensorflow
from utilsTF import test_gpu


from Pets import Pets

# from utilsTorch import test_gpu

# from YOLO import YOLO_Assignment

from DataDownloader import DataDownloader
from DataPreprocessor import DataPreprocessor


def main():

    # load_cuda_libraries()
    # configure_tensorflow()
    # test_gpu()


    # assignment3 = Pets()
    # assignment3.run()

    # yolo = YOLO_Assignment()
    # yolo.run()

    data_downloader = DataDownloader()
    data_downloader.download_data()

    data_preprocessor = DataPreprocessor()
    data_preprocessor.preprocess_data()

    print("end of program")

    return 0


if __name__ == "__main__":
    exit(main()) 