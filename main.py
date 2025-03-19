from src.load_configuration import configuration
from src.data_preprocessing import DataPrepration
from src.model_build import ModelBuild
from src.prediction import PredictionOnNewData
import time
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    start = time.time()
    # print('Loading configuration...')
    # configuration().load_config()
    # print('configuration loaded successfully!!')
    print('Data prepration started...')
    DataPrepration().data_process()
    print('Data prepared successfully!!')
    print('Model building started...')
    ModelBuild().train_model()
    print('Model build and saved successfully!!')
    print('Prediction on new data started...')
    PredictionOnNewData().get_prediction()
    print('Prediction on new data completed successfully!!')
    print(f'Total Elapsed Time: {time.time() - start:.2f}s')