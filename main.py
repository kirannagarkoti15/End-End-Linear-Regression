from src.load_configuration import configuration
from src.data_preprocessing import DataPrepration
from src.model_build import ModelBuild
from src.prediction import PredictionOnNewData
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    DataPrepration().data_process()
    ModelBuild().train_model()
    PredictionOnNewData().get_prediction()