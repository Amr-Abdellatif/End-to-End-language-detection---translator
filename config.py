# utils.py


hidden_size = 256
batch_size = 16


MAX_LENGTH = 10
# translation_data = r'D:\projects\freelance\kemet ai\translation_train.csv'
translation_data = './data/translation_train.csv'


language_detection_path = './language_detection_model/language_detection.pkl'

ENCODER_WEIGHT_PATH_eng_ara = './language_translation_models/english_arabic/english_ara_encoder.pth'
DECODER_WEIGHT_PATH_eng_ara = './language_translation_models/english_arabic/english_ara_decoder.pth'


ENCODER_WEIGHT_PATH_ara_eng = './language_translation_models/arabic_english/ara_encoder.pth'
DECODER_WEIGHT_PATH_ara_eng = './language_translation_models/arabic_english/ara_decoder.pth'