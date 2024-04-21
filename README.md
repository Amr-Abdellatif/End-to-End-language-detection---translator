# Language Detection and Translation

This project includes the development of two API endpoints and the training of three models from scratch for language detection and translation.

## Usage

1. `pip install -r requirements.txt`

2. Run `main.py` this will run uvicorn server with the endpoints

3. Open browser and navigate to the follwoing localhost `http://127.0.0.1:8080/docs`

4. For language detection model you must send data the model was trained on and must be whole words

5. for language translation model i'll pass some sentences i trained the model on as i didnt train it on all the pairs ex : 'انا لا اشعر بالعطش'  - 'i didnt see the need for it'

6. if you try to pass any words that are not in the vocab of the model it would return key-error .. again this can be further trained on the whole dataset and results do improve.


## Project structure 

1. models are included in two folders :
    1. language detection model
    2. language translation models -> includes two models one for each language path: ara->eng / eng->ara
2. config is for paths and other configuration for the whole project
3. i included the training noteooks if you want to take a look at what i did
4. model translation architecture contains the architecture of the translation model
5. i made two utils for translation one for English - Arabic translation and one for Arabic - English translation 
6. data is included also in case you want to take a look at it.

## Language Detection Model

The language detection model is a word-level model that achieved an accuracy of around 97%. It was trained on a dataset containing multiple languages and uses a stratified split to maintain the ratio of each language in the training and testing sets. The API endpoint returns the detected language and the time taken for the request.

## Language Translation Model

The language translation model is also a word-level model that uses the PyTorch deep learning framework and utilized Sequence to Sequence Network and Attention. It includes two models for translation between Arabic and English, with evaluation based on negative log-likelihood and BLEU score. The API endpoint checks the language and passes the sentence to the appropriate model.

The models were trained on word-level data, and while character-level models could perform better with word-level understanding, they would require more time and resources.

## Training Details

The language detection model was trained in approximately 3 hours, while the translation models took longer due to the complexity of the task and the use of RNNs with Attention. The translation models were trained on a subset of data with filtered language pairs based on a specific criteria.

## Conclusion

The language detection and translation models perform well on the subset of data used, but further training and resources would be required for generalization.
