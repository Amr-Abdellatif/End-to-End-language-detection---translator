# Language Detection and Translation

This project includes the development of two API endpoints and the training of three models from scratch for language detection and translation.

## Language Detection Model

The language detection model is a word-level model that achieved an accuracy of around 97%. It was trained on a dataset containing multiple languages and uses a stratified split to maintain the ratio of each language in the training and testing sets. The API endpoint returns the detected language and the time taken for the request.

## Language Translation Model

The language translation model is also a word-level model that uses the PyTorch deep learning framework and utilized Sequence to Sequence Network and Attention. It includes two models for translation between Arabic and English, with evaluation based on negative log-likelihood and BLEU score. The API endpoint checks the language and passes the sentence to the appropriate model.

The models were trained on word-level data, and while character-level models could perform better with word-level understanding, they would require more time and resources.

## Training Details

The language detection model was trained in approximately 3 hours, while the translation models took longer due to the complexity of the task and the use of RNNs with Attention. The translation models were trained on a subset of data with filtered language pairs based on a specific criteria.

## Conclusion

The language detection and translation models perform well on the subset of data used, but further training and resources would be required for generalization.
