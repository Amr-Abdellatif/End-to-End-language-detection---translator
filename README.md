# Language Detection and Translation

This project includes the development of two API endpoints and the training of three models from scratch for language detection and translation.


https://github.com/Amr-Abdellatif/End-to-End-language-detection---translator/assets/92921252/399fbd31-2d4d-41b9-819b-97e3e5c47dd0



## Usage

1. `pip install -r requirements.txt`
    ps : if you face a problem with torch probably you need to install the torch from their website according to your device in my case it was the pip with gpu installation 

2. Run `main.py` this will run uvicorn server with the endpoints

3. Open browser and navigate to the follwoing localhost for swagger ui `http://127.0.0.1:8080/docs`

4. For language detection model you must send data the model was trained on and must be whole words

5. for language translation model i'll pass some sentences i trained the model on as i didnt train it on all the pairs ex : 'انا لا اشعر بالعطش'  - 'i didnt see the need for it'

6. if you try to pass any words that are not in the vocab of the model it would return key-error .. again this can be further trained on the whole dataset and results do improve.

## Test cases for translation model:

1. 'تقول انك تتعمد اخفاء مظهرك الحسن'
> 'you are saying you intentionally hide your good looks <EOS>'

2. 'i didnt see the need for it'
> "لا ارى لذلك حاجة <EOS>"

3. ?!!!!!!? -> this should return unknown

4. 'i read his book' 
> "انا اقرا كتابه <EOS>"

5. 'im sure that she will come back soon' 
> "انا متاكد من انها ستعود قريبا <EOS>"

ps : you can find more test cases at the end of this repo i've evaluated after the training 

## Project structure 

1. models are included in two folders :
    1. language detection model
    2. language translation models -> includes two models one for each language path: ara->eng / eng->ara
2. config is for paths and other configuration for the whole project
3. I included the training noteooks if you want to take a look at what i did
4. model translation architecture contains the architecture of the translation model
5. i made two utils for translation one for English - Arabic translation and one for Arabic - English translation because vocabularies are different
6. data is included also in case you want to take a look at it.

## Language Detection Model

The language detection model is a word-level model that achieved an accuracy of around 97%. It was trained on a dataset containing multiple languages and uses a stratified split to maintain the ratio of each language in the training and testing sets. The API endpoint returns the detected language and the time taken for the request.

## Language Translation Model

The language translation model is also a word-level model that uses the PyTorch deep learning framework and utilized Sequence to Sequence Network and Attention. It includes two models for translation between Arabic to English and English to Arabic, with training loss based on negative log-likelihood and BLEU score for evaluation. The API endpoint checks the language and passes the sentence to the appropriate model.

The models were trained on word-level data, while character-level models could perform better with word-level understanding, they would require more training time and resources.

## Training Details

The language detection model was trained in approximately 3 hours, while the translation models took longer due to the complexity of the task and the use of RNNs with Attention. The translation models were trained on a subset of data with filtered language pairs based on a specific criteria to choose sentences that starts with like "This", "هذه".

## Conclusion

The language detection and translation models perform well on the subset of data used, but further training and resources would be required for generalization.



### Some more test cases from the translator models :

### English -> Arabic
> i dont think tom would want to do that
< لا اظنن توم يريد فعل ذلك <EOS>

> i read his book
< انا اقرا كتابه على الاطلاق <EOS>

> im sure that she will come back soon
< انا متاكد من انها ستعود قريبا <EOS>


> im really hungry
< انا جايع جدا في الصباح <EOS>

> i cant see anything
< لا استطيع ابتكار ارى <EOS>


> i dont know when hell be here
< لا اعرف متى سيكون هنا <EOS>
----------------------------------------------------------------
### Arabic -> English

> حظه يسبق ذكاءه
< he is more lucky than clever <EOS>

> انك لست طالبا
< you are not a student <EOS>

> تقول انك تتعمد اخفاء مظهرك الحسن
< you are saying you intentionally hide your good looks <EOS>

> انه قلق بسبب مرض والده
< he is concerned about his fathers illness <EOS>

> انت امي
< you are my mother <EOS>

> تحبني كل عايلتي
< i am loved by all my family <EOS>

> انا من الاكوادور
< i am from ecuador <EOS>

> هو رجل حكمة
< he is a man of wit <EOS>
