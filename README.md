# BertMultiDomainChatbot

BERT-BASED-UNCASED => ATIS + Banking + ACID + CLINC 4 Domain Based
codeLink:https://colab.research.google.com/drive/1zoHuGpHaf7bjVnxX_NQuNiXDJI5Mfefr?usp=sharing

**Bu deneyde 4 domain intentlere ayrışmayacak şekilde etiketlenmiştir.Totalde 4 kategori olacak şekilde model eğitilmiş ve sonuçlar üretilmiştir.

![image](https://user-images.githubusercontent.com/37930894/159634169-36e7c1d3-f685-444c-92e8-e500e9873e93.png)

Tokenizer Info: BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
Max_Len:  20
Model Info: BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels = number_of_categories, 
    output_attentions = False,
    output_hidden_states = False)
Model Epoch Sayısı: 4
Optimizer: AdamW

![image](https://user-images.githubusercontent.com/37930894/159634233-4010a6c7-edbd-429d-abcb-59d50edae2f4.png)

Model Sonuçları: 

Accuracy_score:  0.9747799191054008

F-Score:  0.9755333446162492

Recall:  0.9766717184703382

Precision:  0.974592410342398

Confusion Matrix:

![image](https://user-images.githubusercontent.com/37930894/159634267-7fe25149-e175-4ffb-b859-6966d42fb1a3.png)
