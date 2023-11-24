# Data Masking

### Problem: Mask the personal information from the email 
Instead of going with state-of-the-art models(Transformers) and relying 100% on probability. <br>
I decided to go with <b>  SPACY (Rule-based + ML-based)</b>, To decrease the chances of error. <br><br>

<I> 💙 And wanted to share the base version of my code with you guys</i>.

#### End Result🎄

![image](https://github.com/LLama2-Ai/spacyCustomNER/assets/142317270/587d58f7-ac4b-40fe-9167-6c2db7375cc7)

### Requirements.
<ul>
  <li> <b>Data:</b> to capture the names correctly, As names cannot be done with the rule-based approach </li>
  <li> <b> Regular expressions:</b> to fetch phoneNo,Dates,Email addresses, Card details  </li>
</ul>

## Table of Contents

- [Installation & Imports](#installation-&-imports)
- [Data Loading](#data-loading)
- [Data Creation](#data-creation)
- [To Spacy Format](#to-spacy-format)
- [Config File For SPACY Training](#config-file-for-spacy-training)
- [Training the SPACY](#training-the-spacy)
- [Preparing Rule-Based](#preparing-rule-based)
- [Finalizing the pipeline](#finalizing-the-pipeline)
- [Results](#results)

## Installation & Imports

<ul>
  <li>import spacy</li>
  <li>! pip install spacy</li>
  <li>!python -m spacy download en_core_web_lg </li>
  <li>import pandas as pd</li>
  <li>from spacy.tokens import DocBin</li>
  <li>import os</li>
  <li>import random as r</li>
  <li>import re</li>
  <li>from spacy.language import Language</li>
</ul>


## Data Loading
<p> 
Bellow code will read all the text files listed in the names directory. And creates a data frame of names exist in different languages </p>

```
def read_data():
    path = r'names' # use your path
    res=os.listdir(path)
    dataList=[]
    for i in res:
        dataList.append(pd.read_csv(path+"\\"+i,sep='\t+',header=None,engine='python'))
    fullDs=pd.concat(dataList,ignore_index=True,axis=0)
    fullDs.rename(columns={0:'Name'},inplace=True)
    return fullDs
```



## Data Creation

<table>
  <tr><td>Sample</td> <td>Output from Function</td></tr>
  <tr>
    <td>
  From: <b>name </b> <name@tevera.com>
  Sent: Friday, September 22, 2023 8:46 AM
  To: <b>name </b> <name@kandi.com>; <b>name </b> <name@kandi.com>; IT Claims Data Model name <name@tevera.com>; name name name <name@tevera.com>
  Cc: <b>name </b><name@kandi.com>; <b>name </b><name@kandi.com>
  Subject: [EXTERNAL] RE: Old GRP's on Load Env    
    </td>
    <td>
      From: <b>InputFromArray</b> <name@tevera.com>
Sent: Friday, September 22, 2023 8:46 AM
To: <b>InputFromArray</b> <name@kandi.com>; <b>InputFromArray</b>  <name@kandi.com>; <b>InputFromArray</b> <InputFromArray@tevera.com>;<b>InputFromArray</b>  <name@tevera.com>
Cc: <b>InputFromArray</b> <name@kandi.com>; <b>InputFromArray</b>  <name@kandi.com>
Subject: [EXTERNAL] RE: Old GRP's on Load Env
    </td>
  </tr>
</table>


<p> 
Bellow code will generate data for training using templates. and a list of some values you want to replace in the template one by one.
like in my case 
  the <b>name</b>  word will be replaced with the names or any attribute provided in the list.  
</p>


<ul>
  <li><b>fullDs:</b> DataFrame of Names</li>
  <li><b>Email Templates:</b> In my case email you can use any template</li>
  <li><b>LowerBound:</b> Min count of template-array, function will choose a random template in each iteration</li>
  <li><b>UpperBound:</b> Max count of template-array, function will choose a random template in each iteration</li>
  <li><b>max:</b> Number of samples you need</li>
</ul>

```
def prepareData(fullDs,email_templates,lowerBound,upperBound,max=0):
    emails=[]
    listNames=fullDs.Name.tolist()
    listLen=len(listNames)
    for i in range(listLen if max==0 else max):
        template=email_templates[r.randint(lowerBound,upperBound)]
        spans=[]
        stri=''
        for index,str in enumerate(template.split('name')):
            stri+=str
            spans.append((len(stri),len(stri+listNames[index]),'NAME'))
            stri+=listNames[index]
        emails.append((stri,{'entities':spans}))
    return emails

fullDs=read_data()
emailsTrain=prepareData(fullDs,email_templates_train,0,4)
emailsTest=prepareData(fullDs,email_templates_test,0,1,500)
print(len(emailsTrain),len(emailsTest))
```

## To Spacy Format

``` 
def saveForSpacy(emails,fileName):
    db=DocBin()
    for text, annot in emails:
        #print(annot['entities'])
        doc=nlp(text)
        ents=[]
        
        for start,end,label in annot['entities']:
            span=doc.char_span(start, end, label=label,alignment_mode="strict")
            if span is None:
                pass
            else:
                ents.append(span)

        doc.ents=ents
        db.add(doc)
    db.to_disk(''+fileName+'.spacy')

saveForSpacy(emailsTrain,'train')
saveForSpacy(emailsTest,'test')
```
## Config File For SPACY Training 
```
! python -m spacy init config config.cfg --lang en --pipeline ner --optimize accuracy 
```
## Training the SPACY
```
! python -m spacy train config.cfg --output ./ --paths.train ./train.spacy --paths.dev ./test.spacy
```
<p> After training two files will be generated </p>
<ol><li>model-best</li><li>model-last</li></ol>

## Preparing Rule-Based

```
@Language.component('regex_matcher')
def regex_matcher(doc):
  expressions={
      "phone": re.compile(r"[\(]*(\d{3})\s*[\)-]\s*([\d-]+)(?:\s*ext[.\s]*(\d+))?\b"),
      "date": re.compile(r"(0[0-9]|1[0-9]|2[0-9]|3[0-1])[/](0[0-9]|1[0-2])[/]([1-2][0-9][0-9][0-9])"),
      "email": re.compile(r"\S+@\S+")                
  }
  spans=[]
  for label, expression in expressions.items():
      for match in re.finditer(expression, doc.text):
          start, end = match.span()
          entity = doc.char_span(start, end, label=label)
          if entity:
            spans.append(entity)

  doc.ents=list(doc.ents)+spacy.util.filter_spans(spans)
  return doc
```

## Finalizing the pipeline
You can load any model.
```
text='Any text for validation'

nlp = spacy.load('en_core_web_lg')
nlp.add_pipe('regex_matcher',before='ner')
doc = nlp(text)

for ent in doc.ents:
    if (ent.label_=='phone') | (ent.label_=='email') | (ent.label_=='DATE'):
         text=text.replace(ent.text,'[External]'))

namesSpcy=spacy.load(r'yourPathToModel\model-best')
doc=namesSpcy(text)
for ent in doc.ents:
    text=text.replace(ent.text,'[External]')

print(text)
 ```
## Results
![image](https://github.com/LLama2-Ai/spacyCustomNER/assets/142317270/b1f07ee9-2ef7-4f0a-8015-7ba6141e3d5e)
