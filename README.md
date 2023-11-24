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

- [Installation & Imports](#installation)
- [Usage](#usage)
- [Data Creation](#data)
- [Training](#training)
- [Loading the model](#modelLoading)
- [Results](#Results)

## Installation

<ul>
  <li>! pip install spacy</li>
  <li>import spacy</li>
  <li>import re</li>
  <li>from spacy.language import Language</li>
</ul>


## Usage
<p> 
Bellow code can be used to generate custom data for training incase you have some templates and a list of some values you want to replace in the template.
like in my case <br>
![image](https://github.com/LLama2-Ai/spacyCustomNER/assets/142317270/749bbd13-a0ea-44b4-8e50-c0be2ce8be89)<br>
the <b>name</b> word will be replaced with the names provided in the names.zip file.  
</p>

## Data

## Traning

## ModelLoading

## Results
