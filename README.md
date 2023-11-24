# SpacyCustom-NER

Some time ago I faced an issue of data masking, Instead of going with state-of-the-art models(Transformers) and relying 100% on probability, 
I decided to go with spacy (rule-based + ML-based), To decrease the chances of error.

And got inspiring results even on the first try. 
### Problem: Mask the personal information from the email
![image](https://github.com/LLama2-Ai/spacyCustomNER/assets/142317270/587d58f7-ac4b-40fe-9167-6c2db7375cc7)

### The following points still required planning.
<ul>
  <li> <b>Data:</b> to capture the names correctly, As names cannot be done with the rule-based approach </li>
  <li> <b> Regular expressions:</b> to fetch phoneNo,Dates,Email addresses, Card details  </li>
</ul>

