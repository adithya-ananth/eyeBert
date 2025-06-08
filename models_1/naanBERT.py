## the code for the chatbot goes here ##
import torch
import textwrap
from transformers import BertForQuestionAnswering,BertTokenizer

model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
input_ids = [101, 2129, 2116, 11709, 2515, 14324, 1011, 2312, 2031, 1029, 102, 14324, 1011, 2312, 2003, 2428, 2502, 1012, 1012, 1012, 2009, 2038, 2484, 1011, 9014, 1998, 2019, 7861, 8270, 4667, 2946, 1997, 1015, 1010, 6185, 2549, 1010, 2005, 1037, 2561, 1997, 16029, 2213, 11709, 999, 10462, 2009, 2003, 1015, 1012, 4090, 18259, 1010, 2061, 5987, 2009, 2000, 2202, 1037, 3232, 2781, 2000, 8816, 2000, 2115, 15270, 2497, 6013, 1012, 102]




def answer_question(question, answer_text):
    input_ids = tokenizer.encode(question, answer_text)
    print('Query has {:,} tokens.\n'.format(len(input_ids)))

    sep_index = input_ids.index(tokenizer.sep_token_id)
    num_seg_a = sep_index + 1
    num_seg_b = len(input_ids) - num_seg_a
    segment_ids = [0] * num_seg_a + [1] * num_seg_b

    assert len(segment_ids) == len(input_ids)

    outputs = model(torch.tensor([input_ids]),
                    token_type_ids=torch.tensor([segment_ids]),
                    return_dict=True)

    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)

    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    answer = tokens[answer_start]

    for i in range(answer_start + 1, answer_end + 1):
        if tokens[i][0:2] == '##':
            answer += tokens[i][2:]
        else:
            answer += ' ' + tokens[i]

    print(answer + '"')

wrapper = textwrap.TextWrapper(width=80) 

abstract = "Aravinda Eyecare System's vision is to eliminate needless blindness. The mission of Aravinda Eyecare System is to eliminate needless blindness by providing compassionate and quality eye care affordable to all. NABH stands for National Accreditation Board for Hospitals and Healthcare providers. It is a constituent board of the Quality Council of India (QCI) and was set up to establish and operate an accreditation programme for healthcare organizations in India. Initially, it was a voluntary programme but now includes mandated entry-level standards. NABH accredits all types of healthcare delivery organizations, including blood banks, imaging centers, AYUSH, nursing homes, clinics, and more. The NABH accreditation process involves three levels: Pre-accreditation entry level, Pre-accreditation progressive level, and Full accreditation. Healthcare organizations can be classified into two main categories: Small healthcare organizations (SHCO), which have less than 50 beds, and larger healthcare organizations (HCO), which have more than 50 beds. Small healthcare organizations (SHCO) under Aravinda Eyecare System are located in Tirupur, Dindigul, Tuticorin, and Udumalpet. Larger healthcare organizations (HCO) are located in Madurai, Tirunelveli, Coimbatore, Pondicherry, Theni, and Salem. The constituents of accreditation include Structure, Process, and Outcome. NABH focuses on ensuring safety in several areas including patient safety, employee safety, visitor safety, and physical safety"
wrapper.fill(bert_abstract)

question = "What is NABH?"

answer_question(question, bert_abstract)


