import torch
from transformers import BertTokenizerFast, BertForQuestionAnswering

model_path = "./fine-tuned-bert"
model = BertForQuestionAnswering.from_pretrained(model_path)
tokenizer = BertTokenizerFast.from_pretrained(model_path)

model.eval()

context = ("Aravinda Eyecare System's vision is to eliminate needless blindness. The mission of Aravinda Eyecare. System is to eliminate needless blindness by providing compassionate and quality eye care affordable to all. NABH stands for National Accreditation Board for Hospitals and Healthcare providers. It is a constituent board of the Quality Council of India (QCI) and was set up to establish and operate an accreditation programme for healthcare organizations in India. Initially, it was a voluntary programme but now includes mandated entry-level standards. NABH accredits all types of healthcare delivery organizations, including blood banks, imaging centers, AYUSH, nursing homes, clinics, and more. The NABH accreditation process involves three levels: Pre-accreditation entry level, Pre-accreditation progressive level, and Full accreditation. Healthcare organizations can be classified into two main categories: Small healthcare organizations (SHCO), which have less than 50 beds, and larger healthcare organizations (HCO), which have more than 50 beds. Small healthcare organizations (SHCO) under Aravinda Eyecare System are located in Tirupur, Dindigul, Tuticorin, and Udumalpet. Larger healthcare organizations (HCO) are located in Madurai, Tirunelveli, Coimbatore, Pondicherry, Theni, and Salem. The constituents of accreditation include Structure, Process, and Outcome.")

questions = [
    "Where are Small healthcare organizations located?",
    "What is the vision of Aravinda Eyecare System?",
    "What are the constituents of accreditation?",
    "Where are Larger healthcare organizations located?",
    "What is the mission of Aravinda Eyecare System?",
    "What is the NABH process?",
    "What are the constituents of NABH?",
    "What is a Healthcare organization?",
    "What types of organizations does NABH accredit?",
]

def post_process_answer(answer, context):
    while answer and not answer.endswith(('.', '!', '?')):
        answer_end_index = context.find('.', len(answer))
        if answer_end_index == -1:  
            break
        answer = context[:answer_end_index + 1]
    return answer.strip()

def get_answer_from_context(question, context):
    inputs = tokenizer(
        question,
        context,
        max_length=384,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
        return_offsets_mapping=True
    )

    offset_mapping = inputs.pop("offset_mapping")

    with torch.no_grad():
        outputs = model(**inputs)

    start_logits = outputs.start_logits
    end_logits = outputs.end_logits

    start_probs = torch.softmax(start_logits, dim=-1)
    end_probs = torch.softmax(end_logits, dim=-1)

    start_idx = torch.argmax(start_probs)
    end_idx = torch.argmax(end_probs)

    if start_idx > end_idx:
        end_idx = start_idx + torch.argmax(end_probs[0, start_idx:start_idx + 15])
    
    for i in range(end_idx + 1, len(end_probs[0])):
        if end_probs[0][i] > 0.5:  
            end_idx = i
    
    answer_start = offset_mapping[0][start_idx][0].item()
    answer_end = offset_mapping[0][end_idx][1].item()
    answer = context[answer_start:answer_end]

    return post_process_answer(answer, context)

for question in questions:
    answer = get_answer_from_context(question, context)
    
    print(f"Question: {question}")
    print(f"Answer: {answer}\n")
