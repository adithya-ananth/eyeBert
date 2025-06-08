import streamlit as st
import torch
from transformers import BertTokenizer, BertForQuestionAnswering, AutoTokenizer, AutoModelForSeq2SeqLM
from pyngrok import ngrok

model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

contexts = {
    "Introduction": "Aravind Eyecare System envisions eliminating needless blindness through compassionate and quality eye care that is affordable to all. The organization is accredited by the National Accreditation Board for Hospitals and Healthcare Providers (NABH), which is a constituent board of the Quality Council of India (QCI). NABH was established to operate an accreditation program for healthcare organizations in India, initially voluntary but now mandated with entry-level standards. It accredits various health delivery organizations, including blood banks, imaging centers, AYUSH facilities, nursing homes, and clinics. The NABH accreditation process includes pre-accreditation entry level, pre-accreditation progressive level, and full accreditation.",

    "Rights and Responsibilities": "At Aravind, patients have rights such as respectful treatment, choice in their care, access to medical information, informed decision-making, the ability to seek a second opinion, confidentiality, refusing treatment, ensuring a healthy environment, and filing complaints. They are responsible for providing accurate information, maintaining a healthy lifestyle, respecting others, handling hospital property safely, following hospital rules, safeguarding personal belongings, and adhering to treatment plans. Complaints can be directed to the department manager, Coordinator, or Patient Care Manager. The quality policy focuses on providing timely, safe, high-quality eye care and ensuring high patient satisfaction.",

    "Bio-Medical Waste (BMW) Management": "Bio-medical waste at Aravind is categorized into infected and non-infected types. Infected waste includes anatomical waste (human body parts, tissues), plastic and non-plastic waste (syringes, IV tubes), glass waste, and sharps (needles, blades). Non-infected waste includes chemical waste such as expired drugs. For needle stick injuries, the protocol involves washing the site, applying a dressing, reporting to the infection control nurse, and consulting a physician if necessary. Blood and body fluid spills are managed with caution boards, PPE, and sodium hypochlorite solution. Mercury spill management includes isolating the area, removing metal objects, and collecting mercury beads.",

    "Leaves, Dress Code, Reference Books": "Male staff at Aravind must wear light-colored shirts with pants and mandatory shoes, keeping hair neat and clean. Female staff should wear clean, simple dresses suitable for their role, such as saris or churidars. Jeans, leggings, shorts, and sleeveless tops are not allowed. Recommended books for ophthalmology residents include titles by A.K. Khurana, Kanski, Ryan, and others. Postgraduates receive 15 days of leave per year, while fellows have varying leave allowances depending on the duration of their fellowship.",

    "Thesis Submission Process": "The thesis submission process at Aravind requires the following documents: a covering letter, confession statement, IHEC protocol submission, informed consent, project proposal, and curriculum vitae.",

    "Aqueous Misdirection Syndrome": "Aqueous misdirection syndrome involves various treatments, starting with cycloplegics and medications like acetazolamide, alpha agonists, or beta blockers. Ensuring a patent iridotomy is crucial. If initial treatments fail, additional interventions like vitrectomy or YAG laser capsulotomy may be considered. Persistent cases may require surgical options such as vitrectomy or lens extraction.",

    "Standard Automated Perimetry (SAP) in Glaucoma Management": "Standard Automated Perimetry (SAP) is crucial in glaucoma management, providing objective assessments to detect early field loss. The Humphrey perimeter's 24-2 test pattern is commonly used. SITA Standard and SITA Fast algorithms are popular, though SITA Fast can introduce variability. Test reliability is evaluated using indices like false positives and fixation losses. The Glaucoma Hemifield Test (GHT) helps identify glaucomatous conditions by requiring at least three clustered points of significantly reduced sensitivity.",

    "Goldmann Size V Stimulus and Hodapp Classification in Glaucoma": "In advanced glaucoma, the Goldmann size V stimulus targets the remaining visual field. The Hodapp Classification stages glaucoma into early, moderate, and advanced stages based on mean deviation (MD) and the number of points below the 5% probability level. This classification helps tailor management according to the severity of the disease.",

    "Gonioscopy Techniques and Instruments in Glaucoma": "Gonioscopy is essential for examining the anterior chamber angle in glaucoma diagnosis and management. It can be done using direct or indirect methods. The Goldmann lens, with its mirrors, is commonly used, offering enhanced image quality. A four-mirror lens allows simultaneous visualization of all quadrants. Dynamic indentation gonioscopy is used to assess appositional angle-closure by pushing the iris back. Proper slit lamp use and lens orientation are crucial for accurate gonioscopy.",


    "Preoperative Protocols and Postoperative Care in Glaucoma Surgery": "Preoperative protocols for glaucoma surgery at Aravind include patient admission, thorough examination, medical history review, and informed consent. Preoperative investigations like vision tests and intraocular pressure measurements are conducted. Postoperative care focuses on monitoring recovery, managing complications, and ensuring optimal visual results to achieve successful surgery outcomes."
}

def get_context(key):
    return contexts.get(key, "Context not found.")

def answer_question(question, answer_text):
    '''
    Takes a `question` string and an `answer_text` string (which contains the
    answer), and identifies the words within the `answer_text` that are the
    answer. Prints them out.
    '''

    input_ids = tokenizer.encode(question, answer_text)

    print('Query has {:,} tokens.\n'.format(len(input_ids)))

    sep_index = input_ids.index(tokenizer.sep_token_id)

    num_seg_a = sep_index + 1

    num_seg_b = len(input_ids) - num_seg_a

    segment_ids = [0]*num_seg_a + [1]*num_seg_b

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

    return answer

def translate(text):
    model_name = "Hemanth-thunder/english-tamil-mt"
    tokenizer_tamil = AutoTokenizer.from_pretrained(model_name)
    model_tamil = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    inputs = tokenizer_tamil.encode(text, return_tensors="pt")
    outputs = model_tamil.generate(inputs, max_length=512, num_beams=5, early_stopping=True)
    return tokenizer_tamil.decode(outputs[0], skip_special_tokens=True)

def main():
    st.title("naanBERT")

    context_names = list(contexts.keys())

    language = st.radio("Language:", ("English", "Tamil"))
    selected_context = st.selectbox("Choose a context", context_names)

    if selected_context:
        context_data = get_context(selected_context)

    question = st.text_input("Ask your question:")
    if question and selected_context:
        answer = answer_question(question, context_data)
        if language == "Tamil":
            answer = translate(answer)
        st.write("Answer:", answer)
        
def admin():
    st.title("Admin: Add New Context")
    
    new_context_name = st.text_input("Context Name")
    new_context_content = st.text_area("Context")
  
    if st.button("Add Context"):
        if new_context_name and new_context_content:
            contexts.update({new_context_name:new_context_content})
            print(contexts[new_context_name])
            st.success("New context added!")
            st.rerun()
        else:
            st.warning("Please provide both context name and content.")

st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Main", "Admin"])

if page == "Main":
    main()
elif page == "Admin":
    admin()

