# eyeBert

![image](https://github.com/user-attachments/assets/d4ba2344-1b19-4d00-98a7-5e3755567380)

Let's begin from the beginning, this project was done part of the Aravinda Eye Care Hackthon, through a group of four members, why don't you meet them? :)
# The Marauders
 - [Adithya Ananth](https://github.com/adithya-ananth)
 - [Arrepu Anirudh](https://github.com/AnirudhArrepu)
 - [Navaneeth Sivadas](https://github.com/NavaneethSivadas025)
 - [Niranjan M](https://github.com/all-coder)

This is **eyeBert**, a question-answering model based on the BERT-uncased model. The primary aim of this project was to create an AI-driven assistant that facilitates smooth onboarding for new candidates within an organization. eyeBert leverages the power of BERT to understand and respond to questions related to organizational structures, rules, procedures, and even specific departmental guidelines.

Built using streamlit, supported by ngroks and database stored on firebase.

## Features

- **Organizational Familiarity**: eyeBert helps new candidates become acquainted with the organization's organogram, including department-specific structures.
- **Policy Understanding**: The model is trained to answer questions about organizational rules, such as conference protocols, punctuality, dress codes, parking rules, canteen services, and more.
- **AECS Information**: eyeBert includes a specialized knowledge base drawn from the book "Infinite Vision," providing detailed insights into the organization of AECS.
- **Case Feedback**: The model can offer feedback to medical professionals about challenging cases seen during camps, emergency duties, or in the OPD, following the SOPs of the respective department.
- **Bilingual Output**: eyeBert provides the user with the option to toggle between Tamil and English languages, allowing trainees to receive information in Tamil as well.

## Model Details

- **Base Model**: bert-large-uncased-whole-word-masking-finetuned-squad
- **Training Data**: Custom dataset tailored to the specific needs of organizational onboarding and medical SOPs.
- **Implementation**: The model is fine-tuned using the Hugging Face Transformers library.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/eyeBert.git
   ```
2. Navigate to the project directory:
   ```bash
   cd eyeBert
