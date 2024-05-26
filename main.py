from flask import Flask, request, jsonify
from langchain_community.llms import LlamaCpp
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel

# cosine_similarity
import torch
from torch.nn.functional import cosine_similarity
import os 
app = Flask(__name__)

n_gpu_layers = 0
n_batch = 1024


llm = LlamaCpp(
    model_path="Phi-3-mini-4k-instruct-q4.gguf",  # path to GGUF file
    temperature=0.1,
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    verbose=True,
    n_ctx=4096
)
model0 = AutoModel.from_pretrained('sentence-transformers/paraphrase-TinyBERT-L6-v2')

model = SentenceTransformer('sentence-transformers/paraphrase-TinyBERT-L6-v2')

file_size = os.stat('Phi-3-mini-4k-instruct-q4.gguf')
print("model size ====> :", file_size.st_size, "bytes")


@app.route('/cv', methods=['POST'])
def get_skills():
    cv_body = request.json.get('cv_body')

    # Simple inference example
    output = llm(
        f"<|user|>\n{cv_body}<|end|>\n<|assistant|>Can you list the skills mentioned in the CV?<|end|>",
        max_tokens=256,  # Generate up to 256 tokens
        stop=["<|end|>"], 
        echo=True,  # Whether to echo the prompt
    )

    return jsonify({'skills': output})

@app.get('/')
def health():
    return jsonify({'status': 'Worked'})

# we will make here post request to compare between lists of skills one has employee just one text and the other has the of jobs has many texts
# the llm will say the most similar job to the cv
@app.route('/compare', methods=['POST'])
def compare():
    employee_skills = request.json.get('employee_skills')# string
    #example: employee_skills = "<|assistant|> Certainly! Based on the provided information, here are the relevant skills listed in my CV:\n\n1. Python programming language proficiency\n2. Java development expertise\n3. Web development experience (including front-end and back-end technologies)\n4. Proficient with machine learning algorithms and frameworks\n5. Strong problem-solving abilities\n6. Excellent communication skills, both written and verbal\n7. Agile methodology adherence\n8. Familiarity with version control systems (e.g., Git)\n9. Experience in designing scalable software architectures\n10. Proficient in testing methodologies such as unit tests, integration tests, and end-to-end tests\n11. Continuous learning mindset to stay updated on the latest technological advancements\n12. Strong collaboration skills for effective teamwork\n\nThese skills enable me to contribute significantly to an employer's objectives by developing innovative features that leverage cutting-edge technology, optimizing software performance through efficient coding practices and algorithmic improvements, enhancing code quality with rigorous testing methodologies, and fostering a collaborative work environment""
    jobs_skills = request.json.get('jobs_skills')
    #example: jobs_skills = ["<|assistant|> Sure! Here are the skills required for the Software Engineer position:\n\n1. Proficiency in Python programming language\n2. Experience with Java development\n3. Knowledge of web development technologies (e.g., HTML, CSS, JavaScript)\n4. Familiarity with machine learning algorithms and frameworks (e.g., TensorFlow, PyTorch)\n5. Strong problem-solving skills\n6. Effective communication abilities\n7. Agile methodology understanding\n8. Version control system expertise (e.g., Git)\n9. Software architecture design experience\n10. Testing methodologies knowledge (unit tests, integration tests, end-to-end tests)\n11. Continuous learning mindset for staying updated on technological advancements\n12. Collaboration skills for effective teamwork\n\nThese skills are essential for the Software Engineer role to contribute to the development of innovative software solutions, optimize performance, ensure code quality, and foster a collaborative work environment.", "<|assistant|> Certainly! Here are the skills required for the Data Scientist position:\n\n1. Proficiency in Python programming language\n2. Experience with data analysis and visualization tools (e.g., Pandas, Matplotlib)\n3. Knowledge of machine learning algorithms and statistical modeling techniques\n4. Strong problem-solving and analytical skills\n5. Effective communication abilities\n6. Agile methodology understanding\n7. Version control system expertise (e.g., Git)\n8. Data preprocessing and cleaning experience\n9. Model evaluation and optimization skills\n10. Continuous learning mindset for staying updated on data science advancements\n11. Collaboration skills for effective teamwork\n\nThese skills are essential for the Data Scientist role to analyze data, develop predictive models, optimize algorithms, and collaborate with cross-functional teams."]
    if not isinstance(jobs_skills, list) or not all(isinstance(skill, str) for skill in jobs_skills):
        raise ValueError("jobs_skills must be a list of strings")
    job_embeddings = model.encode(jobs_skills)
    employee_embeddings = model.encode(employee_skills)
    sim = []
    employee_embeddings_tensor = torch.from_numpy(employee_embeddings).unsqueeze(0)
    for job_e in job_embeddings:
        job_e_tensor = torch.from_numpy(job_e).unsqueeze(0)
        sim.append(cosine_similarity(employee_embeddings_tensor, job_e_tensor,dim=1))
    max_sim = max(sim)
    index = sim.index(max_sim)
    return jsonify({'job': jobs_skills[index]})

if __name__ == '__main__':
    app.run()