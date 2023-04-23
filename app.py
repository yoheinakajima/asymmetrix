import numpy as np

def asymmetric_cosine_similarity(job_posting, applicant):
    dot_product = np.dot(job_posting, applicant)
    magnitude_posting = np.linalg.norm(job_posting)
    magnitude_applicant = np.linalg.norm(applicant)
    cosine_similarity = dot_product / (magnitude_posting * magnitude_applicant)
    
    skills_posting = np.sum(job_posting)
    skills_applicant = np.sum(applicant)
    asymmetric_weight = skills_applicant / skills_posting
    
    asymmetric_similarity = cosine_similarity * asymmetric_weight
    return asymmetric_similarity

# Example job postings and applicants represented by feature vectors
job_postings = [
    np.array([1, 0, 1]),  # Job posting 1: Requires skill A and skill C
    np.array([0, 1, 1])   # Job posting 2: Requires skill B and skill C
]

applicants = [
    np.array([1, 0, 1]),  # Applicant 1: Has skill A and skill C
    np.array([1, 1, 1])   # Applicant 2: Has skill A, skill B, and skill C
]

# Compute asymmetric cosine similarity between job postings and applicants
for i, job_posting in enumerate(job_postings):
    for j, applicant in enumerate(applicants):
        similarity = asymmetric_cosine_similarity(job_posting, applicant)
        print(f"Job posting {i + 1} and applicant {j + 1}:")
        print(f"  - Job posting: {job_posting}")
        print(f"  - Applicant: {applicant}")
        print(f"  - Asymmetric cosine similarity: {similarity:.4f}")
        print()
