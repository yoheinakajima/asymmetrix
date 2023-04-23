# AsymmeTrix: Asymmetric Vector Embeddings for Directional Similarity Search
AsymmeTrix is a Python library that utilizes asymmetric cosine similarity to capture directional relationships between objects. It offers a new approach to vector search and embeddings, addressing the limitations of symmetric similarity measures.

Traditional vector embeddings typically use symmetric similarity measures like cosine similarity. While these measures work well in many applications, they may not accurately represent asymmetric relationships between objects.

AsymmeTrix addresses this issue by introducing a weighting factor based on a domain-specific asymmetric weighting function. This allows for more accurate and meaningful representations in applications where directionality is essential, such as job matching, recommendation systems, and knowledge graph completion.

# Features
Directionality: Captures asymmetric relationships between objects
Flexibility: Customizable asymmetric weighting function
Improved search and recommendations: Provides more accurate results in domains with directionality
Installation
To install AsymmeTrix, simply use pip:

pip install asymmetrix
Usage
Here's a basic example that demonstrates how to use AsymmeTrix for computing asymmetric cosine similarity between job postings and applicants:
```
import numpy as np
from asymmetrix import AsymmetricCosineSimilarity

# Define job posting and applicant feature vectors
job_postings = [
    np.array([0.9, 0.1]),
    np.array([0.1, 0.9]),
    np.array([0.5, 0.5]),
    np.array([0.8, 0.2])
]

applicants = [
    np.array([0.8, 0.2]),
    np.array([0.2, 0.8])
]

# Define asymmetric weighting function
def asymmetric_weighting_function(x):
    return 1.0 - x

# Initialize asymmetric cosine similarity
asymmetric_cosine_similarity = AsymmetricCosineSimilarity(asymmetric_weighting_function)

# Compute and analyze results
for i, applicant in enumerate(applicants):
    for j, job_posting in enumerate(job_postings):
        similarity = asymmetric_cosine_similarity(applicant, job_posting)
        print(f"Applicant {i+1} and Job Posting {j+1}: Asymmetric Cosine Similarity = {similarity:.3f}")

```
License
AsymmeTrix is released under the MIT License.
