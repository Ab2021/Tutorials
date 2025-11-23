# Day 97: Healthcare & Medical Agents
## Deep Dive - Internal Mechanics & Advanced Reasoning

### Implementing a SOAP Note Generator

We will process a transcript into a structured medical note.

```python
class MedicalScribe:
    def __init__(self, client):
        self.client = client

    def generate_soap(self, transcript):
        prompt = f"""
        You are a medical scribe. Convert the following transcript into a SOAP note.
        
        Transcript:
        {transcript}
        
        Format:
        Subjective: Patient's complaints.
        Objective: Vital signs and exam findings.
        Assessment: Diagnosis.
        Plan: Treatment and follow-up.
        """
        
        return self.client.chat.completions.create(
            model="gpt-4", # Needs high intelligence
            messages=[{"role": "user", "content": prompt}]
        ).choices[0].message.content

# Usage
transcript = """
Dr: Hi Bob, what brings you in?
Bob: My chest hurts when I breathe.
Dr: Any fever?
Bob: Yeah, 101 last night.
Dr: Lungs sound crackly. Let's get a CXR.
"""
scribe = MedicalScribe(client)
print(scribe.generate_soap(transcript))
# Output:
# S: Chest pain on inspiration, Fever (101F).
# O: Crackles on auscultation.
# A: Possible Pneumonia.
# P: Order Chest X-Ray.
```

### Medical Entity Extraction (NER)

Extracting structured data from unstructured notes.
*   **Ontologies:** SNOMED-CT, ICD-10, RxNorm.
*   **Task:** Map "Heart attack" -> `ICD-10: I21.9`.
*   **Tools:** AWS Comprehend Medical, Google Cloud Healthcare API, or fine-tuned BERT.

### Med-PaLM & Specialized Models

General models (GPT-4) are good, but specialized models are better.
*   **Med-PaLM 2 (Google):** Fine-tuned on medical exams (USMLE). Achieves expert-level performance.
*   **BioGPT:** Trained on PubMed.
*   **Reasoning:** These models understand that "5mg" might be a lethal dose for a child but fine for an adult.

### Summary

*   **Summarization:** The core value prop. Turning 15 minutes of talk into 1 page of text.
*   **Structuring:** Converting text into billing codes (ICD-10).
