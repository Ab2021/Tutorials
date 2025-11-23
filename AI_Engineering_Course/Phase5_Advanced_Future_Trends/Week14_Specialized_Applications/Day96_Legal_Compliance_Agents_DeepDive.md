# Day 96: Legal & Compliance Agents
## Deep Dive - Internal Mechanics & Advanced Reasoning

### Implementing a Contract Extractor

We will build an agent that extracts key fields from a Lease Agreement.

```python
from pydantic import BaseModel, Field

class LeaseTerms(BaseModel):
    landlord: str = Field(..., description="Name of the landlord")
    tenant: str = Field(..., description="Name of the tenant")
    rent_amount: float = Field(..., description="Monthly rent in USD")
    start_date: str = Field(..., description="Lease start date")
    termination_notice: int = Field(..., description="Days notice required to terminate")

class LegalAgent:
    def __init__(self, client):
        self.client = client

    def extract(self, contract_text):
        prompt = f"""
        Extract the following terms from the contract.
        Contract:
        {contract_text[:4000]} # Truncated for demo
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            functions=[{
                "name": "save_terms",
                "parameters": LeaseTerms.model_json_schema()
            }],
            function_call={"name": "save_terms"}
        )
        
        return response.choices[0].message.function_call.arguments

# Usage
contract = "This Lease is made between John Doe (Landlord) and Jane Smith (Tenant). Rent is $2,000..."
agent = LegalAgent(client)
print(agent.extract(contract))
```

### Retrieval for Case Law (RAG)

Legal RAG is different.
*   **Chunking:** You can't split a statute in the middle. You must chunk by "Section" or "Article".
*   **Citation:** The model must output `(See Section 4.2(a))`.
*   **Hierarchy:** Laws have hierarchy (Constitution > Statute > Regulation). The retrieval must respect this.

### Redlining (Diff Generation)

The agent suggests changes.
*   *Input:* "The Tenant shall pay for all repairs."
*   *Instruction:* "Make this fairer to the Tenant."
*   *Output:* "The Tenant shall pay for repairs **caused by their negligence**."
*   *Format:* Output Markdown diffs (`-old`, `+new`) or Track Changes XML.

### Summary

*   **Structured Output:** Legal data must be structured (JSON), not free text.
*   **Verifiability:** Every extraction should point to the source byte offset in the PDF.
