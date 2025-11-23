# Day 48: Security & Compliance
## Deep Dive - Internal Mechanics & Advanced Reasoning

### 1. Complete Security Pipeline

```python
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import re
import openai
from typing import Optional

app = FastAPI()

class SecurityPipeline:
    def __init__(self):
        self.pii_detector = PIIDetector()
        self.prompt_injection_detector = PromptInjectionDetector()
        self.content_moderator = ContentModerator()
        self.audit_logger = AuditLogger()
    
    async def process_request(self, user_id: str, prompt: str) -> dict:
        """Process request through security pipeline."""
        # 1. Audit log
        request_id = self.audit_logger.log_request(user_id, prompt)
        
        # 2. Input validation
        if len(prompt) > 10000:
            raise HTTPException(400, "Prompt too long")
        
        # 3. PII detection in input
        pii_detected = self.pii_detector.detect(prompt)
        if pii_detected:
            self.audit_logger.log_security_event(
                request_id, "pii_in_input", pii_detected
            )
            # Redact PII
            prompt = self.pii_detector.redact(prompt)
        
        # 4. Prompt injection detection
        if self.prompt_injection_detector.is_malicious(prompt):
            self.audit_logger.log_security_event(
                request_id, "prompt_injection_attempt"
            )
            raise HTTPException(403, "Malicious prompt detected")
        
        # 5. Content moderation (input)
        if self.content_moderator.is_harmful(prompt):
            self.audit_logger.log_security_event(
                request_id, "harmful_input"
            )
            raise HTTPException(403, "Harmful content detected")
        
        # 6. Generate response
        response = await self._generate(prompt)
        
        # 7. Content moderation (output)
        if self.content_moderator.is_harmful(response):
            self.audit_logger.log_security_event(
                request_id, "harmful_output"
            )
            response = "I cannot provide that information."
        
        # 8. PII detection in output
        pii_in_output = self.pii_detector.detect(response)
        if pii_in_output:
            self.audit_logger.log_security_event(
                request_id, "pii_in_output", pii_in_output
            )
            response = self.pii_detector.redact(response)
        
        # 9. Log response
        self.audit_logger.log_response(request_id, response)
        
        return {"response": response, "request_id": request_id}
    
    async def _generate(self, prompt: str) -> str:
        """Generate response with LLM."""
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

security_pipeline = SecurityPipeline()

@app.post("/generate")
async def generate(user_id: str, prompt: str):
    return await security_pipeline.process_request(user_id, prompt)
```

### 2. PII Detection and Redaction

```python
import re
from typing import Dict, List

class PIIDetector:
    def __init__(self):
        self.patterns = {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "phone": r'\b(?:\+?1[-.]?)?\(?([0-9]{3})\)?[-.]?([0-9]{3})[-.]?([0-9]{4})\b',
            "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
            "credit_card": r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            "ip_address": r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
            "name": r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'  # Simple name pattern
        }
        
        self.redaction_map = {
            "email": "[EMAIL]",
            "phone": "[PHONE]",
            "ssn": "[SSN]",
            "credit_card": "[CREDIT_CARD]",
            "ip_address": "[IP]",
            "name": "[NAME]"
        }
    
    def detect(self, text: str) -> Dict[str, List[str]]:
        """Detect all PII in text."""
        detected = {}
        
        for pii_type, pattern in self.patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                detected[pii_type] = matches
        
        return detected
    
    def redact(self, text: str) -> str:
        """Redact all PII from text."""
        redacted = text
        
        for pii_type, pattern in self.patterns.items():
            replacement = self.redaction_map[pii_type]
            redacted = re.sub(pattern, replacement, redacted)
        
        return redacted
    
    def anonymize(self, text: str) -> str:
        """Anonymize PII with fake but realistic data."""
        import faker
        fake = faker.Faker()
        
        # Replace with fake data
        text = re.sub(self.patterns["email"], fake.email(), text)
        text = re.sub(self.patterns["phone"], fake.phone_number(), text)
        text = re.sub(self.patterns["name"], fake.name(), text)
        
        return text
```

### 3. Prompt Injection Detection

```python
class PromptInjectionDetector:
    def __init__(self):
        self.malicious_patterns = [
            r'ignore\s+(previous|all)\s+instructions',
            r'disregard\s+.*\s+instructions',
            r'forget\s+.*\s+instructions',
            r'system\s+prompt',
            r'reveal\s+.*\s+prompt',
            r'you\s+are\s+now',
            r'new\s+instructions',
            r'override\s+.*\s+rules'
        ]
        
        self.encoding_patterns = [
            r'base64',
            r'rot13',
            r'hex\s+encoded'
        ]
    
    def is_malicious(self, prompt: str) -> bool:
        """Check if prompt contains injection attempt."""
        prompt_lower = prompt.lower()
        
        # Check for direct injection patterns
        for pattern in self.malicious_patterns:
            if re.search(pattern, prompt_lower):
                return True
        
        # Check for encoding attempts
        for pattern in self.encoding_patterns:
            if re.search(pattern, prompt_lower):
                return True
        
        # Check for excessive special characters (obfuscation)
        special_char_ratio = sum(1 for c in prompt if not c.isalnum() and not c.isspace()) / len(prompt)
        if special_char_ratio > 0.3:
            return True
        
        return False
    
    def get_safe_prompt(self, prompt: str) -> str:
        """Sanitize prompt by removing malicious parts."""
        # Remove lines containing malicious patterns
        lines = prompt.split('\n')
        safe_lines = []
        
        for line in lines:
            if not self.is_malicious(line):
                safe_lines.append(line)
        
        return '\n'.join(safe_lines)
```

### 4. Content Moderation

```python
import openai

class ContentModerator:
    def __init__(self):
        self.categories = [
            "hate", "hate/threatening",
            "self-harm", "sexual", "sexual/minors",
            "violence", "violence/graphic"
        ]
    
    def is_harmful(self, text: str, threshold: float = 0.5) -> bool:
        """Check if content is harmful using OpenAI Moderation API."""
        try:
            response = openai.Moderation.create(input=text)
            results = response["results"][0]
            
            # Check if any category exceeds threshold
            for category in self.categories:
                if results["category_scores"].get(category, 0) > threshold:
                    return True
            
            return results["flagged"]
        
        except Exception as e:
            # Fail open (allow) or fail closed (block)?
            # Fail closed is safer
            return True
    
    def get_violation_categories(self, text: str) -> List[str]:
        """Get list of violated categories."""
        response = openai.Moderation.create(input=text)
        results = response["results"][0]
        
        violations = []
        for category in self.categories:
            if results["categories"].get(category, False):
                violations.append(category)
        
        return violations
```

### 5. Audit Logging

```python
import json
import logging
from datetime import datetime

class AuditLogger:
    def __init__(self):
        self.logger = logging.getLogger("audit")
        self.logger.setLevel(logging.INFO)
        
        # Log to file
        handler = logging.FileHandler("audit.log")
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)
    
    def log_request(self, user_id: str, prompt: str) -> str:
        """Log incoming request."""
        request_id = str(uuid.uuid4())
        
        self.logger.info(json.dumps({
            "event": "request",
            "request_id": request_id,
            "user_id": user_id,
            "prompt_length": len(prompt),
            "timestamp": datetime.utcnow().isoformat()
        }))
        
        return request_id
    
    def log_response(self, request_id: str, response: str):
        """Log response."""
        self.logger.info(json.dumps({
            "event": "response",
            "request_id": request_id,
            "response_length": len(response),
            "timestamp": datetime.utcnow().isoformat()
        }))
    
    def log_security_event(self, request_id: str, event_type: str, details: dict = None):
        """Log security event."""
        self.logger.warning(json.dumps({
            "event": "security",
            "request_id": request_id,
            "type": event_type,
            "details": details,
            "timestamp": datetime.utcnow().isoformat()
        }))
```

### 6. GDPR Compliance

```python
class GDPRCompliance:
    def __init__(self, db):
        self.db = db
    
    def export_user_data(self, user_id: str) -> dict:
        """Export all data for user (GDPR right to access)."""
        return {
            "user_id": user_id,
            "requests": self.db.get_user_requests(user_id),
            "preferences": self.db.get_user_preferences(user_id),
            "created_at": self.db.get_user_created_at(user_id)
        }
    
    def delete_user_data(self, user_id: str):
        """Delete all data for user (GDPR right to deletion)."""
        self.db.delete_user_requests(user_id)
        self.db.delete_user_preferences(user_id)
        self.db.delete_user_account(user_id)
        
        # Log deletion for audit
        logging.info(f"Deleted all data for user {user_id}")
    
    def anonymize_old_data(self, days: int = 90):
        """Anonymize data older than N days."""
        old_requests = self.db.get_requests_older_than(days)
        
        for request in old_requests:
            # Remove PII
            request["user_id"] = "ANONYMIZED"
            request["prompt"] = "[REDACTED]"
            request["response"] = "[REDACTED]"
            
            self.db.update_request(request)
```
