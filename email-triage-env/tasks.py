"""
tasks.py — Email datasets for 3 difficulty levels.

Each task level returns a list of email dicts, each with:
  - id:               Unique email ID
  - subject:          Subject line
  - body:             Email body text
  - sender:           Sender email address
  - metadata:         Dict of extra info (date, headers, etc.)
  - ground_truth:     Dict with correct "department" and "priority"

Difficulty scaling:
  EASY   — 3 emails, obvious keywords, clear categories
  MEDIUM — 5 emails, some ambiguity, more departments used
  HARD   — 8 emails, misleading subjects, cross-department, edge cases
"""

from typing import List, Dict, Any


def get_easy_tasks() -> List[Dict[str, Any]]:
    """3 straightforward emails with obvious keywords."""
    return [
        {
            "id": "easy-001",
            "subject": "Billing question about my last invoice",
            "body": (
                "Hi,\n\n"
                "I received invoice #INV-2026-1042 last week and the total "
                "seems higher than what I expected. I was charged $249.99 but "
                "my plan is the Basic tier at $19.99/month. Could you please "
                "review and correct this?\n\n"
                "My account ID is ACC-78432.\n\n"
                "Thanks,\nJohn Smith"
            ),
            "sender": "john.smith@example.com",
            "metadata": {
                "date": "2026-04-01T10:30:00Z",
                "account_id": "ACC-78432",
                "has_attachment": False,
            },
            "ground_truth": {
                "department": "billing",
                "priority": "high",
            },
        },
        {
            "id": "easy-002",
            "subject": "Cannot login to my account — error 500",
            "body": (
                "Hello Support,\n\n"
                "I've been trying to log into my dashboard for the past hour "
                "but keep getting an 'Internal Server Error (500)' page. I've "
                "tried clearing my browser cache and using a different browser "
                "(Chrome and Firefox) but the problem persists.\n\n"
                "My username is sarah.jones and I'm on the Enterprise plan.\n\n"
                "This is blocking my entire team.\n\n"
                "Best,\nSarah Jones"
            ),
            "sender": "sarah.jones@bigcorp.com",
            "metadata": {
                "date": "2026-04-01T11:15:00Z",
                "account_id": "ACC-91205",
                "has_attachment": False,
            },
            "ground_truth": {
                "department": "technical_support",
                "priority": "urgent",
            },
        },
        {
            "id": "easy-003",
            "subject": "Interested in upgrading to Enterprise plan",
            "body": (
                "Hi Sales Team,\n\n"
                "Our company has been using the Pro plan for 6 months and "
                "we're very happy with the product. We'd like to explore the "
                "Enterprise plan for our upcoming expansion. We currently have "
                "50 users and expect to grow to 200 by Q3.\n\n"
                "Could you schedule a call to discuss pricing and features?\n\n"
                "Regards,\nMike Chen\nCTO, TechFlow Inc."
            ),
            "sender": "mike.chen@techflow.io",
            "metadata": {
                "date": "2026-04-01T14:00:00Z",
                "account_id": "ACC-44321",
                "has_attachment": False,
            },
            "ground_truth": {
                "department": "sales",
                "priority": "medium",
            },
        },
    ]


def get_medium_tasks() -> List[Dict[str, Any]]:
    """5 emails with some ambiguity and more departments."""
    return [
        {
            "id": "med-001",
            "subject": "Refund request for double charge",
            "body": (
                "Hi,\n\n"
                "I was charged twice for my March subscription — once on "
                "March 1st ($49.99) and again on March 3rd ($49.99). I only "
                "authorized one payment. Please refund the duplicate charge "
                "to my Visa ending in 4521.\n\n"
                "Order IDs: ORD-88201 and ORD-88203.\n\n"
                "Thanks,\nAmanda Lee"
            ),
            "sender": "amanda.lee@gmail.com",
            "metadata": {
                "date": "2026-04-02T09:00:00Z",
                "account_id": "ACC-55102",
                "has_attachment": True,
                "attachment_name": "bank_statement.pdf",
            },
            "ground_truth": {
                "department": "billing",
                "priority": "high",
            },
        },
        {
            "id": "med-002",
            "subject": "Feature not working after update",
            "body": (
                "Hello,\n\n"
                "After the latest update (v3.2.1), the export-to-PDF feature "
                "is broken. When I click 'Export', the spinner runs for 30 "
                "seconds and then shows 'Export failed: timeout'. This worked "
                "fine before the update.\n\n"
                "I'm on macOS 14.3, Chrome 124. My team of 15 people all "
                "have the same issue.\n\n"
                "Steps to reproduce:\n"
                "1. Open any report\n"
                "2. Click 'Export' → 'PDF'\n"
                "3. Wait 30s → error\n\n"
                "Raj Patel"
            ),
            "sender": "raj.patel@startup.co",
            "metadata": {
                "date": "2026-04-02T10:30:00Z",
                "account_id": "ACC-67890",
                "has_attachment": True,
                "attachment_name": "error_screenshot.png",
            },
            "ground_truth": {
                "department": "technical_support",
                "priority": "high",
            },
        },
        {
            "id": "med-003",
            "subject": "Question about remote work policy",
            "body": (
                "Hi HR,\n\n"
                "I'm a new employee (started last Monday) and I had a question "
                "about the remote work policy. My offer letter mentions "
                "'hybrid flexible' but my manager said we need to be in office "
                "4 days a week. Could you clarify what the official policy is?\n\n"
                "Also, I haven't received my laptop yet — is there a form I "
                "need to fill out?\n\n"
                "Thanks,\nLisa Wang"
            ),
            "sender": "lisa.wang@company-internal.com",
            "metadata": {
                "date": "2026-04-02T11:45:00Z",
                "employee_id": "EMP-2026-089",
                "has_attachment": False,
            },
            "ground_truth": {
                "department": "human_resources",
                "priority": "medium",
            },
        },
        {
            "id": "med-004",
            "subject": "Partnership opportunity — AI integration",
            "body": (
                "Dear Team,\n\n"
                "I'm the Head of Partnerships at DataVision AI. We've built "
                "an AI-powered analytics module that integrates with platforms "
                "like yours via API. Several of your competitors already use "
                "our solution.\n\n"
                "Would your sales or partnerships team be available for a "
                "30-minute call this week? We believe there's a strong "
                "mutual opportunity here.\n\n"
                "Best regards,\nDr. Priya Sharma\nHead of Partnerships, "
                "DataVision AI"
            ),
            "sender": "priya.sharma@datavision.ai",
            "metadata": {
                "date": "2026-04-02T13:00:00Z",
                "has_attachment": True,
                "attachment_name": "partnership_deck.pdf",
            },
            "ground_truth": {
                "department": "sales",
                "priority": "medium",
            },
        },
        {
            "id": "med-005",
            "subject": "How do I reset my password?",
            "body": (
                "Hi,\n\n"
                "I forgot my password and the 'Reset Password' link on your "
                "website sends me to a page that says 'Service Unavailable'. "
                "Can someone manually reset it for me? My email on file is "
                "this one.\n\n"
                "Thanks,\nTom Brooks"
            ),
            "sender": "tom.brooks@yahoo.com",
            "metadata": {
                "date": "2026-04-02T15:20:00Z",
                "account_id": "ACC-33210",
                "has_attachment": False,
            },
            "ground_truth": {
                "department": "technical_support",
                "priority": "medium",
            },
        },
    ]


def get_hard_tasks() -> List[Dict[str, Any]]:
    """8 emails with misleading subjects, ambiguity, edge cases."""
    return [
        {
            "id": "hard-001",
            "subject": "URGENT: System completely down!!!",
            "body": (
                "Hey,\n\n"
                "Just wanted to let you know that I can't find the option to "
                "change my notification preferences in the settings page. "
                "It's not really urgent but the subject got your attention "
                "didn't it? :)\n\n"
                "Where do I go to turn off email notifications?\n\n"
                "Cheers,\nDave"
            ),
            "sender": "dave.wilson@funmail.com",
            "metadata": {
                "date": "2026-04-03T08:00:00Z",
                "account_id": "ACC-12345",
                "has_attachment": False,
            },
            "ground_truth": {
                "department": "general_inquiry",
                "priority": "low",
            },
        },
        {
            "id": "hard-002",
            "subject": "Invoice question",
            "body": (
                "Hi,\n\n"
                "I'm trying to integrate your payment API into our system but "
                "the documentation for the /invoices endpoint returns a 404. "
                "The REST API docs page seems outdated — it references v1 but "
                "your changelog says you're on v3.\n\n"
                "Can your engineering team update the docs? Our integration "
                "deadline is this Friday.\n\n"
                "Thanks,\nCarlos Mendez\nSenior Developer, PayTech Solutions"
            ),
            "sender": "carlos.mendez@paytech.com",
            "metadata": {
                "date": "2026-04-03T09:15:00Z",
                "account_id": "ACC-99871",
                "has_attachment": False,
            },
            "ground_truth": {
                "department": "technical_support",
                "priority": "high",
            },
        },
        {
            "id": "hard-003",
            "subject": "Re: Team outing next Friday",
            "body": (
                "Hi HR Team,\n\n"
                "Thanks for organizing the team outing! Quick question though "
                "— I noticed my last paycheck was missing the overtime hours "
                "I logged in March (32 extra hours). I submitted them through "
                "the portal on March 28th.\n\n"
                "Can someone from payroll look into this? The team outing "
                "sounds fun by the way, count me in!\n\n"
                "Best,\nNina Rodriguez"
            ),
            "sender": "nina.rodriguez@company-internal.com",
            "metadata": {
                "date": "2026-04-03T10:00:00Z",
                "employee_id": "EMP-2025-412",
                "has_attachment": False,
            },
            "ground_truth": {
                "department": "human_resources",
                "priority": "high",
            },
        },
        {
            "id": "hard-004",
            "subject": "Want to buy 500 licenses",
            "body": (
                "Hello,\n\n"
                "I represent a group purchasing organization and we're "
                "interested in procuring 500 Enterprise licenses. However, "
                "before we proceed, we need your SOC 2 Type II compliance "
                "report and a completed security questionnaire (attached).\n\n"
                "Also, can you confirm if your data centers are in the EU? "
                "GDPR compliance is mandatory for us.\n\n"
                "Please route this to both your sales and compliance teams.\n\n"
                "Regards,\nHelga Braun\nProcurement Director, EuroMed Group"
            ),
            "sender": "helga.braun@euromed-group.eu",
            "metadata": {
                "date": "2026-04-03T11:30:00Z",
                "has_attachment": True,
                "attachment_name": "security_questionnaire.xlsx",
            },
            "ground_truth": {
                "department": "sales",
                "priority": "high",
            },
        },
        {
            "id": "hard-005",
            "subject": "Technical support needed",
            "body": (
                "Dear Support,\n\n"
                "I need to cancel my subscription effective immediately and "
                "get a prorated refund for the remaining 18 days of my "
                "billing cycle. I already exported all my data.\n\n"
                "Account: ACC-44556\n"
                "Plan: Pro Monthly ($99.99/mo)\n"
                "Billing date: April 20th\n\n"
                "Please confirm the cancellation and refund amount.\n\n"
                "Maria Gonzalez"
            ),
            "sender": "maria.gonzalez@outlook.com",
            "metadata": {
                "date": "2026-04-03T12:45:00Z",
                "account_id": "ACC-44556",
                "has_attachment": False,
            },
            "ground_truth": {
                "department": "billing",
                "priority": "high",
            },
        },
        {
            "id": "hard-006",
            "subject": "Feedback on your product",
            "body": (
                "Hi there,\n\n"
                "I've been a customer for 3 years and overall love the "
                "product. A few suggestions:\n\n"
                "1. Dark mode would be amazing\n"
                "2. The mobile app crashes when I try to upload files > 10MB\n"
                "3. Would love Slack integration\n\n"
                "The mobile crash is actually a real problem for me — I use "
                "the app in the field daily and need to upload photos. "
                "Everything else is just nice-to-have.\n\n"
                "Thanks for a great product!\nKevin Park"
            ),
            "sender": "kevin.park@fieldwork.co",
            "metadata": {
                "date": "2026-04-03T14:00:00Z",
                "account_id": "ACC-77654",
                "has_attachment": False,
            },
            "ground_truth": {
                "department": "technical_support",
                "priority": "medium",
            },
        },
        {
            "id": "hard-007",
            "subject": "Hiring for VP of Engineering",
            "body": (
                "Hello,\n\n"
                "I'm a recruiter reaching out about a VP of Engineering "
                "opportunity at a Series C startup. The role offers $350K+ "
                "base with significant equity.\n\n"
                "I know this isn't what your support inbox is for, but I "
                "couldn't find a careers page on your site. Could you forward "
                "this to whoever handles recruitment?\n\n"
                "Best,\nJake Morrison\nTalentFirst Recruiting"
            ),
            "sender": "jake.morrison@talentfirst.com",
            "metadata": {
                "date": "2026-04-03T15:15:00Z",
                "has_attachment": True,
                "attachment_name": "job_description.pdf",
            },
            "ground_truth": {
                "department": "human_resources",
                "priority": "low",
            },
        },
        {
            "id": "hard-008",
            "subject": "Compliance audit — IMMEDIATE RESPONSE REQUIRED",
            "body": (
                "Dear Sir/Madam,\n\n"
                "This is an automated notification from the International "
                "Data Compliance Authority. Your organization must respond "
                "to audit request #AUD-2026-5543 within 48 hours or face "
                "penalties.\n\n"
                "Please click the link below to verify your compliance "
                "status: http://totally-legit-audit.com/verify\n\n"
                "Failure to respond will result in fines up to $500,000.\n\n"
                "Compliance Department\n"
                "International Data Compliance Authority"
            ),
            "sender": "audit@compliance-authority-intl.com",
            "metadata": {
                "date": "2026-04-03T16:30:00Z",
                "has_attachment": False,
                "spam_score": 0.95,
            },
            "ground_truth": {
                "department": "general_inquiry",
                "priority": "low",
            },
        },
    ]


# ──────────────────────────────────────────────────────────────
#  TASK REGISTRY
# ──────────────────────────────────────────────────────────────

TASK_REGISTRY = {
    "easy": get_easy_tasks,
    "medium": get_medium_tasks,
    "hard": get_hard_tasks,
}


def get_tasks(difficulty: str = "easy") -> List[Dict[str, Any]]:
    """
    Return email tasks for the given difficulty level.

    Args:
        difficulty: One of "easy", "medium", "hard"

    Returns:
        List of email task dictionaries
    """
    if difficulty not in TASK_REGISTRY:
        raise ValueError(
            f"Unknown difficulty '{difficulty}'. "
            f"Choose from: {list(TASK_REGISTRY.keys())}"
        )
    return TASK_REGISTRY[difficulty]()
