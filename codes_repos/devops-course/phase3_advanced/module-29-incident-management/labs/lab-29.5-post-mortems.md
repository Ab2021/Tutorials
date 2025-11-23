# Lab 29.5: Post-Mortems

## Objective
Conduct blameless post-mortems for continuous improvement.

## Learning Objectives
- Facilitate post-mortem meetings
- Document incidents
- Identify root causes
- Create action items

---

## Post-Mortem Template

```markdown
# Post-Mortem: [Incident Title]

## Incident Summary
**Date:** 2024-01-15  
**Duration:** 2 hours  
**Severity:** P1  
**Impact:** 50% of users unable to login

## Timeline
- 14:00: Deployment to production
- 14:15: First alerts fired
- 14:20: Incident declared
- 14:30: Root cause identified
- 15:00: Fix deployed
- 16:00: Incident resolved

## Root Cause
Database connection pool exhausted due to connection leak in new code.

## What Went Well
- ✅ Alerts fired within 15 minutes
- ✅ Team responded quickly
- ✅ Rollback procedure worked

## What Went Wrong
- ❌ No load testing before deployment
- ❌ Connection leak not caught in code review
- ❌ Monitoring didn't catch connection pool usage

## Action Items
1. [ ] Add connection pool monitoring (Owner: @alice, Due: 2024-01-20)
2. [ ] Implement load testing in CI/CD (Owner: @bob, Due: 2024-01-25)
3. [ ] Add code review checklist item for resource cleanup (Owner: @charlie, Due: 2024-01-18)

## Lessons Learned
- Always test under load before production deployment
- Monitor all critical resources (CPU, memory, connections)
- Code review should include resource management checks
```

## Post-Mortem Meeting Agenda

```
1. Review timeline (10 min)
2. Discuss root cause (15 min)
3. What went well (10 min)
4. What went wrong (15 min)
5. Action items (10 min)

Rules:
- No blame
- Focus on systems, not people
- Everyone participates
- Document everything
```

## Success Criteria
✅ Post-mortem document created  
✅ Root cause identified  
✅ Action items assigned  
✅ Lessons learned documented  

**Time:** 35 min
