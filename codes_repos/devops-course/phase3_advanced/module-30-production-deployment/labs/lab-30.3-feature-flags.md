# Lab 30.3: Feature Flags

## Objective
Implement feature flags to decouple deployment from release.

## Learning Objectives
- Set up feature flag service
- Implement flags in code
- Gradual rollout
- A/B testing

---

## LaunchDarkly Setup

```python
import ldclient
from ldclient.config import Config

ldclient.set_config(Config("sdk-key"))
client = ldclient.get()

user = {
    "key": "user@example.com",
    "email": "user@example.com",
    "custom": {
        "groups": ["beta-testers"]
    }
}

# Check flag
if client.variation("new-checkout", user, False):
    # New checkout flow
    render_new_checkout()
else:
    # Old checkout flow
    render_old_checkout()
```

## Custom Feature Flags

```python
# Simple in-memory flags
class FeatureFlags:
    def __init__(self):
        self.flags = {
            "new-ui": {"enabled": True, "rollout": 50},  # 50% of users
            "dark-mode": {"enabled": True, "rollout": 100}
        }
    
    def is_enabled(self, flag_name, user_id):
        flag = self.flags.get(flag_name)
        if not flag or not flag["enabled"]:
            return False
        
        # Hash user_id to get consistent percentage
        hash_val = hash(user_id) % 100
        return hash_val < flag["rollout"]

flags = FeatureFlags()

# Usage
if flags.is_enabled("new-ui", user.id):
    return render_new_ui()
```

## Gradual Rollout

```python
# Day 1: 10% of users
flags.update("new-feature", rollout=10)

# Day 2: 25% of users
flags.update("new-feature", rollout=25)

# Day 3: 50% of users
flags.update("new-feature", rollout=50)

# Day 4: 100% of users
flags.update("new-feature", rollout=100)

# Remove flag from code after full rollout
```

## A/B Testing

```python
variant = client.variation("checkout-variant", user, "control")

if variant == "variant-a":
    # Show variant A
    show_one_page_checkout()
elif variant == "variant-b":
    # Show variant B
    show_multi_step_checkout()
else:
    # Control group
    show_original_checkout()

# Track conversion
analytics.track("checkout_completed", {
    "variant": variant,
    "user_id": user.id
})
```

## Kill Switch

```python
# Emergency disable feature
if client.variation("feature-kill-switch", user, False):
    # Feature is killed, use fallback
    return fallback_behavior()
```

## Success Criteria
✅ Feature flags implemented  
✅ Gradual rollout working  
✅ A/B testing configured  
✅ Kill switch tested  

**Time:** 45 min
