# Lab 24.7: Real User Monitoring

## Objective
Implement Real User Monitoring (RUM) for frontend applications.

## Learning Objectives
- Set up RUM tools
- Track user interactions
- Monitor page performance
- Analyze user experience

---

## RUM with Datadog

```html
<script>
  (function(h,o,u,n,d) {
    h=h[d]=h[d]||{q:[],onReady:function(c){h.q.push(c)}}
    d=o.createElement(u);d.async=1;d.src=n
    n=o.getElementsByTagName(u)[0];n.parentNode.insertBefore(d,n)
  })(window,document,'script','https://www.datadoghq-browser-agent.com/datadog-rum.js','DD_RUM')
  
  DD_RUM.onReady(function() {
    DD_RUM.init({
      clientToken: 'YOUR_CLIENT_TOKEN',
      applicationId: 'YOUR_APP_ID',
      site: 'datadoghq.com',
      service: 'my-web-app',
      env: 'production',
      version: '1.0.0',
      sampleRate: 100,
      trackInteractions: true,
      defaultPrivacyLevel: 'mask-user-input'
    })
  })
</script>
```

## Custom User Actions

```javascript
// Track custom action
DD_RUM.addAction('checkout_completed', {
  orderId: '12345',
  amount: 99.99,
  items: 3
})

// Track errors
DD_RUM.addError(new Error('Payment failed'), {
  userId: 'user123',
  paymentMethod: 'credit_card'
})

// Add user context
DD_RUM.setUser({
  id: 'user123',
  name: 'John Doe',
  email: 'john@example.com',
  plan: 'premium'
})
```

## Performance Monitoring

```javascript
// Monitor Core Web Vitals
DD_RUM.onReady(function() {
  // LCP, FID, CLS automatically tracked
  
  // Custom timing
  const startTime = performance.now()
  doExpensiveOperation()
  const duration = performance.now() - startTime
  
  DD_RUM.addTiming('expensive_operation', duration)
})
```

## Success Criteria
✅ RUM configured  
✅ User interactions tracked  
✅ Performance monitored  
✅ User experience analyzed  

**Time:** 40 min
