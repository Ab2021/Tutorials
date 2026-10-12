# Low-Level Design & OOP — 50 Interview Questions with Model Answers

> **TL;DR** — This is the drill file for the machine-coding and LLD half of an interview. Every question has a model answer written the way you would say it, plus a **`Probe:`** follow-up and a **`Weak:` vs `Strong:`** contrast where it matters. The topics that generate the most signal are **SOLID as consequences rather than definitions** (because everyone can recite the letters and almost nobody can name the change that would violate one), **concurrency in the class you just drew** (because "is it thread-safe" is the question that separates people who have shipped from people who have read), and **the classic problems** (because a parking lot is not a test of parking lots — it is a test of whether you can find the extension seam). Read it once end to end, then live in the questions marked **⚡** — those are the ones that separate a senior answer from a correct one.

---

## How to use this file

| Symbol | Meaning |
|---|---|
| **Q / A** | The question and a model answer you could say out loud |
| **Probe** | The follow-up that tests depth — the interviewer's second question |
| **Weak / Strong** | Calibration contrast |
| **Level** | E = easy, M = medium, H = hard |
| **⚡** | **Senior-separating.** The question where a correct answer is table stakes and a *reasoned* answer is the signal |

**Three rules for every LLD answer.** Name the **entity** (what has identity and a lifecycle), name the **seam** (the one place this design expects to change), and name the **failure** (what happens when the thing you are drawing goes wrong). An LLD answer that is only a class diagram is a drawing; an LLD answer with a seam and a failure mode is a design.

**And one meta-rule worth more than the rest:** in LLD, the interviewer is not testing whether you know how a parking lot works. They are testing whether your design survives the **second** requirement they are about to add. Every answer below tries to get to the second requirement before they have to say it.

---

## A. OOP and SOLID (Q1–Q8)

### Q1. Explain encapsulation, inheritance, polymorphism and abstraction — with a concrete example each. — E

**A:** They are four answers to four different questions, and the way to say them is by what each one *prevents*.

**Encapsulation** asks "who is allowed to put this object into an invalid state?" The answer is "only the object". A `BankAccount.balance` that is a public field can be set to a negative number by any caller; a `balance` with a `withdraw()` that checks funds and a `deposit()` cannot. The example I would give is an `Order` whose `status` is private and only mutable through `ship()` and `cancel()`, because the invariant "you cannot ship a cancelled order" has to live somewhere and the only place it can live is the class.

**Inheritance** asks "what is genuinely a *kind of* this, sharing both interface and implementation?" The example that survives scrutiny is `PushChannel` and `EmailChannel` extending a base that already implements "call the provider, translate the response, return an outcome" — because the *implementation* is shared and the *behaviour that varies* is one method. The counter-example is the one that shows up in every interview: `Stack extends Vector` in the old Java library, which shares the implementation and gets an `insertElementAt` it should never have had. If inheritance only shares the interface, you wanted an interface.

**Polymorphism** asks "who decides which code runs?" The answer is the object, not the caller. `for channel in channels: channel.send(message)` — the sender does not know or care which provider is behind it, and that is what makes adding Slack a new class rather than an `if`. The measurable version of this: if adding a fourth channel requires editing an existing `switch`, you had conditionals where you should have had polymorphism.

**Abstraction** asks "what is the caller allowed to not know?" A `Clock` interface with `now()` lets the caller *not know* that production reads the wall clock and tests read a counter they control. The test of good abstraction is not "is it short" but "how much does the caller have to know to use it correctly" — and the honest answer is that abstraction is only earned when there is a second implementation, which is why I would not introduce an interface for a single caller with no foreseeable second one.

### Q2. Single Responsibility — how do you know when a class has more than one? — M

**A:** Not by counting methods. The reliable test is **"who asks to change it, and why?"** — because that is the definition: one class, one reason to change, one *audience*.

A `Notification` class that formats the message body, sends it, and stores the delivery status has three audiences: the product team that changes copy, the infrastructure team that changes providers, and the reporting team that changes what gets recorded. Three audiences means three change streams, which means three reasons for the file to conflict in review, which means the class is doing three jobs.

The concrete smell I look for in a machine-coding round: a class whose methods divide cleanly into groups that share no fields. If `send()` uses `provider` and `token` while `format()` uses `template` and `locale` and neither touches the other's fields, that is two classes in one body. The second smell is a method name with an "and" in it, and the third is a test that has to set up five unrelated things to assert one.

**Probe: does SRP mean one class per file, or small classes?** Neither. It means one *axis of change*. A `NotificationStateStore` with ten methods is one responsibility if every one of them changes for the same reason — the lifecycle of a notification. Ten tiny classes that all change together is the same responsibility scattered, which is worse.

### Q3. Open/Closed — what is a real example, and what is a fake one? — M

**A:** Open for extension, closed for modification: you add behaviour by adding code, not by editing code that already works.

The **real** example is a `ClassificationTable` mapping provider error codes to `RETRYABLE | TERMINAL | SUPPRESS_TOKEN`. A new provider's new error code is a new *row*, and the code that classifies — one `dict.get` with a default — does not change. The second real example is a `Channel` interface: adding Slack is a new class and a registration line, and `Sender.send` is untouched.

The **fake** example — and this is the one worth saying out loud — is a `Shape` hierarchy with `area()` on each subclass, presented as the canonical OCP demonstration. It is closed only until somebody needs the perimeter, or needs a shape from another module, or needs to serialize. The real question is always **what is the axis of variation and did you actually keep it in one place?** In the notification case the axis is "a provider invents a new error code", which is genuinely unbounded and genuinely data. In many systems the axis you thought you had turns out to be closed after all, and an interface for three fixed cases is a cost with no benefit.

**Strong:** "OCP is not 'always use an interface'. It is 'the thing that varies should be data or a new implementation, and the thing that does not vary should not be edited'. If I cannot name two real future implementations, I write the concrete class and I say why."

### Q4. Liskov Substitution — give an example that actually breaks it. — H ⚡

**A:** The rule is that a subclass must be usable anywhere the base class is, without the caller knowing. What actually breaks it is almost always a **strengthened precondition** or a **weakened postcondition**, and the fact that it usually shows up as a runtime surprise rather than a compile error is why it matters.

My favourite example is a `Rectangle` with independent `setWidth`/`setHeight` and a `Square` that sets both. `area()` is fine; the caller that does `r.setWidth(5); r.setHeight(4); assert r.area() == 20` is broken, and it is broken by an invariant the *subclass* added. The subclass is not substitutable.

The example I would actually bring to an interview is closer to home: a `Channel` interface whose `send` is documented as "returns an outcome, never throws", and a `SlackChannel` that throws on a 429 because the author found exceptions easier. Every caller that trusted the contract is now broken at runtime. That is the LSP violation people ship — not a wrong shape, a **strengthened guarantee**.

The practical defence: put the contract in the interface as *behaviour*, not as a signature. "Never throws, never retries internally, always classifies the failure" is the contract; a `SendOutcome` return type is only half of it. And in a language with no way to enforce that, the honest answer is that the interface docstring is load-bearing and a conformance test per implementation is what keeps it true.

### Q5. Interface Segregation — a fat interface is obvious in theory. Show me one in practice. — M

**A:** In practice a fat interface is almost always produced by **one implementation that needed more**, and the tell is a client forced to implement a method it will never use.

The classic: a `Worker` interface with `work()` and `eat()`, where a `RobotWorker` implements `eat()` by throwing. The real-world version I would use is a `NotificationChannel` interface with `send`, `schedule`, `cancel` and `getStatus` where a webhook channel can only send. Every implementer now has three methods whose bodies are `raise NotImplementedError`, which is an interface documenting what it is *not*.

The fix is to split by client: `Sender` with `send`, and a separate `Schedulable` with `schedule`/`cancel`, so the webhook channel implements only what it does. The measurable test: **how many implementations have at least one method that throws or returns a constant?** Each one is a client paying for a method it does not want.

**Probe: what about `NotImplementedError` on a genuinely abstract method?** Different thing. That is an abstract base saying "there is no sensible default and you must supply one" — the interface is *demanding* the method. The bad case is a *concrete* base whose subclass overrides it to escape an obligation the base imposed.

### Q6. Dependency Inversion — what does the inversion actually invert? — M

**A:** The direction of the source dependency. Without it, a high-level policy class imports a low-level detail: `Sender` importing `FirebaseClient` means the send policy cannot be tested without Firebase, and cannot be reused with a different provider without editing. With it, both depend on an abstraction — `Sender` takes a `provider` callable, and Firebase is one implementation of it.

The **inversion** is real and worth stating precisely: the naive dependency is `policy → detail`. The inverted one is `policy → interface ← detail`, so the detail now depends on something the policy owns. The policy defines what it needs; the detail conforms.

The practical version of this in a machine-coding round is small: a `Sender` that takes `provider: Callable[[Message], Outcome]` in its constructor rather than constructing a client internally. That one choice is the difference between a test suite that runs in milliseconds with scripted errors — including the 429 followed by a hard bounce followed by a 410 that nobody reproduces against a real provider on purpose — and a test suite that needs credentials and network access.

**Strong:** "Dependency injection is the mechanism; DIP is the reason. Injecting a concrete class you also import is not inversion, it is just a parameter."

### Q7. Composition over inheritance — when is inheritance actually right? — M

**A:** Inheritance is right when three things are true at once: the subtype genuinely **is-a** the supertype, the supertype's **implementation** is what you want to reuse (not just its signature), and the supertype was **designed** for subclassing.

The notification case satisfies all three: `PushChannel` and `EmailChannel` are kinds of channel, the base's `send` implementation (call provider, translate, classify, return) is exactly right, and it has one overridable method by design. `SmsChannel` overrides that method to add an allow-list gate and calls `super()` — which is the pattern working as intended.

Composition is right when what you want is *behaviour*, not *identity*. A notification that can be retried, logged and rate-limited is not three kinds of notification; it is one notification with three collaborators. Deep hierarchies are the symptom: if you have `RetryingLoggedNotification extends LoggedNotification extends Notification`, you have chosen inheritance to add orthogonal features and you will need a class per combination — four features, sixteen classes. The test is **"does the subclass change what the object IS, or what it DOES?"** What it does is a collaborator.

### Q8. ⚡ Where does SOLID actually help, and where does it make designs worse? — H ⚡

**A:** SOLID helps in exactly one situation: **when there is a second thing coming, and it was not there when you wrote the first.** Every letter is a way of arranging code so that the second thing is cheap: SRP so a change has one home, OCP so a new case is a new row or class, LSP so a caller can trust an abstraction, ISP so a client is not forced to implement the union of everyone's needs, DIP so policy can be tested and reused without its details.

Where it makes designs worse is when there is **no second thing** and you build for it anyway. The cost is concrete: an interface with one implementation and no test double is a file that exists to be navigated past; a factory that constructs one product is indirection with no payoff; a `Strategy` for a rule that will never vary is a class the reader has to open to learn nothing. And the damage is worse than clutter — every layer of indirection is a place a behaviour can hide, which is precisely the failure in the notification design where a "pre-send hook" swallows a `SUPPRESS_TOKEN` outcome and the dead token never gets deleted.

So the answer I would give is that SOLID is a **response to anticipated change, and the anticipation has to be named**. If I can say "the provider will change", DIP is justified. If I can say "there will be a fourth channel", the interface is justified. If I cannot name the second thing, the concrete class is the better design, and I would rather say that out loud than produce an interface inventory the interviewer can count.

---

## B. Design patterns (Q9–Q16)

### Q9. Which patterns do you actually reach for, and which do you avoid? — M

**A:** The ones I reach for unprompted, because they solve problems I have actually had:

- **Strategy** — when a rule varies and the caller should not know which one runs. Notification channels, quiet-hours policies, analytics sinks.
- **Value Object** — a frozen, hashable, comparable type for anything that is *described by its data*: money, an email address, a `SendOutcome`, a `Cursor`. This is the highest-value pattern per line of code in the whole catalogue, and the least discussed.
- **Entity** — the small number of things with identity and a lifecycle. In most systems there are far fewer of these than people think.
- **Repository** — when there is more than one place data lives, or the query needs to be swappable, or the locking needs to be hidden from callers.
- **State** — when the legal transitions are the design. A notification's lifecycle, an elevator's motion, an order's status.
- **Ports & Adapters** — for every external system. It is the pattern that makes tests fast, and its cost is one constructor parameter.

The ones I avoid by default: **Singleton** (it makes test isolation impossible and hides a dependency that should have been a parameter), **Observer** where the notification is on a critical path (a subscriber that throws must not be able to fail the publisher), and **Builder** where exactly one class constructs the object (the Builder becomes a way to skip the invariants that class enforces).

**The single best answer to a "which patterns" question** is to name the *problem* first and the pattern second, because the pattern is only defensible as the answer to a specific problem. "I used Strategy here because there are three channels and the sender must not know which one it is talking to" is an answer. "I used the Strategy pattern" is a vocabulary test.

### Q10. Strategy versus State — they look identical in a diagram. — M

**A:** They have the same shape — a context delegating to an interchangeable object — and they differ in **who decides when it changes**. That is the whole distinction and it is worth saying in exactly those words.

In **Strategy**, the caller picks, and it does not change while the context is in use. A `Sender` is constructed with a `ClassificationTable` and keeps it. In **State**, the object itself moves between states, and the transitions are part of the domain. A notification in `PENDING` becomes `ACCEPTED` becomes `DELIVERED`, and *which* transitions are legal is the design question. In State, each state often knows what can come next; in Strategy, the strategies do not know about each other at all.

The test: **"is the change driven by a transition inside the object, or by a choice from outside?"** Inside is State. Outside is Strategy. The notification design has both, deliberately: `ClassificationTable` is Strategy, `NotificationStateStore` is State, and separating them is what keeps the retry *policy* from being entangled with the lifecycle *mechanics*.

### Q11. Adapter versus Decorator versus Facade. — M

**A:** All three wrap, and they answer three different questions.

**Adapter** answers "the interface I have does not match the interface I need". A provider SDK exposes `sendPush(token, payload, opts)` and my `Channel` needs `send(message) -> SendOutcome`. The adapter translates — including the return *type*, which is the part people forget: the SDK throws on failure and my contract says "never throws, always classifies", so the adapter is where the exception becomes a `FailureKind`.

**Decorator** answers "I need to add behaviour to an existing object without changing it, and I need to be able to stack the additions". A `RetryingChannel` wrapping a `PushChannel`, or a `MetricsChannel` wrapping either, with the same interface all the way down. The tell that it is a Decorator rather than an Adapter is that the interface is *unchanged* — it is still a `Channel`, and the caller cannot tell it is wrapped.

**Facade** answers "this subsystem is complicated and callers should not have to know". A `NotificationApi` with two methods over a decision service, an idempotency store and a state store. The tell is that a Facade is allowed to *reduce* the interface and it does not implement the same interface as what it wraps.

**The one I would warn against:** Decorator on a critical path where the decoration can fail silently. A `RetryingChannel` that retries internally makes the attempt count on the state row a lie, which is why the notification design says channels never retry and the retry lives one level up where it is observable.

### Q12. Why is Singleton so tempting, and what do you do instead? — M

**A:** It is tempting because it solves a real problem badly. The real problem is "there should be exactly one of these and everyone needs it" — a config, a connection pool, a clock. The bad solution is a static instance, and the reasons are concrete:

1. **Tests cannot isolate.** Two test cases share the singleton's state, so order matters and a test that passes alone fails in a suite. That is a debugging cost measured in hours.
2. **The dependency is hidden.** `Config.get()` inside a method is invisible from the signature. A constructor parameter is a documented requirement; a static call is a surprise a reader only finds by grepping.
3. **It cannot be replaced.** A test needs a fake clock; a migration needs two stores at once; a canary needs different config per instance. A singleton makes all three a refactor.

What I do instead: make it an **instance and inject it**. The "exactly one" property then belongs to the composition root — the process builds one and passes it — which is where "exactly one" is actually true and where it can be changed. If the language makes that painful, a DI container is a singleton holder, and the difference is that the *class* is not the singleton, so tests can still make their own.

**The line I would use:** "A singleton is a lifetime decision expressed as a class-level fact. Lifetimes belong to the wiring, not to the type."

### Q13. ⚡ When is a pattern the wrong answer? — H ⚡

**A:** When the pattern is doing the *naming* instead of the *work*. The three cases I have seen most:

**One implementation, no test double, no named second.** An interface with a single implementer and no fake is a file that exists to be navigated. The cost is real: reading the code now requires opening two files to understand one behaviour, and every future reader has to check whether the second implementation is elsewhere.

**The pattern hides the ordering.** This is the subtle one and it is the best example I have. The notification decision engine runs eight checks — consent, preference, destination, priority, quiet hours, dedupe, rate limit, template — and the **order is the design**: a dropped notification must not consume rate-limit budget, so quiet hours comes before the rate limit; a duplicate must not consume it either, so dedupe comes before it too. Splitting those checks into a `PreferenceChecker`, a `QuietHoursChecker` and a `RateLimitChecker` reads better in a diagram and makes the ordering a property of nothing. The pattern would have destroyed the thing the code exists to get right.

**The pattern is a policy that hides a decision.** Failover as a Decorator is my favourite example: a `FailoverChannel` that transparently tries a second provider is where "failover" quietly becomes double-sending, because the caller's retry and the decorator's failover are now independent. Anything that changes *what happened* should be visible in the outcome type, not absorbed by a wrapper.

**Strong:** "A pattern earns its place by making a change cheaper. If I cannot say which change, the pattern is decoration — and an interview is a good place to say that rather than to demonstrate range."

### Q14. Explain dependency injection without using the word "framework". — E

**A:** Dependency injection is: **a class does not create what it needs; it is given it.** `Sender(...)` takes a provider, a clock, a template store and a limiter rather than constructing them, and `DecisionService` takes the preference store, the dedupe store and the timezone rather than reaching for globals.

That is the whole idea, and the payoff is three things: the class can be tested with fakes (a clock I control, a provider that returns exactly the errors I want to test), the class can be reused with different collaborators (a different provider per environment), and the class's requirements are **visible in its signature** — reading the constructor tells you what it depends on, which a static call never does.

**Probe: is constructor injection always right?** No. Method injection when the dependency varies per call — a `Cursor` passed to a paginate method is a parameter, not a dependency. And for a genuinely global, genuinely constant thing (a country's tax table that ships with the binary), direct construction is fine and injection is ceremony.

### Q15. Name a pattern you would use for an operation that must happen exactly once. — M

**A:** The pattern is not a GoF pattern — it is a **claim**, and the naming matters because the naive version is a check. A `is_processed(id)` followed by a `mark_processed(id)` has a window between the two calls in which a second thread sees "not processed" and proceeds. The correct shape is one atomic operation that returns **ownership**:

```
fresh, existing = store.claim(key, owner)
if not fresh:
    return existing          # somebody else owns it, and here is their result
... do the work ...
```

This appears three times in the notification design in three different costumes, which is a good thing to point out: an `IdempotencyStore` claiming a producer's key, a `DedupeStore` claiming an event's key, and a `TokenBucket` taking tokens inside the lock it measured them in. Same shape, three problems, and in every case the API is built so the unsafe version cannot be written.

The second half of the answer is where to put the claim: **record the owner on the claim, not after the work.** Then a crash mid-operation leaves the key held and the retry gets the original owner's id back rather than starting a second operation. For an OTP that is the correct direction to fail; for a payment it is the difference between one charge and two.

### Q16. What is the difference between a value object and an entity, and why does it matter? — M

**A:** An **entity** has identity that persists through change: a `Notification` is the same notification after it moves from `PENDING` to `DELIVERED`, so it is compared by id and it is mutable. A **value object** has no identity at all — it *is* its data: a `SendOutcome(accepted=True, provider_message_id="pm_1")` is interchangeable with any other outcome holding the same values, so it is immutable and compared by value.

Why it matters, concretely, three ways:

**Mutation safety.** A frozen value can be passed across threads, put in a set, used as a dict key and shared between two callers without any possibility of one of them changing what the other sees. A mutable one cannot. `SendOutcome` being frozen is why the Sender can return it up through several frames and nobody has to wonder who owns it.

**Where behaviour goes.** Because a value object is immutable, any method on it must return a *new* value, which is exactly the right signature for things like `Money.add(other)`. And because an entity is mutable, the methods that change it should be the ones that enforce its invariants — `notification.transition_to(...)` rather than a public `state` field.

**The bug it prevents.** The single most common one: a mutable object shared between two components where one of them "just tweaks a field". Making the small, described-by-data types frozen removes an entire class of bug for free — which is why it is the pattern I reach for first and the one I would mention first in an interview, precisely because it is not in the GoF book and most candidates do not name it.

---

## C. Concurrency and thread-safety (Q17–Q23)

### Q17. A class you just designed is used by ten threads. Where do the races live? — M

**A:** I would answer it as a checklist over the object graph rather than by staring at the code, because the races are always in the same four places.

**Shared mutable state.** Not "is there a lock" but "which fields can two threads write". In the notification design the answer is: the stores (guarded), the `Notification` rows (guarded by the store, mutated only through `transition`) and the diagnostic counters (guarded, because `+=` on an attribute is a load, an add and a store).

**Check-then-act.** The race that actually ships. `if not seen: mark()` is two operations; the fix is one atomic `claim()` that returns ownership. This is the single most valuable answer to this question and it generalises: dedupe, idempotency, rate limiting and "is this seat taken" are all the same race with different nouns.

**Read-modify-write across a lock boundary.** A `TokenBucket` that reads `tokens`, computes a new value, and writes it in two separate critical sections will over-issue under concurrency. Counting inside the lock you measured in is the fix, which is why `check_and_count` is one method and not two.

**Anything held across I/O.** A lock held across a network call converts a slow provider into a stalled process. The rule I would state: no lock is held across a provider call, which is why the limiter returns a *wait time* and the caller schedules rather than blocking.

### Q18. ⚡ Your `due()` method selects PENDING rows and two workers run it. What happens? — H ⚡

**A:** **Both workers select the same row and both send it.** This is the honest answer and giving it is worth more than giving a confident wrong one.

And note why the obvious defence does not cover it: the dedupe store already claimed this notification at *decision* time, before the row existed, so a second dispatcher's dedupe check would either pass (different process state) or refuse a notification that legitimately needs sending. The dedupe half is about the *event*; this is about the *work item*.

The fix is a **lease**, and it is small: `due()` takes a visibility timeout and, in the same critical section that selects the rows, marks them claimed with `claimed_until = now + timeout`. A row whose claim expires is reclaimable, which is what makes the worker crash recoverable. That is exactly what a queue does with a visibility timeout, and the reason to implement it in the store rather than to reach for a queue is that the row is the source of truth.

**Strong:** "I would say in the design review that this system has the dedupe half and not the lease half, because naming it is cheaper than being asked. The change is two columns and a timeout, not a redesign."

### Q19. How do you make a class thread-safe without a lock? — H ⚡

**A:** Four ways, in the order I would try them.

**Don't share it.** The best fix. If the state is per-request, it never needs a lock. `RenderedMessage`, `SendOutcome` and the `Cursor` in the feed design are values that get created and discarded; sharing them is safe because they are frozen, not because anything guards them.

**Make it immutable.** A frozen dataclass is thread-safe by construction, no lock, no reasoning required. This is why the value-object pattern earns its keep beyond tidiness.

**Make the operation atomic rather than the object.** A `Counter` implemented over an atomic increment needs no lock at the API level. A `claim()` built on a compare-and-set is atomic without a lock the caller can see. This is the right answer when the state is one number or one flag.

**Confine it.** One owner per thread: partition by user id and let exactly one consumer hold each partition. That is how ordering is bought too, so it is a single mechanism solving two problems — at the cost of throughput, and the cost is why I would not default to it.

**And the honest fifth:** sometimes you *do* need the lock, and then the goal is to make the critical section as small as possible and to never hold it across I/O. Saying that is better than pretending a lock-free design always exists.

### Q20. What is the difference between a `Lock` and an `RLock`, and when does the wrong choice bite? — M

**A:** An `RLock` is reentrant: the same thread can acquire it again without deadlocking. A plain `Lock` cannot be re-acquired by a thread that already holds it, so the second acquisition blocks forever — and the *same thread* blocking on itself is a deadlock with no other thread involved.

When it bites is subtle, and this is the part worth saying: it bites when a **composed read method calls another read method on the same object**. In the notification state store, `progress()` and `counts()` are both built from `for_request()`, which takes the lock. With a plain `Lock`, both would deadlock — and only under concurrency, and only in the code path somebody added later, which is the kind of bug that ships and then appears once a week in production.

The reason it is worth mentioning is that it demonstrates you have thought about **method composition**, not just about individual methods. My rule: if any method on the class calls any other method on the class, use an `RLock` and say why in the docstring, because the next person will "simplify" it otherwise.

### Q21. Is `i += 1` thread-safe in Python? — M

**A:** No. It is three bytecode operations — load, add, store — and a thread switch between the load and the store loses an increment. The GIL makes individual bytecodes atomic; it does not make a *sequence* of bytecodes atomic, and this is the misconception that makes people skip the lock.

It matters where the counter is load-bearing rather than cosmetic. A lost increment on a request counter is a metric that is slightly wrong; a lost increment on a webhook handler's `rejected` counter is a number an incident review reads and draws a conclusion from, which is why that counter in the notification design is a `Counter` class with a lock behind a one-line `bump()`. In production it is an atomic counter in the metrics library with the same interface, and the point is that the interface is the same either way.

**Probe: what about `list.append`?** Atomic in CPython — the list's internal operation is a single bytecode — but I would not design against that. It is an implementation detail of one interpreter, it does not extend to "check the length then append", and relying on it is exactly the kind of knowledge that stops being true on a different runtime.

### Q22. How do you test that something is thread-safe? — H ⚡

**A:** I would say plainly that **you cannot test it, you can only increase confidence**, and that the structural argument is the primary evidence.

The structural argument is: check-and-effect pairs are single operations (a `claim` that returns ownership, a `check_and_count` that counts inside the lock it checked in, a `transition` that decides legality inside the lock it mutates under), and no lock is held across a network call. Those are properties you can *read* and review, and they are worth more than a green test.

The empirical evidence is a stress harness, and it has to be built so a failure is detectable:
- **The claim race.** N threads call `claim()` on the same key simultaneously; assert exactly one got `fresh=True` and the other N-1 got the same owner back. Run it in a loop, because a race that happens once in ten thousand runs passes a single-shot test.
- **The state race.** One thread marking `ACCEPTED` and one consumer marking `DELIVERED`, repeatedly, on a fresh row; assert the state never walks backwards and the final state is always `DELIVERED`.
- **A deliberate break.** The most valuable version: temporarily remove the atomicity, confirm the test *fails*, and put it back. A concurrency test that has never been seen to fail is a test you do not know the shape of.

**Strong:** "In practice I write the stress test for the two operations where a lost race causes a user-visible bug — the duplicate send and the double charge — and I rely on the structural argument everywhere else, because a thread-safety test suite that runs for an hour and passes is still not a proof."

### Q23. ⚡ What ordering guarantees can you give, and what do they cost? — H ⚡

**A:** The honest default is **none between independent notifications**, and the useful answer says that, says why, and says what it would cost to change.

Why none: each notification is an independent row, workers are pooled across rows, and the provider reserves the right to reorder. Nothing in the design makes `order.shipped` arrive after `order.confirmed`, and a design that *claims* it does without partitioning is a design that lies. The promise I would state instead is "at-least-once with deduplication", which is a promise the system can keep.

What ordering would cost: partition by user id and consume each partition in exactly one thread. That gives per-user ordering — and it costs throughput (a hot user serialises, and the pool is only as parallel as the partition count), it costs a rebalance protocol so a partition moves without losing its in-flight work, and it introduces a new failure mode where one stuck user blocks their whole partition. For "confirmations before shipping notifications" that is usually not worth it; for a per-user sequence number on the *payload*, where the client sorts, it usually is.

**The framing I would use:** "Ordering is a per-key property, so the question is always *ordered by what?* — and the answer determines the cost. Global ordering is a fiction; per-user ordering is a partition key; per-entity ordering is usually a sequence number."

---

## D. Class modelling and UML (Q24–Q29)

### Q24. How do you find the entities in a problem you have never seen? — M

**A:** I read the requirements and pull out **nouns that have a lifecycle**, then throw the rest away. The test for an entity is not "is it a noun" but "does it have states, identity, and a reason to be tracked over time?"

Concretely, on a parking lot: "vehicle" is a noun, but a vehicle in a parking lot system usually has no lifecycle of its own — it is a **value** (a plate, a size) attached to something else. "Ticket" has identity, a state (open, paid, closed), and it is what everything else is queried by. "Spot" has a *type* and an occupancy state. So the entities are Ticket and Spot, and Vehicle is a value object, and that distinction is exactly the kind of thing that separates a model from a noun list.

The three questions I ask about every candidate noun: **does it have an id? must it survive a restart? does it change state in a way somebody cares about?** Three yeses is an entity. No id and described entirely by its data is a value object. Neither is probably an operation that belongs on something else.

**Probe: what about "the lot" itself?** An aggregate, not an entity — it is the boundary that makes the other invariants enforceable ("this ticket's spot belongs to this lot"). Naming it as an entity and then mutating spots through it is how a model acquires a god object.

### Q25. What does the class diagram need to show, and what should it leave out? — M

**A:** It needs three things: **the entities and their relationships** (including cardinality), **the interfaces that are the extension seams**, and **the direction of the dependencies**. Anything else is decoration that costs you drawing time.

What to leave out: getters and setters, every field type, constructors, and — this is the one people get wrong — most of the inheritance. If the diagram has a deep hierarchy, the reader learns less, not more, because they cannot see the collaborators. The notification diagram in the bridge puts a large box in the middle labelled `DecisionService.decide` with its **eight numbered steps** listed inside it, because that box is the design and its internals are the specification. That is a diagram doing work.

**Two notation points worth being precise about**, because getting them wrong reads as not knowing UML:

- A **filled diamond** is composition — the part cannot exist without the whole, and the whole owns its lifecycle. A **hollow diamond** is aggregation — the part exists independently. An open arrow is association; a **hollow triangle** is inheritance ("is-a"); a **dashed line with a hollow triangle** is realisation ("implements").
- **Multiplicity goes on the line, at the end whose count you are stating.** `Lot 1 ── * Spot` means one lot has many spots. Getting the ends backwards is the single most common UML error and it is worth slowing down for.

**Strong:** "I draw the diagram to answer one question — *where does the next requirement attach?* If the seam is not visible on the page, the diagram has not done its job and I would redraw it rather than add more boxes."

### Q26. ⚡ Abstract class or interface — and when does it actually matter? — H ⚡

**A:** An interface is a **contract with no implementation**; an abstract class is a **partial implementation with a contract**. The mechanical difference in most languages is that you can implement several interfaces and extend one class, and that an abstract class can hold state.

When it actually matters — and this is the answer, because the textbook version is easy:

- **Use an interface when the implementers share nothing.** `QuietHoursPolicy` has a `decide` method and no implementation to share, because a jurisdiction's rules are entirely their own. There is nothing useful to put in a base class.
- **Use an abstract class when the implementers share a spine and vary in one place.** `_ProviderChannel` holds the `provider` callable, the `ClassificationTable` and the call log, and implements `send` while `kind` and sometimes an allow-list gate vary. That is a Template Method, and the shared state is why it is a class and not an interface.
- **The pragmatic tie-breaker in an interview:** interfaces are cheaper to fake, an abstract class with state means every subclass's test has to satisfy the state. Given the choice for a *seam*, I default to an interface and pay for a little duplication; I use an abstract class only when the duplication would be substantial or the invariant has to be enforced once.

**And the honest caveat:** in Python, `ABC` with an abstract method is a *convention* — nothing stops a caller from ignoring it. The enforcement is the docstring plus a conformance test per implementation. I would say that rather than claim the language enforces it.

### Q27. Where do enums go, and why not strings? — M

**A:** Enums for **closed sets that code branches on**. `NotificationState`, `Priority`, `ChannelKind`, `FailureKind` — the set is fixed, every member means something to the logic, and a typo in a string is a silent bug that reaches production.

The concrete argument for enums, and the one worth making: a string state means `if state == "delivered"` can be misspelled as `"deliverd"` and the branch silently never fires. An enum member that does not exist fails at import. That is the difference between a bug found in a second and a bug found by a customer.

**Where strings are right:** open sets that arrive from outside — a provider's error code is a string, because the provider invents new ones and I do not control the vocabulary. That is exactly why the notification design has `ClassificationTable` mapping *strings* to a `FailureKind` *enum*: the enum is the closed set my code branches on, the string is the open set that arrives from the world, and the table is the boundary between them. That is a good thing to be able to say, because it shows the distinction is a decision rather than a habit.

**Probe: what about `NotificationState` having a rank?** A rank is what turns a set of states into a machine you can reason about. `{PENDING: 0, ACCEPTED: 1, ...}` is what makes "an out-of-order webhook cannot walk this backwards" a one-line check rather than a table of pairs. A closed set with an ordering is worth more than either alone.

### Q28. ⚡ Draw the classes for a system where the requirements are still moving. — H ⚡

**A:** When the requirements are moving, the diagram should be **deliberately shallow and deliberately centred on one interface**, and I would say that out loud as the design intent rather than drawing a full model and hoping.

The method: pick the **one sentence in the requirements that is most likely to have an "and also" attached**, and make that a seam. In a notification system, the sentence is "we send push and email" — the "and also" is Slack, or WhatsApp, or a new provider. So `Channel` is an interface from the first minute. In a parking lot, the sentence is "we charge by the hour" — the "and also" is a flat daily rate, or validation, or free for the first thirty minutes. So pricing is a `PricingPolicy` seam.

Everything else I would write concretely, because **guessing wrong about a seam costs more than adding one later**. A wrongly-guessed interface is a layer every future reader has to see through, and — as with the split-checks example in Q13 — it can hide an ordering the design depends on. Adding a seam when the second implementation actually arrives is a small refactor; removing a speculative one is a larger one, because callers have grown to depend on it.

**Strong:** "I would rather present a design with one obvious seam, name the requirement that might break it, and say where I would cut if it did — than present a fully generalised model with six interfaces and no explanation of which one earns its keep."

### Q29. How do you model a relationship that is many-to-many and changing? — M

**A:** With an explicit **association object**, not with two collections pointing at each other. If users follow users, `User.following: set[User]` and `User.followers: set[User]` are two places that must be updated together, and nothing enforces it — so they drift, and one of them is a follower list that is wrong.

The fix is a `FollowGraph` (or a `Follow` row) that owns the relation and is the only thing allowed to change it, with `follow(a, b)` maintaining both directions in one operation. That gives you: one place to add a timestamp or a state ("requested" vs "accepted"), one place to lock, and — in the feed bridge — one place to answer the two questions that actually matter, `followers_of(author)` and `following_of(viewer)`, with the read path that each needs.

The scaling consequence is worth stating because it is the interview's real question: at 100 M users the relation does not live in one object graph at all. It is a table, the in-memory class is a **repository over it**, and the association object is the seam between the model and whatever actually stores it. Getting that boundary right early is why `followers_of` is a method on a graph rather than a property on a user.

---

## E. The classic problems (Q30–Q44)

### Q30. Parking lot — what is the core modelling decision? — M

**A:** Not the vehicle types. **The core decision is that the Ticket is the entity and the Spot is a resource with a state, and they are linked by an assignment.** Everything hard about a parking lot follows from that pairing: a spot can be `FREE`, `OCCUPIED` or `RESERVED`; a ticket has a state and a price; and the interesting operation is the atomic one — *given a vehicle size, find a free spot and claim it* — which is a check-then-act race and must be a single claim, not a search followed by a mark.

If the design instead hangs occupancy off the vehicle (`vehicle.spot = s3`) then the spot's availability must be derived by scanning vehicles, which is both slow and racy. The state belongs on the resource being contended for.

**Probe: how many spot types?** Ask. It is almost always presented as "motorcycle, car, bus", and the useful question is whether a bus occupies one spot of a bigger type or three adjacent standard spots — the second answer changes the allocation algorithm from a lookup to an interval search, and it is a real requirement difference, not a detail.

### Q31. Parking lot — where is the extension seam for pricing and allocation? — M

**A:** Two seams, and naming both is the answer.

**`PricingPolicy`** takes a ticket (and a clock) and returns an amount. The seam exists because pricing is the requirement most likely to acquire an "and also": an hourly rate, a flat daily maximum, free for the first thirty minutes, a validated rate for a shop's customers, a lost-ticket penalty. Every one of those is a new implementation and none of them touches the ticket or the spots. And the clock is a constructor parameter rather than a `datetime.now()` call, because otherwise none of it is testable.

**`SpotAllocator`** takes a request (size, and any accessibility requirement) and a lot and returns a claim or a refusal. The seam exists because "find the nearest free spot" becomes "find the nearest free spot of a compatible type", then "keep buses together", then "reserve the ground floor for accessibility". Each is a different strategy over the same `SpotStore`.

**Strong:** "The seam I would *not* create is a `Vehicle` hierarchy. A motorcycle and a bus differ in one attribute — the size they need — and a class hierarchy for that is three classes and a `switch` in disguise. Size is a value."

### Q32. LRU versus LFU — implement and compare. — M

**A:** **LRU** evicts the least recently used, which needs a hash map plus a doubly-linked list so that both `get` and `put` are O(1): the map finds the node, the list gives you O(1) move-to-front and O(1) removal of the tail. **LFU** evicts the least frequently used, which needs a frequency counter per key plus, for O(1), a map from frequency to an ordered list of keys at that frequency, and a pointer to the current minimum frequency.

The comparison that matters: **LRU adapts to a changing working set and LFU does not.** LFU's counter is cumulative, so a key that was hot last week keeps its count and a genuinely hot new key can be evicted by a cold old one. The standard fix is LFU with ageing — halve all counts periodically, or use a sliding window of counts — which is what a real cache does, and saying that is worth more than reciting two data structures.

**And the practical point:** LRU is the right default because its failure is graceful (a slightly suboptimal eviction) and its state is small. LFU's O(1) version has a lot of bookkeeping, and the bookkeeping is where the concurrency bugs live.

### Q33. ⚡ LRU — make it thread-safe without making it slow. — H ⚡

**A:** The O(1) LRU is a hash map plus a doubly-linked list, and the concurrency problem is that `get` is *not* a read — it moves the node to the front, so every read is a write and a plain read/write lock does not help.

Three approaches, and I would present them in this order:

1. **One lock, small critical section.** Correct and simple. Both `get` and `put` take a lock for the pointer surgery, and the surgery is a handful of assignments with no I/O and no allocation inside. For most caches this is entirely adequate and I would ship it first.
2. **Stripe the cache.** N independent LRUs, shard by `hash(key) % N`, one lock each. Contention drops by roughly N and each stripe is a correct LRU. The cost is that the *global* eviction order is now approximate — a stripe can evict a hot key while another stripe holds a cold one — and that approximation is usually fine, which is the trade to state.
3. **Approximate LRU without a list.** A ring of counters per key (like a clock/second-chance cache) sampled on eviction. No list surgery, no lock on read, and the eviction is approximate by construction. This is what most production caches actually do at scale, and knowing that is a strong signal.

**The failure I would name:** the naive fix is a `Lock` around the whole cache, and the second naive fix is a read/write lock. Both are wrong for LRU, and the read/write lock is wrong for a *specific* reason — a `get` mutates, so the "read" path serialises anyway and you have added overhead and a false sense of concurrency.

### Q34. Hotel management — how do you model availability? — M

**A:** **Not with a per-room `is_booked` flag**, which is the answer that fails the moment somebody books for next month. Availability is a property of a **(room, date range)**, so the model is a `Reservation` with a check-in and check-out, and the availability question is "does any reservation for this room overlap this range?"

The two things that make it a real design: **a room type is not a room.** The customer books a *type* (a king room), and the specific room is assigned at check-in — which means availability is a count per type per night, and the assignment is a separate, later decision. That single distinction is the core of the problem, and candidate designs that reserve a specific room at booking time cannot handle "we'll give you a comparable room".

**The second thing is the half-open interval.** Checkout on the 5th and check-in on the 5th do not conflict, so the overlap test is `arrive < other.depart and other.arrive < depart` — and the number of candidates who write `<=` on both sides is a large fraction. It is worth saying explicitly because it is a concrete correctness bug in a five-line method.

**Probe: how do you price it?** Per night, not per day, and the room-type-plus-rate-plan is the seam: a `RatePlan` per type per date range, so "3 nights, member rate" is a lookup rather than arithmetic.

### Q35. ⚡ BookMyShow — how do you stop two people booking the same seat? — H ⚡

**A:** **An atomic claim on the seat row, not an application-level check.** The naive design is `if seat.available: seat.book()` — two operations with a window between them, and under load that window is where the double-booking happens. Every candidate knows this in the abstract; the answer is in the mechanism.

The mechanism I would present: a conditional update that is atomic in the database — `UPDATE seats SET status='HELD', hold_id=? WHERE id=? AND status='AVAILABLE'` — and the **row count of the update is the answer**. One row updated means I own it; zero means somebody else does. No lock held in the application, no window, and the check and the effect are the same statement. A unique constraint on `(show_id, seat_id, active)` is the backstop that makes double-booking impossible rather than unlikely, including against the code path somebody adds later.

**And what I would say about the failure:** the alternative — a distributed lock per seat — is worse, because it introduces a lock service on the critical path, needs lease expiry, and still does not protect against the code path that forgets to take the lock. The database's own atomicity is the cheapest correct primitive, and the constraint is what makes it a guarantee.

### Q36. ⚡ BookMyShow — the hold expires while the user is paying. What happens? — H ⚡

**A:** This is the real question, and the honest answer is that **the hold and the payment are two systems with no shared transaction**, so something has to give, and the design decision is which.

The mechanism: a hold has `expires_at`. A sweeper releases expired holds so the seats return to inventory. The problem is the window where a user is in the payment provider's flow, has paid, and the hold expires — either the sweeper releases seats the user is about to claim, or the user pays for seats that are gone.

The resolution I would present, in order:

1. **Make the hold long enough and confirm synchronously where possible.** A ten-minute hold with a payment that takes thirty seconds is a hold that never expires mid-payment in practice; make the payment initiation part of the hold, so the hold is *extended* on payment start rather than left to expire.
2. **Make the release conditional, not time-based alone.** The sweeper does not release a hold that has an in-flight payment; it marks it `EXPIRING` and the payment's callback resolves it. That is a state machine, and it is the correct shape because the set of outcomes is finite: paid (confirm), failed (release), timed out (release), unknown (reconcile).
3. **Reconcile rather than decide.** The payment provider's webhook is the authority. A payment that succeeded for a hold that was released triggers a refund-and-notify path, which is a bad outcome but a *defined* one — and defining it is the answer. The failure mode you cannot accept is undefined: seats sold twice, or money taken with no seats and nobody noticing.

**Strong:** "There is no design without this window. The design question is which failure you choose and whether you have written it down. I would choose 'refund and apologise' over 'double-book', because one is a support ticket and the other is a broken venue."

### Q37. Elevator — why is this a state machine and not a scheduler? — M

**A:** Because the hard part is not choosing the next floor, it is **the set of states the car can be in and which transitions are legal**. Moving, doors-open, stopped-at-floor, idle — and the invariants are what break: you cannot open the doors while moving, you cannot start moving with the doors open, and a request must not be lost while the car transitions between them.

Once the machine exists, the scheduling policy is a **strategy** over it — the classic SCAN/elevator algorithm, or nearest-first, or a fair round-robin for a building where the top floor complains. That separation is the design: `Elevator` owns state and transitions, `SchedulingPolicy` chooses the next target. Confusing the two produces the code where the dispatch algorithm is threaded through the door logic, and the door logic is the part with safety requirements.

**Probe: how do you handle a request arriving while the doors are open?** It goes into the machine as an event, not a direct mutation of the target list — which is why the requests are a queue the machine consumes rather than a field the policy writes. That is the same reason the notification decision service writes a row before enqueueing: an event that only exists inside a running method is an event that can be lost in a transition.

### Q38. Chess — how do you model the pieces, and why not a `switch`? — M

**A:** A `Piece` abstraction with a `moves(board, from) -> list[Square]` method and one subclass per piece, because the movement rule is the thing that varies and it varies by *kind*. A `switch(piece.type)` in a `moveValidator` means every new variant (a fairy piece, a custom rule set) edits the validator, and the validator becomes the file everybody conflicts on.

The subtlety that makes this a real modelling question rather than a pattern demonstration: **movement and legality are different things.** A rook's *movement* is "any square along a rank or file"; its *legality* requires that the path is unblocked, that it does not leave the king in check, and in some rules that it is not castling through an attacked square. So the piece returns candidate squares and a separate rule layer filters them — because "does this leave my king in check" is a property of the *position*, not of the piece, and putting it on the piece would force every piece to know about the king.

**That split is the answer.** Most candidates put both on the piece and then discover that the king's own move depends on every other piece on the board.

### Q39. Chess — undo, and how do you not re-derive it? — M

**A:** If a move is a **value** — `Move(from_square, to_square, piece, captured, promotion)` — then undo is not a reverse algorithm, it is the inverse of a small, known mutation: put the piece back, put the captured piece back, restore the flags. That is a Memento where the memento is tiny, and the reason it works is that a chess move changes very little: two squares, plus castling rights and an en-passant target and a halfmove clock.

The trap is deriving undo from the rules — recomputing what the board must have looked like — which breaks the moment a rule has an exception, and chess is a game made of exceptions. **Store the delta.** The state that a move changes is small enough to store whole, and storing it whole means undo is O(1) and provably correct rather than reasoned about.

The extension to say out loud: this is also how you get move history for free, which the UI wants, and how you rewind to an arbitrary ply rather than one step — the mementos are already a list.

### Q40. ATM — the state machine and the two-phase dispense. — H

**A:** Two separate things, and both are the interesting part.

**The state machine** is `IDLE → CARD_INSERTED → PIN_ENTERED → TRANSACTION_SELECTED → DISPENSING → IDLE` with an explicit `EJECTING`/`OUT_OF_SERVICE` side. The design point is that illegal transitions are *refused by the machine*, not by the caller remembering: you cannot select a transaction before the PIN verifies, and the machine is what says so. And the state has to be persisted, because the machine restarts.

**The two-phase dispense** is the part worth getting to. Dispensing cash is irreversible and the balance debit is not, so the order is: (1) debit the account with an idempotency key, (2) dispense, (3) record the result. If step 2 fails after step 1 succeeded, the correct behaviour is to *reverse* the debit — and if step 2 succeeds and the process dies before step 3, the reconciliation job must find a debit with no dispense record and, because the cash is gone, mark it dispensed rather than refunding. Which side you fail towards is a **business decision**: give the customer the money and eat the loss, or take the money and have an angry customer. It is not a technical choice and I would say that.

**Strong:** "The irreversibility is what makes this hard. Every design question here is 'which side of an irreversible action do I put the reversible one', and the answer is always: do the reversible one first, and make it idempotent so you can retry it."

### Q41. Rate limiter — token bucket versus sliding window versus fixed window. — M

**A:**

| Algorithm | How | The failure it has |
|---|---|---|
| **Fixed window** | Count per calendar window | The boundary burst: 100 requests at 11:59:59 and 100 at 12:00:00 is 200 in two seconds against a limit of 100/minute |
| **Sliding log** | Timestamps per request, count in the trailing window | Exact, and memory proportional to the request rate — usually too expensive |
| **Sliding window counter** | Weighted sum of the current and previous window | Approximates the sliding window with two counters; the standard production choice |
| **Token bucket** | Tokens refill at a rate, each request takes one | Allows bursts up to the bucket size — which is usually the *desired* behaviour and is why it is the default |
| **Leaky bucket** | Queue drains at a fixed rate | Smooths output to a constant rate; adds latency by design |

**The design point I would make:** the choice depends on whether bursts are acceptable. A token bucket with capacity 100 and refill 10/sec permits a burst of 100 and then a sustained 10/sec — for a notification system that is exactly right, and for a payment API it is not. And the return type matters as much as the algorithm: returning a **wait time** rather than a boolean is what lets the caller distinguish "delay this" from "reject this", and the OTP case shows why that distinction has to be made by the caller and not by the limiter.

**And the distributed version**, which is the follow-up: a per-node limiter multiplies the limit by the node count, so the counter has to be shared — Redis with an atomic increment and a TTL, or a lease where each node takes a slice of the global budget. The second is faster and less exact, and it is what most large systems do.

### Q42. Logger — levels, handlers and why async logging loses messages. — M

**A:** The structure: the **logger** is what the caller talks to and it decides *whether* to emit based on level; the **appender/handler** is where the record goes and there can be several; the **formatter** turns a record into text; and the **level** is a threshold that is checked as early as possible so a disabled debug line costs a comparison rather than a string concatenation.

**Why async logging loses messages** is the part worth getting to: an async appender puts records on a queue and returns immediately, so the caller's latency drops — and now the queue is a buffer that is lost on a crash, has a capacity that must be bounded (or the process OOMs when the disk stalls), and must define what happens when it fills. The three choices are block (the app stops when logging backs up, which is why async logging exists), drop (silent data loss, and the lost records are exactly the ones from the incident), and drop-with-a-counter (the right answer, and it must be a counter somebody watches).

**Strong:** "The design decision is not sync versus async, it is what happens when the log sink is slower than the application. If that answer is not written down, the answer at 3 a.m. is whatever the library defaults to."

**And the operational point I would add:** never log a level that is only checked after the expensive part is built. If the disabled path still formats the message, the level filter is decoration.

### Q43. ⚡ Pub-sub — what delivery guarantee do you actually provide? — H ⚡

**A:** **At-least-once, with deduplication, and I would say the whole sentence.** Exactly-once delivery between two processes is not achievable without a transaction spanning both, and a broker that claims it is either deduplicating for you or lying. At-most-once is achievable and is what you get by acknowledging before processing — fast, and it loses messages on a crash.

At-least-once plus a deduplicating consumer is the design that actually works, and the two halves matter equally: the broker must not acknowledge until the work is durable (which is what makes a crash re-deliver rather than lose), and the consumer must be idempotent (which is what makes a re-delivery a no-op rather than a duplicate effect). Neither half alone is a guarantee — a broker that redelivers into a non-idempotent consumer is a duplicate generator.

The consequence to state: **the consumer's idempotency key has to come from the message, not from the delivery**, because the whole point is that two deliveries of the same event produce the same key. That is exactly why the notification design's dedupe key is `event_type:entity_id` — two services emitting `order.shipped` for the same order produce the same key, and one of them is discarded.

**Probe: how do you handle ordering?** Per-partition. A partition is an ordered log and a consumer group assigns each partition to one consumer, so ordering is per-key and the key is the partition key. Global ordering costs a single consumer, which costs throughput — the same trade as Q23, arrived at from the broker side.

### Q44. ⚡ Food delivery — who assigns the order to a driver? — H ⚡

**A:** The question behind the question is **push versus pull**, and the answer is that it depends on who has the better information and who bears the cost of a bad match.

**Push** (the platform assigns) optimises globally: it can consider all available drivers, prefer the one already heading that way, batch two orders into one trip, and guarantee an order is never unassigned. It costs a dispatcher that must know every driver's state, and it makes the driver an input rather than a participant — which matters, because drivers reject assignments and a rejected push is a wasted round trip.

**Pull** (drivers accept offers) optimises locally and handles the real world better: the driver knows the traffic, the building, and whether the tip is worth it. It costs an offer protocol (offer, timeout, cascade to the next driver), and it makes assignment latency a function of the cascade depth — which is where the customer-visible delay lives.

The design I would present is a **hybrid with an explicit offer cascade**: the platform computes a ranked shortlist from global state (proximity, direction, current load, acceptance history) and pushes to the top candidate with a short timeout, cascading down the list. That gets global information into the ranking and local knowledge into the acceptance, and the timeout bounds the latency.

**And the failure mode is the real content:** the cascade must be idempotent per order, because a timeout that fires while the driver's acceptance is in flight produces two assignments for one order. The fix is the same atomic claim as everywhere else — `assign(order_id, driver_id)` succeeds once, and the second attempt loses — plus an explicit `ASSIGNMENT_PENDING` state so the order is never in the hands of two drivers and never in the hands of none.

---

## F. Machine coding and process (Q45–Q50)

### Q45. How do you start a machine-coding round? — M

**A:** With five minutes of requirements clarification and a written list, because the alternative is twenty minutes of building the wrong thing.

The four things I always pin down: **the core operations** (which two or three methods does everything else serve?), **the one requirement that is likely to grow an "and also"** (that is where the seam goes), **the scale** (a few hundred items in memory, or millions — which changes whether the design needs an index), and **the explicitly-out-of-scope list** (stating what I am not building is what stops the interviewer thinking I forgot).

Then I write, in this order: the entities, the interfaces, the main flow as a method with the steps commented, then the implementations, then tests. Writing the main flow as commented steps before implementing is the single highest-value habit — it is where the interviewer sees the design rather than the typing, and it is where I catch the ordering problem (like "the dropped notification must not consume rate-limit budget") while it is still a comment.

**Probe: what if they say "just start coding"?** I start coding the entities, and I narrate the two assumptions I am making as I make them. The clarification is not a phase, it is a thing that continues.

### Q46. Do you write tests in a machine-coding round? — M

**A:** Yes, and I write **assertion-style checks with printed labels** rather than a test framework, because the framework is setup time and the assertions are the signal. A `check("label", condition)` helper that prints PASS or FAIL gives the interviewer a visible, runnable demonstration of every claim, and it takes ten lines.

What I test, in priority order: the **happy path** (it works at all), the **one boundary** the problem is actually about (the half-open interval in a booking, the strict cursor comparison in a feed, the exactly-once claim), and **two failures** (what happens on the invalid input, what happens on the concurrent one). That is five checks and it covers the ground an interviewer cares about.

**Strong:** "The tests are also how I find my own design bugs. In the notification bridge, the deferral arithmetic across a daylight-saving transition was wrong — the naive version landed an hour early — and the test is what surfaced it. A claim I cannot execute is a claim I should not make."

### Q47. You have fifteen minutes left and the design is half built. What do you do? — M

**A:** Stop adding features and make what exists **complete and coherent**. Concretely, in this order:

1. **Make the main path run end to end**, even if it is narrow. A working happy path with stubs elsewhere is worth more than five half-built features, because it demonstrates the design holds together.
2. **Write the checks for what runs.** Three passing checks beats a wide surface with nothing verified.
3. **Say what is missing and where it would go.** "The concurrency story for `due()` is not implemented; here is the two-column change" is a complete answer. Silence reads as not knowing.
4. **Do not start anything new.** A half-finished class is worse than an absent one, because it makes the existing design look uncertain.

**Probe: what if the interviewer pushes you to add one more thing?** Then I add it in the smallest form that demonstrates the seam — the whole point of having a seam is that the addition is small, and demonstrating that is the highest-value use of the remaining minutes.

### Q48. How do you name things? — E

**A:** Names carry the design, and there are four rules I actually follow.

**Domain names, not pattern names.** `Ticket`, `Spot`, `Reservation` — not `ParkingLotManager`, `SpotFactory`, `TicketHelper`. A class named after a pattern is a class whose purpose nobody wrote down.

**No "Manager", "Helper", "Util" or "Processor".** They mean "I did not decide what this is". The notification design has `DecisionService`, `Sender`, `Dispatcher`, `WebhookHandler` — each one is a verb-shaped noun that says what it does.

**Booleans read as assertions.** `is_deleted`, `has_seen`, `terminal`, `accepted` — so `if not outcome.terminal` reads as a sentence. `flag` and `status` read as nothing.

**The unit goes in the name when the unit is not obvious.** `ttl_s`, `retry_after_s`, `refill_per_s`, `dedupe_ttl_s`. A bare `timeout` in a codebase where three units are in play is a bug waiting for a Friday.

**And the one I would add:** if a name needs a comment to explain it, the name is wrong; if a *method* needs a comment to explain its ordering, the comment is load-bearing and should be there. That distinction is worth stating, because it is the difference between a codebase that documents its names and one that documents its decisions.

### Q49. ⚡ Do you apply patterns as you go, or refactor into them? — H ⚡

**A:** **I name the seam and I write the concrete thing**, and I introduce the abstraction at the moment the second implementation appears — with two exceptions where I go abstract immediately.

The exceptions: a **dependency on an external system** goes behind an injected parameter from the first line, because the payoff is testability rather than flexibility and it is immediate — a provider callable and a clock make the whole thing runnable in milliseconds with scripted failures. And a **value that crosses a boundary** gets frozen from the start, because retrofitting immutability after callers have learned to mutate it is a refactor across every call site.

Everything else — a pricing policy, an allocator, a scheduling strategy — I write concretely, because the first implementation is also the requirements document. Writing `HourlyPricing` tells me what the policy interface actually needs to pass, and an interface designed before any implementation tends to be either too wide (methods nobody uses) or too narrow (missing the parameter the second case needs). The refactor from one concrete class to an interface plus two implementations is mechanical; the reverse — unwinding a speculative interface — is not.

**Strong:** "The pattern question I would rather answer is 'what change does this make cheap', and if I cannot name the change, I would rather say the design is concrete here and explain where I would cut. A pattern applied without a named change is a guess wearing a name."

### Q50. How do you handle a requirement you do not understand? — M

**A:** I say so, and I say what I *would* assume, and I keep building. That is three moves and each one matters.

Saying so is not a weakness; pretending is a much larger risk, because a misunderstanding discovered at minute forty makes the design unrecoverable. Saying what I would assume means the interview keeps moving and the interviewer can correct the assumption with one sentence. Keeping building means the round does not stall on a discussion.

The phrasing I use: "I do not know how the venue's allocation works. I am going to assume a booking holds a specific seat for a specific show, and I will put the allocation behind a method so if it is actually per-section, only that method changes." That sentence does four things — it admits the gap, names an assumption, puts it behind a seam, and demonstrates the design principle at the same time.

**Probe: what if it is a domain you do not know at all — say, a dialysis scheduling system?** Then I ask about the *one* constraint that shapes everything. For dialysis it is that a patient needs the same slot three times a week and the machine must be cleaned between uses — and those two facts determine the entire model. Asking for the shaping constraint is a better question than asking for the requirements list, because the list is long and the constraint is short.

---

## G. Pushback round

These are the challenges that come *after* your design, when the interviewer stops asking and starts pushing. The answer to every one of them has the same three parts: **acknowledge, quantify, and offer the change.**

### P1. "Why not just put all of this in the database?"

**A:** For most of it, I would — and the honest split is worth stating. The authoritative state (notifications, tickets, seats, holds) belongs in the database, and in a real system these classes are repositories over tables rather than in-memory stores. What does *not* belong in the database, and what this design puts in memory, is the state that is per-process and ephemeral: a token bucket's refill state, an in-process idempotency cache, a dedupe window. Asking the database on every rate-limit check adds a round trip to the hot path for state that is allowed to be approximate.

So the answer is that the design has both, and which is which is a decision I would want to defend: **durable state in the database, approximate and hot state in memory, and the boundary between them is where the failure tolerance changes.** A lost token bucket is a few seconds of over-limit; a lost notification row is a notification nobody can audit.

### P2. "Why is there no Singleton for the config / the store / the pool?"

**A:** Because it would make the tests impossible and hide a real dependency. Every store here is a constructor parameter, which means the signature documents what the class needs, a test can pass a fake, and two instances can coexist — which the demo actually does, constructing several worlds with different limits in one process.

The "exactly one" property is real and it belongs to the **composition root**: the process builds one instance and passes it to everyone. That is where the lifetime decision is true and where it can be changed for a canary or a migration. Putting it on the class makes it a fact about the type instead of a fact about the wiring, and those are different things that should not be conflated.

### P3. "Just retry everything on failure — simplify it."

**A:** That is simpler and it fails in two directions at once, and being able to name both is the answer. A retryable provider error retried forever burns quota on a provider limit shared with the password resets. A **terminal** error retried five times is pure cost — and worse, an invalid device token retried until the budget runs out means the token is never deleted, so every future notification to that user fails too.

That is why the classification is a table rather than an `if`: the *default* for an unknown code is retryable, because an uncatalogued transient error classified as terminal silently drops a notification nothing will ever notice, while an uncatalogued terminal error retried a bounded number of times lands noisily in the DLQ. Prefer the failure that is loud. And the third outcome, `SUPPRESS_TOKEN`, exists precisely because "retry" and "drop" are both wrong for a dead destination.

### P4. "Isn't this over-engineered for a parking lot?"

**A:** Probably, and I would say which parts I would cut first. If the requirement is one lot with 200 spots and no dynamic pricing, then `PricingPolicy` as an interface is a file with one implementation and I would delete it — the flat hourly computation goes on the ticket. What I would *keep* is the atomic spot claim, because that is a correctness property under concurrency and it costs one method; and the clock as a parameter, because without it nothing is testable.

That is the honest shape of the answer: **name the parts that are load-bearing and the parts that are speculative, and cut the speculative ones yourself.** An interviewer asking this question is usually checking whether you can tell the difference, and the answer that fails is defending everything.

### P5. "Why write this yourself? There's a library for it."

**A:** For a production system I would use the library, and I would say which one. `cachetools` for an LRU, a real rate-limiter library, `zoneinfo` for timezones instead of the hand-written resolver in the demo. Reimplementing infrastructure is how you get a cache with a subtle eviction bug.

The reason it is in the exercise is that the interview is asking whether I know what the library does. The DST bug in the notification bridge is the perfect illustration: the naive deferral arithmetic — replace the hour, convert back — is an hour wrong on the fall-back day, and knowing *why* is what makes me trust `zoneinfo` rather than assume it. Writing the thirty-line version once is what makes the library a choice rather than a black box.

### P6. "You have twenty classes for a parking lot. Isn't that too many?"

**A:** I would count the ones doing work and the ones doing structure. Entities: `Ticket`, `Spot`, `Vehicle` as a value — three. `SpotStore`, `TicketStore` — two. Two seams, `PricingPolicy` and `SpotAllocator`. One facade. That is eight, and the rest of the box is values and enums, which are cheap and comparable and hashable.

The question I would answer is not "how many classes" but **"how many of these would I have to open to answer a question about the design?"** If the answer is eight, the design is eight classes and the count is not the metric. And if the count is genuinely high, the honest answer is to say which two I would merge.

### P7. "Your design doesn't handle X." (where X is something real you missed)

**A:** Ask what X costs. Then do one of three things, in this order:

1. **If it is a small change behind an existing seam**, say so and describe the change in one sentence. "That is a new `PricingPolicy`, and it does not touch anything else." This is the best outcome and it is what the seam was for.
2. **If it is a real gap**, say so plainly, say what it would cost, and say where it would go. "The dispatcher has no lease, so two workers can send the same row twice. It is two columns and a visibility timeout." Naming it is worth more than a confident wrong answer, and much more than pretending the question was unclear.
3. **If it is genuinely out of scope**, say that, and say what would have to change for it to be in scope. "Ordering is not guaranteed; getting it means partitioning by user id and the throughput cost that implies."

**The move that never works** is arguing that X does not matter. The interviewer chose X because they think it does, and disagreeing about the requirement is a different conversation than the one being had.

### P8. "Why is your interface so small? There's no `getStatus` or `cancel`."

**A:** Because a method nobody calls is a method whose contract nobody has tested. Every method on an interface is a promise that every implementation has to keep, and an implementation that keeps it with `raise NotImplementedError` is documenting what it is *not*.

The test I use: for each method, name a caller. `Channel.send` has one — the sender. `AnalyticsSink.record` has one. When the second caller arrives, the method gets added then, and it will be added with the requirements of a real second caller instead of a guess. That is the same reasoning as Q49: an interface designed before its second implementation tends to be too wide or too narrow, and too wide is the failure that leaves dead methods in every implementation.

---

## If the interviewer says X → answer Y

| They say | You say |
|---|---|
| "Walk me through the classes" | "The entities are … and the seams are … — I will start with the flow through the facade and then the classes it touches" |
| "Is it thread-safe?" | "The check-and-effect pairs are single operations; no lock is held across I/O; and here is the race I have *not* closed" |
| "How would you extend it?" | Name the seam, name the change it makes cheap, and name one change it does **not** accommodate |
| "What if the requirements change?" | "I would put it behind X, because X is the sentence most likely to grow an 'and also'" |
| "Why not a Singleton?" | "Test isolation, hidden dependencies, and 'exactly one' belongs to the wiring, not the type" |
| "Why not just retry?" | "Retryable, terminal and suppress-token are three outcomes; retrying a dead token burns shared capacity and never deletes it" |
| "Isn't that over-engineered?" | Cut the speculative parts yourself and name the load-bearing ones: the atomic claim and the injected clock stay |
| "Why not use a library?" | Name the library you would use, and say what writing it once taught you about what it does |
| "You have too many classes" | "How many would I have to open to answer a design question? Eight — the rest are values and enums" |
| "It doesn't handle X" | "That is a new `XP`olicy behind an existing seam" / "That is a real gap: two columns and a timeout" / "Out of scope, and here is what changes that" |
| "How do you test it?" | "Time and the external system are inputs — a manual clock and a scripted provider — so the whole suite runs in milliseconds" |
| "What did you not do?" | Name the two gaps before they find them; a named gap reads as judgement, a found gap reads as an oversight |
| "Any questions?" | "What is the one requirement you think I have not accounted for?" — the best closing question in an LLD round |

---

## Cross-links

**The LLD problems, worked in full:**

- [Parking lot](../LLD/problems/01-parking-lot.md) · [LRU/LFU cache](../LLD/problems/02-lru-lfu-cache.md) · [Hotel management](../LLD/problems/03-hotel-management.md) · [BookMyShow seat locking](../LLD/problems/04-bookmyshow-seat-locking.md) · [Elevator system](../LLD/problems/05-elevator-system.md) · [Splitwise](../LLD/problems/06-splitwise.md) · [ATM machine](../LLD/problems/07-atm-machine.md) · [Library management](../LLD/problems/08-library-management.md) · [Chess game](../LLD/problems/09-chess-game.md) · [Vending machine](../LLD/problems/10-vending-machine.md) · [Rate limiter](../LLD/problems/11-rate-limiter.md) · [Logger framework](../LLD/problems/12-logger-framework.md) · [Pub-sub system](../LLD/problems/13-pub-sub-system.md) · [Tic-tac-toe](../LLD/problems/14-tic-tac-toe.md) · [Snake and ladder](../LLD/problems/15-snake-and-ladder.md) · [Food delivery](../LLD/problems/16-food-delivery.md)

**The bridges — HLD systems taken one level down:**

- [URL shortener LLD](../Case-Studies/LLD/01-url-shortener-lld.md) · [WhatsApp LLD](../Case-Studies/LLD/02-whatsapp-lld.md) · [Uber matching LLD](../Case-Studies/LLD/03-uber-matching-lld.md) · [Instagram feed LLD](../Case-Studies/LLD/04-instagram-feed-lld.md) · [Notification system LLD](../Case-Studies/LLD/05-notification-system-lld.md)

**The HLDs those bridges descend from:** [URL shortener](../Case-Studies/HLD/01-url-shortener.md) · [WhatsApp](../Case-Studies/HLD/02-whatsapp-chat-system.md) · [Notification system](../Case-Studies/HLD/03-notification-system.md) · [Instagram news feed](../Case-Studies/HLD/04-instagram-news-feed.md) · [Uber ride-hailing](../Case-Studies/HLD/06-uber-ride-hailing.md) · [BookMyShow seat booking](../Case-Studies/HLD/07-bookmyshow-seat-booking.md)

**The revision sheet:** [LLD cheat sheet](../Cheat-Sheets/07-lld-cheatsheet.md)

**Other question banks in this series:** [Foundations and capacity](01-foundations-and-capacity-questions.md) · [Networking and API](02-networking-and-api-questions.md) · [Scaling, caching and DB](03-scaling-caching-db-questions.md) · [LLM, RAG and GenAI](04-llm-rag-genai-questions.md) · [Agentic AI](05-agentic-ai-questions.md) · [LLMOps and operationalisation](06-llmops-and-operationalisation-questions.md) · [Behavioural and project](07-behavioral-and-project-questions.md)

---

## Grounding note

**What is grounded.** The cross-links above are real files in this repository, and every problem named in Part E has a full worked file at the link given. The techniques referenced in the answers — the atomic claim, the classification table, the reserved bucket, the version-pinned template, the DST deferral bug, the maximum-versus-minimum aggregate — are implementations that exist in the bridges and, in the notification case, are executed by 68 passing assertions.

**What is not grounded.** The reference repositories this knowledge base was assembled from contain **no OOP or LLD material at all** — no pattern catalogue, no class diagrams, no machine-coding transcripts. The source material names "low-level design" and the classic problems without teaching them. Every model answer here is **constructed**, reasoned from first principles and from the implementations in the bridge files. Where an answer states a number (a bucket capacity, a retry schedule, a timeout) it is taken from the HLD's tuning table or from a bridge's code, not from a cited external source.

**How to use it honestly.** Treat Parts A–D and F as drill: say the answers out loud, because the difference between knowing SOLID and *saying* SOLID under time pressure is the thing being tested. Treat Part E as an index into the worked files rather than a substitute for them — one paragraph cannot teach a seat-locking design that needs 600 lines. And treat the ⚡ questions as the ones worth rehearsing twice: they are the ones where a correct answer and a senior answer are different answers.

