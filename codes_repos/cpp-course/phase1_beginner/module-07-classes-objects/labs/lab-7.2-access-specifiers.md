# Lab 7.2: Access Specifiers

## Objective
Understand `public` and `private` to control access to class members.

## Instructions

### Step 1: Private Members
Create `access.cpp`.

```cpp
#include <iostream>

class BankAccount {
private:
    double balance; // Hidden data
public:
    void deposit(double amount) {
        if (amount > 0) balance += amount;
    }
    
    double getBalance() {
        return balance;
    }
};

int main() {
    BankAccount account;
    // account.balance = 1000; // Error: 'balance' is private
    
    account.deposit(100);
    std::cout << "Balance: " << account.getBalance() << std::endl;
    
    return 0;
}
```

### Step 2: Why Private?
Try to set balance to a negative number via `deposit`. The logic prevents it.
If `balance` were public, anyone could set it to `-1000000`.

### Step 3: Protected (Preview)
`protected` is like private, but accessible to derived classes (inheritance). We'll cover this in Module 8.

## Challenges

### Challenge 1: Withdraw Function
Add a `withdraw(double amount)` function.
- Check if `amount > 0`.
- Check if `amount <= balance`.
- Return `true` if successful, `false` otherwise.

### Challenge 2: Default Access
Remove `private:` label. Is `balance` still private?
Yes, because `class` defaults to private.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>

class BankAccount {
    double balance = 0.0; // Private by default
public:
    void deposit(double amount) {
        if (amount > 0) balance += amount;
    }
    
    bool withdraw(double amount) {
        if (amount > 0 && amount <= balance) {
            balance -= amount;
            return true;
        }
        return false;
    }
    
    double getBalance() { return balance; }
};

int main() {
    BankAccount acc;
    acc.deposit(500);
    if (acc.withdraw(200)) std::cout << "Withdrew 200\n";
    else std::cout << "Failed\n";
    
    std::cout << "Remaining: " << acc.getBalance() << std::endl;
    return 0;
}
```
</details>

## Success Criteria
✅ Used `private` to hide data
✅ Used `public` to expose interface
✅ Implemented validation logic in public methods
✅ Understood default access for class

## Key Learnings
- Encapsulation protects data integrity
- Only expose what is necessary
- Use public methods to manipulate private data safely

## Next Steps
Proceed to **Lab 7.3: Constructors** to initialize objects properly.
