# Lab 7.10: Bank Account System (Capstone)

## Objective
Build a mini-banking system using classes, encapsulation, vectors, and static members.

## Instructions

### Step 1: Account Class
Create `bank_system.cpp`.
- Private: `accountNumber` (int), `balance` (double), `ownerName` (string).
- Public: Constructor, `deposit`, `withdraw`, `display`.
- Static: `nextAccountNumber` (auto-increment).

```cpp
#include <iostream>
#include <string>
#include <vector>

class Account {
    static int nextId;
    int id;
    std::string owner;
    double balance;
public:
    Account(std::string name, double initialDep) 
        : owner(name), balance(initialDep) {
        id = nextId++;
    }
    
    int getId() const { return id; }
    
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
    
    void display() const {
        std::cout << "Acc " << id << " (" << owner << "): $" << balance << std::endl;
    }
};

int Account::nextId = 1000;
```

### Step 2: Bank Class
Manages a collection of accounts.
- `std::vector<Account> accounts`
- `createAccount(name, deposit)`
- `findAccount(id)`

```cpp
class Bank {
    std::vector<Account> accounts;
public:
    void createAccount(std::string name, double deposit) {
        Account newAcc(name, deposit);
        accounts.push_back(newAcc);
        std::cout << "Created account " << newAcc.getId() << std::endl;
    }
    
    void listAccounts() const {
        for (const auto& acc : accounts) {
            acc.display();
        }
    }
};
```

### Step 3: Main Loop
Create a menu to Add Account, List Accounts, Quit.

## Challenges

### Challenge 1: Transfer
Add `transfer(fromId, toId, amount)` to `Bank`.
You'll need to find both accounts (by reference!) and call withdraw/deposit.

### Challenge 2: Pointer Storage
Change `vector<Account>` to `vector<Account*>`.
Use `new` to create accounts.
Remember to `delete` them in `Bank`'s destructor!

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <string>
#include <vector>

class Account {
    static int nextId;
    int id;
    std::string owner;
    double balance;
public:
    Account(std::string name, double initialDep) 
        : owner(name), balance(initialDep) {
        id = nextId++;
    }
    int getId() const { return id; }
    void deposit(double amount) { balance += amount; }
    bool withdraw(double amount) {
        if (amount <= balance) { balance -= amount; return true; }
        return false;
    }
    void display() const {
        std::cout << "ID: " << id << " | " << owner << " | $" << balance << std::endl;
    }
};
int Account::nextId = 1000;

class Bank {
    std::vector<Account*> accounts;
public:
    ~Bank() {
        for (auto acc : accounts) delete acc;
    }
    
    void createAccount(std::string name, double deposit) {
        accounts.push_back(new Account(name, deposit));
    }
    
    Account* findAccount(int id) {
        for (auto acc : accounts) {
            if (acc->getId() == id) return acc;
        }
        return nullptr;
    }
    
    void transfer(int fromId, int toId, double amount) {
        Account* src = findAccount(fromId);
        Account* dest = findAccount(toId);
        if (src && dest && src->withdraw(amount)) {
            dest->deposit(amount);
            std::cout << "Transfer successful.\n";
        } else {
            std::cout << "Transfer failed.\n";
        }
    }
    
    void list() const {
        for (auto acc : accounts) acc->display();
    }
};

int main() {
    Bank bank;
    bank.createAccount("Alice", 1000);
    bank.createAccount("Bob", 500);
    
    bank.list();
    bank.transfer(1000, 1001, 200);
    bank.list();
    
    return 0;
}
```
</details>

## Success Criteria
✅ Implemented Account class with encapsulation
✅ Implemented Bank class with vector
✅ Used static member for ID generation
✅ Implemented money transfer logic (Challenge 1)

## Key Learnings
- Real-world modeling with classes
- Managing collections of objects
- Inter-object communication (transfer)

## Next Steps
Congratulations! You've completed Module 7.

Proceed to **Module 8: Inheritance and Polymorphism** to unlock the full power of OOP.
