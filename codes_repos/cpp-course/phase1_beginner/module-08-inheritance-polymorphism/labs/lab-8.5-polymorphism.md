# Lab 8.5: Virtual Functions and Polymorphism

## Objective
Enable runtime polymorphism using `virtual` functions.

## Instructions

### Step 1: Virtual Function
Create `polymorphism.cpp`. Add `virtual` to the base method.

```cpp
#include <iostream>

class Animal {
public:
    virtual void speak() { std::cout << "Animal sound\n"; }
};

class Dog : public Animal {
public:
    void speak() override { std::cout << "Woof!\n"; }
};

class Cat : public Animal {
public:
    void speak() override { std::cout << "Meow!\n"; }
};
```

### Step 2: Polymorphic Call
Use a pointer to Base.

```cpp
int main() {
    Animal* a1 = new Dog();
    Animal* a2 = new Cat();
    
    a1->speak(); // Woof! (Dynamic Binding)
    a2->speak(); // Meow!
    
    delete a1;
    delete a2;
    return 0;
}
```

### Step 3: The `override` Keyword
Always use `override` in derived classes. It asks the compiler to check if you are actually overriding something.
`void speek() override;` // Error: Typo in name, compiler catches it!

## Challenges

### Challenge 1: Array of Pointers
Create `Animal* zoo[2];`
Assign a Dog and a Cat. Loop through and call `speak()`.

### Challenge 2: Non-Virtual Override?
Remove `virtual` from Base. Run Step 2 again.
It prints "Animal sound" twice.
Verify that `override` keyword generates an error if base is not virtual.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <vector>

class Animal {
public:
    virtual void speak() { std::cout << "Generic sound\n"; }
    virtual ~Animal() {} // Important!
};

class Dog : public Animal {
public:
    void speak() override { std::cout << "Woof\n"; }
};

class Cat : public Animal {
public:
    void speak() override { std::cout << "Meow\n"; }
};

int main() {
    std::vector<Animal*> zoo;
    zoo.push_back(new Dog());
    zoo.push_back(new Cat());
    
    for(auto a : zoo) {
        a->speak();
    }
    
    for(auto a : zoo) delete a;
    
    return 0;
}
```
</details>

## Success Criteria
✅ Implemented virtual function
✅ Used `override` keyword
✅ Achieved dynamic dispatch via pointers
✅ Created polymorphic collection (Challenge 1)

## Key Learnings
- `virtual` enables Dynamic Dispatch (VTable)
- `override` ensures correctness
- Polymorphism works via Pointers or References, not Values

## Next Steps
Proceed to **Lab 8.6: Abstract Classes** to define interfaces.
