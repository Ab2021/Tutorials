# Lab 20.10: Pattern Combination (Capstone)

## Objective
Combine multiple design patterns to build a complete application.

## Project: Document Editor with Patterns

Build a document editor using:
- Factory (create documents)
- Observer (notify views)
- Command (undo/redo)
- Decorator (add features)
- Strategy (save formats)

## Implementation

```cpp
#include <iostream>
#include <memory>
#include <vector>
#include <stack>
#include <string>

// Document (Subject + Product)
class Document : public std::enable_shared_from_this<Document> {
    std::string content;
    std::vector<std::weak_ptr<class DocumentView>> views;
    
public:
    void setContent(const std::string& c) {
        content = c;
        notifyViews();
    }
    
    std::string getContent() const { return content; }
    
    void attach(std::shared_ptr<DocumentView> view) {
        views.push_back(view);
    }
    
    void notifyViews();
};

// Observer
class DocumentView {
public:
    virtual ~DocumentView() = default;
    virtual void update(const std::string& content) = 0;
};

void Document::notifyViews() {
    for (auto& wp : views) {
        if (auto sp = wp.lock()) {
            sp->update(content);
        }
    }
}

// Command pattern
class Command {
public:
    virtual ~Command() = default;
    virtual void execute() = 0;
    virtual void undo() = 0;
};

class InsertTextCommand : public Command {
    std::shared_ptr<Document> doc;
    std::string text;
    std::string previousContent;
    
public:
    InsertTextCommand(std::shared_ptr<Document> d, const std::string& t)
        : doc(d), text(t) {}
    
    void execute() override {
        previousContent = doc->getContent();
        doc->setContent(previousContent + text);
    }
    
    void undo() override {
        doc->setContent(previousContent);
    }
};

// Command manager
class CommandManager {
    std::stack<std::unique_ptr<Command>> undoStack;
    std::stack<std::unique_ptr<Command>> redoStack;
    
public:
    void executeCommand(std::unique_ptr<Command> cmd) {
        cmd->execute();
        undoStack.push(std::move(cmd));
        // Clear redo stack
        while (!redoStack.empty()) redoStack.pop();
    }
    
    void undo() {
        if (!undoStack.empty()) {
            auto cmd = std::move(undoStack.top());
            undoStack.pop();
            cmd->undo();
            redoStack.push(std::move(cmd));
        }
    }
    
    void redo() {
        if (!redoStack.empty()) {
            auto cmd = std::move(redoStack.top());
            redoStack.pop();
            cmd->execute();
            undoStack.push(std::move(cmd));
        }
    }
};

// Strategy pattern for saving
class SaveStrategy {
public:
    virtual ~SaveStrategy() = default;
    virtual void save(const std::string& content, const std::string& filename) = 0;
};

class TextSaveStrategy : public SaveStrategy {
public:
    void save(const std::string& content, const std::string& filename) override {
        std::cout << "Saving as text: " << filename << "\n";
    }
};

class HTMLSaveStrategy : public SaveStrategy {
public:
    void save(const std::string& content, const std::string& filename) override {
        std::cout << "Saving as HTML: " << filename << "\n";
    }
};

// Factory
class DocumentFactory {
public:
    static std::shared_ptr<Document> createDocument(const std::string& type) {
        return std::make_shared<Document>();
    }
};

// Concrete view
class ConsoleView : public DocumentView {
public:
    void update(const std::string& content) override {
        std::cout << "View updated: " << content << "\n";
    }
};

int main() {
    // Create document
    auto doc = DocumentFactory::createDocument("text");
    
    // Attach view
    auto view = std::make_shared<ConsoleView>();
    doc->attach(view);
    
    // Command manager
    CommandManager cmdMgr;
    
    // Execute commands
    cmdMgr.executeCommand(std::make_unique<InsertTextCommand>(doc, "Hello "));
    cmdMgr.executeCommand(std::make_unique<InsertTextCommand>(doc, "World!"));
    
    // Undo
    cmdMgr.undo();
    
    // Redo
    cmdMgr.redo();
    
    // Save with strategy
    std::unique_ptr<SaveStrategy> strategy = std::make_unique<HTMLSaveStrategy>();
    strategy->save(doc->getContent(), "document.html");
    
    return 0;
}
```

## Success Criteria
✅ Combined multiple patterns
✅ Implemented complete application
✅ Demonstrated pattern synergy

## Key Learnings
- Patterns work together naturally
- Each pattern solves specific problem
- Combination creates flexible architecture

## Module 20 Complete!
You've mastered design patterns in C++.
