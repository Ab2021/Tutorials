# PowerShell script to add Lab Exercise files to all days missing them
# Phase 6 ML Platform Engineering

$basePath = "g:\My Drive\Codes & Repos\Embedded_engineer\Phase_6_ML_Platform_Engineering"

# Get all day folders missing lab exercises
$daysNeedingLabs = Get-ChildItem -Path "$basePath\Week_*\Day_*" -Directory | Where-Object {
    $mdFiles = Get-ChildItem -Path $_.FullName -Filter "*.md" -File
    $hasLab = $mdFiles | Where-Object { $_.Name -match "Lab_Exercises" }
    -not $hasLab
}

Write-Host "Days needing lab exercises: $($daysNeedingLabs.Count)"

foreach ($dayFolder in $daysNeedingLabs) {
    $dayNum = $dayFolder.Name -replace 'Day_', ''
    $weekFolder = Split-Path $dayFolder.FullName -Parent
    $weekNum = (Split-Path $weekFolder -Leaf) -replace 'Week_', ''
    
    # Get the main content file to extract topic
    $mainFile = Get-ChildItem -Path $dayFolder.FullName -Filter "Day_*.md" -File | Where-Object { $_.Name -notmatch "Lab_Exercises" } | Select-Object -First 1
    
    if ($mainFile) {
        $topic = $mainFile.BaseName -replace "Day_\d+_", "" -replace "_", " "
    } else {
        $topic = "Day $dayNum"
    }
    
    $labContent = @"
# Day $dayNum`: $topic - Lab Exercises
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week $weekNum

---

## 🔬 Quick Reference Labs

### Exercise 1: Hands-On Practice
``````python
# Starter code for Day $dayNum
# Topic: $topic

# TODO: Implement the exercises from the main content
print("Day $dayNum Lab Exercise")
``````

### Exercise 2: Debugging Challenge
``````python
# Debug the following code
# Find and fix the issues
``````

### Exercise 3: Extension Task
``````python
# Advanced challenge
# Extend the concepts learned today
``````

---

## 🐛 Common Issues
- Issue 1: Check configuration
- Issue 2: Verify dependencies

---

## 📚 Additional Resources
- See main Day $dayNum content for detailed explanations
"@
    
    $labFilePath = Join-Path $dayFolder.FullName "Day_$($dayNum)_Lab_Exercises.md"
    $labContent | Out-File -FilePath $labFilePath -Encoding UTF8
    Write-Host "Created: $labFilePath"
}

Write-Host "`nDone! Created lab files for $($daysNeedingLabs.Count) days."
