# PowerShell script to create all Phase 7 folder structure
# 30 weeks, 210 days (7 days per week)

$basePath = "g:\My Drive\Codes & Repos\Embedded_engineer\Phase_7_Parallel_Programming"

Write-Host "Creating Phase 7 folder structure..." -ForegroundColor Cyan

$dayCounter = 1

for ($week = 1; $week -le 30; $week++) {
    $weekFolder = Join-Path $basePath ("Week_{0:D2}" -f $week)
    
    # Create week folder
    New-Item -Path $weekFolder -ItemType Directory -Force | Out-Null
    Write-Host "Created: Week_$($week.ToString('00'))" -ForegroundColor Green
    
    # Create 7 day folders for this week
    for ($day = 1; $day -le 7; $day++) {
        $dayFolder = Join-Path $weekFolder ("Day_{0:D3}" -f $dayCounter)
        New-Item -Path $dayFolder -ItemType Directory -Force | Out-Null
        Write-Host "  Created: Day_$($dayCounter.ToString('000'))" -ForegroundColor Gray
        $dayCounter++
    }
}

Write-Host "`nFolder structure creation complete!" -ForegroundColor Cyan
Write-Host "Total weeks created: 30" -ForegroundColor Yellow
Write-Host "Total day folders created: 210" -ForegroundColor Yellow
