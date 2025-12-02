$root = "g:\My Drive\Codes & Repos\Embedded_engineer\Phase_3_Camera_Systems_ISP"
$weekOffset = 22
$dayOffset = 170

# Iterate through the NEW weeks (16 to 28)
for ($w = 16; $w -le 28; $w++) {
    $weekPath = Join-Path $root "Week_$w"
    if (Test-Path $weekPath) {
        $oldWeekNum = $w + $weekOffset
        Write-Host "Fixing Week_$w (was $oldWeekNum)"

        $files = Get-ChildItem -Path $weekPath -Recurse -Filter "*.md"
        foreach ($file in $files) {
            $content = Get-Content $file.FullName -Raw
            $originalContent = $content

            # Fix Week References
            # "Week 46" -> "Week 24"
            $content = $content -replace "Week $oldWeekNum", "Week $w"
            $content = $content -replace "Week: $oldWeekNum", "Week: $w"
            
            # Fix Day References
            # We need to be careful not to replace "Day 1" with "Day 100" if we just do simple replace.
            # But here we are replacing LARGE numbers (261+) with SMALL numbers (91+), so it should be safe.
            # Iterate backwards to avoid collision? No, 261 doesn't contain 91.
            
            # We need to find which Day this file belongs to, to know the specific mapping?
            # Or just replace ALL occurrences of old day numbers in the range?
            # Replacing all is safer for "Next: Day X" references.
            
            # Range of old days for this week?
            # Actually, let's just loop through the days that *could* be in this file.
            # The file might refer to previous/next days.
            
            # Optimization: Just loop through all days 261..350 and replace with 91..180
            # But do it in a way that doesn't overlap.
            # 261 -> 91. 
            
            for ($d = 261; $d -le 350; $d++) {
                $newD = $d - $dayOffset
                # Use regex word boundaries to avoid replacing "Day 2610"
                $content = $content -replace "\bDay $d\b", "Day $newD"
                $content = $content -replace "\bDay_$d\b", "Day_$newD"
                $content = $content -replace "\bLab $d\b", "Lab $newD"
            }

            if ($content -ne $originalContent) {
                Write-Host "  Updating $($file.Name)"
                Set-Content -Path $file.FullName -Value $content
            }
        }
    }
}
