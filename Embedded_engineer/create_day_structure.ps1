
# Function to process a phase
function Process-Phase {
    param (
        [string]$OutlineFile,
        [string]$PhaseDir,
        [int]$StartDay,
        [int]$EndDay,
        [string]$TemplateFile
    )

    Write-Host "Processing Phase: $PhaseDir"
    if (-not (Test-Path $OutlineFile)) {
        Write-Error "Outline file not found: $OutlineFile"
        return
    }
    if (-not (Test-Path $TemplateFile)) {
        Write-Error "Template file not found: $TemplateFile"
        return
    }

    $content = Get-Content $OutlineFile -Raw
    $template = Get-Content $TemplateFile -Raw
    
    # Split by "#### **Day" to separate days.
    $days = $content -split "#### \*\*Day"
    
    $phaseName = Split-Path $PhaseDir -Leaf

    foreach ($dayBlock in $days) {
        # Match "1: Title**" or "121: Title**"
        if ($dayBlock -match "^\s*(\d+):\s*(.*?)\*\*") {
            $dayNum = [int]$matches[1]
            $dayTitle = $matches[2].Trim()

            if ($dayNum -ge $StartDay -and $dayNum -le $EndDay) {
                # Calculate Week
                $weekNum = [math]::Ceiling($dayNum / 7)
                
                $weekDir = Join-Path $PhaseDir "Week_$weekNum"
                if (-not (Test-Path $weekDir)) {
                    New-Item -ItemType Directory -Path $weekDir -Force | Out-Null
                }

                $dayDir = Join-Path $weekDir "Day_$dayNum"
                if (-not (Test-Path $dayDir)) {
                    New-Item -ItemType Directory -Path $dayDir -Force | Out-Null
                }

                # Extract Topics, Sections, Labs
                $topics = ""
                $sections = ""
                $labs = ""

                if ($dayBlock -match "\*\*Topics:\*\*\r?\n([\s\S]*?)(?=\*\*Sections:|\*\*Labs:|$)") {
                    $topics = $matches[1].Trim()
                }
                if ($dayBlock -match "\*\*Sections:\*\*\r?\n([\s\S]*?)(?=\*\*Labs:|$)") {
                    $sections = $matches[1].Trim()
                }
                if ($dayBlock -match "\*\*Labs:\*\*\r?\n([\s\S]*?)(?=$|####)") {
                    $labs = $matches[1].Trim()
                }

                # Generate contents.md content using sequential Replace to avoid line continuation issues
                $mdContent = $template.Replace("__DAY_NUM__", "$dayNum")
                $mdContent = $mdContent.Replace("__DAY_TITLE__", $dayTitle)
                $mdContent = $mdContent.Replace("__PHASE_NAME__", $phaseName)
                $mdContent = $mdContent.Replace("__WEEK_NUM__", "$weekNum")
                $mdContent = $mdContent.Replace("__TOPICS__", $topics)
                $mdContent = $mdContent.Replace("__SECTIONS__", $sections)
                $mdContent = $mdContent.Replace("__LABS__", $labs)
                
                $targetFile = Join-Path $dayDir "contents.md"
                Set-Content -Path $targetFile -Value $mdContent
                Write-Host "Created: $targetFile"
            }
        }
    }
}

$templatePath = "g:\My Drive\Codes & Repos\Embedded_engineer\day_content_template.txt"

# Execute for Phase 1
Process-Phase -OutlineFile "g:\My Drive\Codes & Repos\Embedded_engineer\Phase_1_Core_Foundations\Phase_1_Course_Outline.md" -PhaseDir "g:\My Drive\Codes & Repos\Embedded_engineer\Phase_1_Core_Foundations" -StartDay 1 -EndDay 120 -TemplateFile $templatePath

# Execute for Phase 2
Process-Phase -OutlineFile "g:\My Drive\Codes & Repos\Embedded_engineer\Phase_2_Linux_Kernel_Drivers\Phase_2_Course_Outline.md" -PhaseDir "g:\My Drive\Codes & Repos\Embedded_engineer\Phase_2_Linux_Kernel_Drivers" -StartDay 121 -EndDay 260 -TemplateFile $templatePath

# Execute for Phase 3
Process-Phase -OutlineFile "g:\My Drive\Codes & Repos\Embedded_engineer\Phase_3_Camera_Systems_ISP\Phase_3_Course_Outline.md" -PhaseDir "g:\My Drive\Codes & Repos\Embedded_engineer\Phase_3_Camera_Systems_ISP" -StartDay 261 -EndDay 350 -TemplateFile $templatePath
