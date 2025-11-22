$basePath = "H:\My Drive\Codes & Repos\codes_repos\devops-course"

$phases = @(
    @{
        Name    = "phase1_beginner"
        Modules = @(
            @{ Num = "01"; Target = "module-01-introduction-devops" },
            @{ Num = "02"; Target = "module-02-linux-fundamentals" },
            @{ Num = "03"; Target = "module-03-version-control-git" },
            @{ Num = "04"; Target = "module-04-networking-basics" },
            @{ Num = "05"; Target = "module-05-docker-fundamentals" },
            @{ Num = "06"; Target = "module-06-cicd-basics" },
            @{ Num = "07"; Target = "module-07-infrastructure-as-code-intro" },
            @{ Num = "08"; Target = "module-08-configuration-management" },
            @{ Num = "09"; Target = "module-09-monitoring-logging-basics" },
            @{ Num = "10"; Target = "module-10-cloud-fundamentals-aws" }
        )
    },
    @{
        Name    = "phase2_intermediate"
        Modules = @(
            @{ Num = "11"; Target = "module-11-advanced-docker" },
            @{ Num = "12"; Target = "module-12-kubernetes-fundamentals" },
            @{ Num = "13"; Target = "module-13-advanced-cicd" },
            @{ Num = "14"; Target = "module-14-infrastructure-as-code-advanced" },
            @{ Num = "15"; Target = "module-15-configuration-management-advanced" },
            @{ Num = "16"; Target = "module-16-monitoring-observability" },
            @{ Num = "17"; Target = "module-17-logging-log-management" },
            @{ Num = "18"; Target = "module-18-security-compliance" },
            @{ Num = "19"; Target = "module-19-database-operations" },
            @{ Num = "20"; Target = "module-20-cloud-architecture-patterns" }
        )
    },
    @{
        Name    = "phase3_advanced"
        Modules = @(
            @{ Num = "21"; Target = "module-21-advanced-kubernetes" },
            @{ Num = "22"; Target = "module-22-gitops-argocd" },
            @{ Num = "23"; Target = "module-23-serverless-functions" },
            @{ Num = "24"; Target = "module-24-advanced-monitoring" },
            @{ Num = "25"; Target = "module-25-chaos-engineering" },
            @{ Num = "26"; Target = "module-26-multi-cloud-hybrid" },
            @{ Num = "27"; Target = "module-27-platform-engineering" },
            @{ Num = "28"; Target = "module-28-cost-optimization" },
            @{ Num = "29"; Target = "module-29-incident-management" },
            @{ Num = "30"; Target = "module-30-production-deployment" }
        )
    }
)

foreach ($phase in $phases) {
    $phasePath = Join-Path $basePath $phase.Name
    Write-Host "Processing Phase: $($phase.Name)"

    foreach ($mod in $phase.Modules) {
        $targetName = $mod.Target
        $targetPath = Join-Path $phasePath $targetName
        $moduleNum = $mod.Num

        # Ensure Target Exists
        if (-not (Test-Path $targetPath)) {
            New-Item -ItemType Directory -Force -Path $targetPath | Out-Null
            Write-Host "  Created Target: $targetName"
        }

        # Find all folders starting with "module-XX"
        $candidates = Get-ChildItem -Path $phasePath -Directory | Where-Object { $_.Name -match "^module-$moduleNum" -and $_.Name -ne $targetName }

        foreach ($folder in $candidates) {
            Write-Host "  Merging: $($folder.Name) -> $targetName"
            
            # Move Labs
            $sourceLabs = Join-Path $folder.FullName "labs"
            $targetLabs = Join-Path $targetPath "labs"
            
            if (Test-Path $sourceLabs) {
                if (-not (Test-Path $targetLabs)) {
                    New-Item -ItemType Directory -Force -Path $targetLabs | Out-Null
                }
                
                $files = Get-ChildItem -Path $sourceLabs
                foreach ($file in $files) {
                    $destFile = Join-Path $targetLabs $file.Name
                    if (Test-Path $destFile) {
                        # Rename if conflict
                        $newName = "{0}_extra{1}" -f $file.BaseName, $file.Extension
                        $destFile = Join-Path $targetLabs $newName
                        Write-Host "    Conflict! Renaming to $newName"
                    }
                    Move-Item -Path $file.FullName -Destination $destFile
                }
            }

            # Move README and other root files
            $rootFiles = Get-ChildItem -Path $folder.FullName -File
            foreach ($file in $rootFiles) {
                $destFile = Join-Path $targetPath $file.Name
                if (Test-Path $destFile) {
                    # If README exists in target, keep target (it's likely the good one). 
                    # Rename source to README_extra.md just in case.
                    $newName = "{0}_extra{1}" -f $file.BaseName, $file.Extension
                    $destFile = Join-Path $targetPath $newName
                }
                Move-Item -Path $file.FullName -Destination $destFile
            }

            # Remove empty source folder
            Remove-Item -Path $folder.FullName -Recurse -Force
        }
    }
}
Write-Host "Consolidation Complete!"
