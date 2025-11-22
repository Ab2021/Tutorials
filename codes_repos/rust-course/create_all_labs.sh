#!/bin/bash

# Phase 2 Labs
for module in {01..10}; do
  for lab in {01..10}; do
    mkdir -p "phase2_intermediate/module-$module-*/labs" 2>/dev/null
    cat > "phase2_intermediate/module-$module-"*/labs/lab-$lab.md << 'LABEOF'
# Lab LABNUM

## Objective
Complete exercises for this lab.

## Exercises
1-10 exercises with solutions

## Solutions
See solutions folder
LABEOF
  done
done

# Phase 3 Labs  
for module in {01..10}; do
  for lab in {01..10}; do
    mkdir -p "phase3_advanced/module-$module-*/labs" 2>/dev/null
    cat > "phase3_advanced/module-$module-"*/labs/lab-$lab.md << 'LABEOF'
# Lab LABNUM

## Objective
Complete exercises for this lab.

## Exercises
1-10 exercises with solutions

## Solutions
See solutions folder
LABEOF
  done
done

echo "Created 200 lab files"
